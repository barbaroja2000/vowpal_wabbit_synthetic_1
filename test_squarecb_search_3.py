import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from vowpalwabbit import pyvw
from datetime import datetime
import time
import gc
import json
from collections import deque
import multiprocessing as mp
from functools import partial
import csv
import threading

# Define user types, times of day, and actions (same as original)
user_types = ["high_roller", "casual_player", "sports_enthusiast", "newbie"]
times_of_day = ["morning", "afternoon", "evening"]
actions = ["slots_heavy", "live_casino", "sports_betting", "mixed_games", "promotional"]

# Base mean rewards for each user-type and action combination
base_mean_reward = {
    "high_roller": {
        "slots_heavy": 50,
        "live_casino": 200,
        "sports_betting": 10,
        "mixed_games": 30,
        "promotional": 20
    },
    "casual_player": {
        "slots_heavy": 30,
        "live_casino": 15,
        "sports_betting": 5,
        "mixed_games": 10,
        "promotional": 10
    },
    "sports_enthusiast": {
        "slots_heavy": 10,
        "live_casino": 20,
        "sports_betting": 100,
        "mixed_games": 20,
        "promotional": 5
    },
    "newbie": {
        "slots_heavy": 15,
        "live_casino": 10,
        "sports_betting": 5,
        "mixed_games": 10,
        "promotional": 30
    }
}

user_type_time_multiplier = {
    "high_roller": {"morning": 0.9, "afternoon": 1.0, "evening": 1.5},
    "casual_player": {"morning": 1.0, "afternoon": 0.8, "evening": 1.2},
    "sports_enthusiast": {"morning": 0.7, "afternoon": 1.3, "evening": 1.3},
    "newbie": {"morning": 0.8, "afternoon": 1.0, "evening": 1.4}
}

param_grid = {
    'gamma': [5.0, 15.0, 30.0, 40.0, 50.0],          # Expanded to include higher values up to 50.0
    'learning_rate': [0.1, 0.5, 1.0, 1.5, 2.0],      # Added higher values up to 2.0
    'initial_t': [0.5, 1.0, 3.0, 5.0, 8.0],          # Expanded range from 0.5 to 8.0
    'power_t': [0.1, 0.3, 0.5, 0.7, 0.9],            # Expanded to include 0.3 and more values
    'noise_sigma': [0.05]      # More granular search with additional values
}

# Output file
RESULTS_FILE = 'context_aware_hyperparameter_search_results.csv'
CHECKPOINT_FILE = 'context_aware_hyperparameter_search_checkpoint.json'
MAX_ITERATIONS = 5000  # Reduced for faster processing with parallelization
EVAL_INTERVAL = 250  # Increased interval for better performance

# Core functions from original code
def get_cost(context, action, noise_sigma=0.1):
    user_type = context['user_type']
    time_of_day = context['time_of_day']
    base_mu = base_mean_reward[user_type][action]
    multiplier = user_type_time_multiplier[user_type][time_of_day]
    mu = base_mu * multiplier
    sigma = noise_sigma * mu
    noise = np.random.normal(0, 1)
    noise = max(min(noise, 2.0), -2.0)  # Limit noise to ±2.0 SD
    reward = max(0, min(mu + sigma * noise, mu * 1.3))  # Cap reward
    cost = -reward
    return cost

def to_vw_example_format(context, actions, cb_label=None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared |UserType user_type={} time_of_day={}\n".format(context["user_type"], context["time_of_day"])
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += "0:{}:{} ".format(cost, prob)
        example_string += "|Action layout={} \n".format(action)
    return example_string[:-1]

def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1 / total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if sum_prob > draw:
            return index, prob

def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context, actions)
    pmf = vw.predict(vw_text_example)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob

def choose_user(users):
    return random.choice(users)

def choose_time_of_day(times_of_day):
    return random.choice(times_of_day)

# Evaluation metrics
def calculate_time_sensitivity(vw, user_types, times_of_day, actions, num_samples=50):
    """
    Measure how differently the model behaves across time periods for the same user_type.
    Higher score means greater time sensitivity.
    Optimized with fewer samples for better performance in parallel processing.
    """
    time_sensitivity_score = 0
    
    # Only use a subset of user types to speed up calculation
    sampled_user_types = random.sample(user_types, min(2, len(user_types)))
    
    for user_type in sampled_user_types:
        time_distributions = {}
        
        for time in times_of_day:
            action_counts = {action: 0 for action in actions}
            
            for _ in range(num_samples):
                context = {'user_type': user_type, 'time_of_day': time}
                action, _ = get_action(vw, context, actions)
                action_counts[action] += 1
                
            # Normalize to get a probability distribution
            total = sum(action_counts.values())
            time_distributions[time] = {action: count/total for action, count in action_counts.items()}
        
        # Calculate Jensen-Shannon divergence between time periods
        for t1, t2 in itertools.combinations(times_of_day, 2):
            dist1 = [time_distributions[t1][a] for a in actions]
            dist2 = [time_distributions[t2][a] for a in actions]
            
            # Simple distance measure between distributions
            distance = sum(abs(p1 - p2) for p1, p2 in zip(dist1, dist2)) / 2.0
            time_sensitivity_score += distance
    
    # Adjust normalization for the subset of user types
    return time_sensitivity_score / (len(sampled_user_types) * len(list(itertools.combinations(times_of_day, 2))))

def run_experiment(params, seed=None):
    """Run a single experiment with given hyperparameters with memory optimization"""
    # Set a unique seed if none provided
    if seed is None:
        seed = int(time.time() * 1000) % 100000 + os.getpid()
    
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    
    # Extract parameters
    gamma = params['gamma']
    learning_rate = params['learning_rate']
    initial_t = params['initial_t']
    power_t = params['power_t']
    noise_sigma = params['noise_sigma']
    
    # Initialize VW
    vw_args = f"--cb_explore_adf -q UA -q TA --quiet --squarecb --gamma {gamma} -l {learning_rate} --initial_t {initial_t} --power_t {power_t} --cb_type mtr --normalize"
    vw = pyvw.Workspace(vw_args)
    
    # Track metrics (only keep final values to save memory)
    cost_sum = 0.0
    last_ctr = 0.0
    
    # Run iterations
    for i in range(1, MAX_ITERATIONS + 1):
        # Sample context
        user_type = choose_user(user_types)
        time_of_day = choose_time_of_day(times_of_day)
        context = {'user_type': user_type, 'time_of_day': time_of_day}
        
        # Get action
        action, prob = get_action(vw, context, actions)
        
        # Get cost and update
        cost = get_cost(context, action, noise_sigma)
        cost_sum += cost
        
        # Learn
        vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)), pyvw.Workspace.lContextualBandit)
        vw.learn(vw_format)
        
        # Calculate final CTR
        if i == MAX_ITERATIONS:
            last_ctr = -1 * cost_sum / i
    
    # Calculate time sensitivity at end of experiment
    time_sensitivity = calculate_time_sensitivity(vw, user_types, times_of_day, actions)
    
    # Calculate context-specific performance
    context_performance = calculate_context_specific_performance(vw, user_types, times_of_day, actions)
    
    # Clean up
    vw.finish()
    del vw
    gc.collect()  # Force garbage collection
    
    # Return minimal result structure to save memory
    return {
        'final_ctr': last_ctr,
        'time_sensitivity': time_sensitivity,
        'context_coverage': context_performance['context_coverage'],
        'average_regret': context_performance['average_regret'],
        'max_regret': context_performance['max_regret'],
        'worst_context': context_performance['worst_context'][0],  # Just store the key to save memory
        'ctr_history': [],  # Empty to save memory
        'iterations': []    # Empty to save memory
    }

def get_completed_runs():
    """Get a set of parameter combinations that have already been run"""
    if not os.path.exists(RESULTS_FILE):
        return set()
    
    try:
        df = pd.read_csv(RESULTS_FILE)
        # Convert parameters to tuple for hashable comparison
        completed = set()
        for _, row in df.iterrows():
            param_tuple = (row['gamma'], row['learning_rate'], row['initial_t'], 
                          row['power_t'], row['noise_sigma'])
            completed.add(param_tuple)
        return completed
    except:
        return set()

def save_checkpoint(current_idx, grid):
    """Save current progress to resume later"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'current_idx': current_idx, 'total': len(grid)}, f)

def load_checkpoint():
    """Load checkpoint if it exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'current_idx': 0, 'total': 0}

def process_param_combination(param_tuple, param_keys, completed_runs, result_lock):
    """Process a single parameter combination in a separate process"""
    try:
        # Convert tuple to dict of parameters
        params = {key: param_tuple[i] for i, key in enumerate(param_keys)}
        
        # Skip if already done
        if param_tuple in completed_runs:
            return None
        
        # Print status with minimal output to avoid buffer issues
        print(f"Running: gamma={params['gamma']}, lr={params['learning_rate']}")
        
        # Time the experiment
        start_time = time.time()
        
        # Run experiment with memory optimization
        result = run_experiment(params)
        
        # Measure elapsed time
        elapsed = time.time() - start_time
        
        # Extract only the necessary results to save memory
        row = {
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'initial_t': params['initial_t'],
            'power_t': params['power_t'],
            'noise_sigma': params['noise_sigma'],
            'final_ctr': result['final_ctr'],
            'time_sensitivity': result['time_sensitivity'],
            'context_coverage': result['context_coverage'],
            'average_regret': result['average_regret'],
            'max_regret': result['max_regret'],
            'worst_context': result['worst_context'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Write to CSV with lock to prevent race conditions
        with result_lock:
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)
        
        print(f"Completed: gamma={params['gamma']}, lr={params['learning_rate']} in {elapsed:.2f}s")
        
        # Force cleanup
        result.clear()
        del result
        gc.collect()
        
        return row
    
    except Exception as e:
        print(f"Error processing parameters: {param_tuple}")
        print(f"Exception: {str(e)}")
        return None

def run_grid_search(num_processes=None):
    """Run hyperparameter grid search with multiprocessing and memory optimization"""
    # Generate all parameter combinations
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    total_combinations = len(param_combinations)
    print(f"Total parameter combinations: {total_combinations}")
    
    # Get already completed runs
    completed_runs = get_completed_runs()
    print(f"Found {len(completed_runs)} already completed runs")
    completed_count = len(completed_runs)
    
    # Define result columns to ensure consistency
    result_columns = [
        'gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma',
        'final_ctr', 'time_sensitivity', 'context_coverage', 'average_regret', 
        'max_regret', 'worst_context', 'timestamp'
    ]
    
    # Filter out already completed runs
    param_combinations_to_run = [pc for pc in param_combinations if pc not in completed_runs]
    print(f"Combinations remaining to run: {len(param_combinations_to_run)}")
    
    # Create results file if it doesn't exist
    if not os.path.exists(RESULTS_FILE):
        # Create empty dataframe with columns
        df = pd.DataFrame(columns=result_columns)
        df.to_csv(RESULTS_FILE, index=False)
    else:
        # Check if existing file has all necessary columns
        existing_df = pd.read_csv(RESULTS_FILE)
        missing_columns = [col for col in result_columns if col not in existing_df.columns]
        
        if missing_columns:
            print(f"Warning: Results file missing columns: {missing_columns}")
            print("Creating a new results file with the correct format.")
            
            # Create a new file with correct columns, preserving existing results if possible
            new_df = pd.DataFrame(columns=result_columns)
            
            # Copy data from existing columns
            common_columns = [col for col in result_columns if col in existing_df.columns]
            if common_columns:
                for col in common_columns:
                    if col in existing_df.columns:
                        new_df[col] = existing_df[col]
            
            # Save the new format
            new_df.to_csv(RESULTS_FILE, index=False)
    
    # If no combinations to run, we're done
    if not param_combinations_to_run:
        print("All parameter combinations already completed.")
        return
    
    # Determine the number of processes to use (default to 3)
    if num_processes is None:
        num_processes = 3
    print(f"Running with {num_processes} parallel processes")
    
    # Create a progress tracking mechanism
    progress_lock = threading.Lock()
    progress_count = mp.Value('i', 0)
    
    def track_progress(result):
        if result is not None:  # Skip None results (already completed runs)
            with progress_lock:
                progress_count.value += 1
                processed = completed_count + progress_count.value
                percent = processed / total_combinations * 100
                print(f"\rProgress: {processed}/{total_combinations} ({percent:.1f}%)", end="", flush=True)
    
    # Process in smaller batches to manage memory better
    BATCH_SIZE = num_processes * 3  # Process 3 batches per process at a time
    
    # Use a Manager object to create a shareable lock
    with mp.Manager() as manager:
        result_lock = manager.Lock()
        
        # Process combinations in batches to control memory usage
        for batch_start in range(0, len(param_combinations_to_run), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(param_combinations_to_run))
            batch = param_combinations_to_run[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1}/{(len(param_combinations_to_run)+BATCH_SIZE-1)//BATCH_SIZE}")
            
            # Create a partial function with fixed parameters
            process_func = partial(process_param_combination, 
                                param_keys=param_keys, 
                                completed_runs=completed_runs,
                                result_lock=result_lock)
            
            # Run batch in parallel
            with mp.Pool(processes=num_processes) as pool:
                async_results = []
                for param_combination in batch:
                    async_result = pool.apply_async(process_func, (param_combination,), callback=track_progress)
                    async_results.append(async_result)
                
                # Wait for all processes in this batch to complete
                for async_result in async_results:
                    async_result.wait()
            
            # Force garbage collection between batches
            gc.collect()
    
    print("\nGrid search completed!")

def analyze_results():
    """Analyze results and find best hyperparameter combinations"""
    if not os.path.exists(RESULTS_FILE):
        print("No results file found.")
        return
    
    df = pd.read_csv(RESULTS_FILE)
    
    print(f"Analyzed {len(df)} experiment runs")
    
    # Check which columns exist in the dataframe
    context_aware_columns = ['context_coverage', 'average_regret', 'max_regret', 'worst_context']
    has_context_metrics = all(col in df.columns for col in context_aware_columns)
    
    if not has_context_metrics:
        print("\nWARNING: Results file does not contain context-specific metrics.")
        print("Only showing basic CTR and time sensitivity metrics.")
    
    # Top 10 by CTR
    print("\nTop 10 parameter combinations by CTR:")
    top_ctr = df.sort_values('final_ctr', ascending=False).head(10)
    print(top_ctr[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 'final_ctr']])
    
    # Top 10 by time sensitivity
    print("\nTop 10 parameter combinations by time sensitivity:")
    top_time = df.sort_values('time_sensitivity', ascending=False).head(10)
    print(top_time[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 'time_sensitivity']])
    
    # Context-specific metrics if available
    if has_context_metrics:
        # Top 10 by context coverage
        print("\nTop 10 parameter combinations by context coverage:")
        top_coverage = df.sort_values('context_coverage', ascending=False).head(10)
        print(top_coverage[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 'context_coverage']])
        
        # Best combinations with low regret
        print("\nTop 10 parameter combinations by lowest average regret:")
        top_regret = df.sort_values('average_regret', ascending=True).head(10)
        print(top_regret[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 'average_regret', 'max_regret']])
    
    # Calculate correlation between parameters and metrics
    print("\nCorrelation between parameters and metrics:")
    correlation_cols = ['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 
                         'final_ctr', 'time_sensitivity']
    if has_context_metrics:
        correlation_cols.extend(['context_coverage', 'average_regret', 'max_regret'])
    
    correlation = df[correlation_cols].corr()
    
    if has_context_metrics:
        print(correlation[['final_ctr', 'time_sensitivity', 'context_coverage', 'average_regret']])
    else:
        print(correlation[['final_ctr', 'time_sensitivity']])
    
    # Create a context-aware composite score
    # Normalize all scores first
    df['norm_ctr'] = (df['final_ctr'] - df['final_ctr'].min()) / (df['final_ctr'].max() - df['final_ctr'].min())
    df['norm_time'] = (df['time_sensitivity'] - df['time_sensitivity'].min()) / (df['time_sensitivity'].max() - df['time_sensitivity'].min())
    
    # Original composite score
    df['original_composite_score'] = 0.7 * df['norm_ctr'] + 0.3 * df['norm_time']
    
    # Context-aware score if metrics are available
    if has_context_metrics:
        df['norm_coverage'] = (df['context_coverage'] - df['context_coverage'].min()) / (df['context_coverage'].max() - df['context_coverage'].min())
        
        # Invert regret scores since lower is better
        if df['average_regret'].max() != df['average_regret'].min():
            df['norm_avg_regret'] = 1 - (df['average_regret'] - df['average_regret'].min()) / (df['average_regret'].max() - df['average_regret'].min())
        else:
            df['norm_avg_regret'] = 1.0
        
        if df['max_regret'].max() != df['max_regret'].min():
            df['norm_max_regret'] = 1 - (df['max_regret'] - df['max_regret'].min()) / (df['max_regret'].max() - df['max_regret'].min())
        else:
            df['norm_max_regret'] = 1.0
        
        # Balanced composite score with more weight on context-specific metrics
        df['context_aware_score'] = (
            0.3 * df['norm_ctr'] + 
            0.1 * df['norm_time'] + 
            0.2 * df['norm_coverage'] + 
            0.25 * df['norm_avg_regret'] + 
            0.15 * df['norm_max_regret']
        )
        
        print("\nTop 10 parameter combinations by context-aware composite score:")
        print("(30% CTR, 10% time sensitivity, 20% context coverage, 40% regret metrics)")
        top_context_aware = df.sort_values('context_aware_score', ascending=False).head(10)
        print(top_context_aware[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 
                             'final_ctr', 'time_sensitivity', 'context_coverage', 'average_regret', 'context_aware_score']])
    
    # Original composite score for comparison
    print("\nTop 10 parameter combinations by original composite score (70% CTR, 30% time sensitivity):")
    top_original = df.sort_values('original_composite_score', ascending=False).head(10)
    print(top_original[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 
                      'final_ctr', 'time_sensitivity', 'original_composite_score']])
    
    # Visualizations
    if has_context_metrics:
        # Visualize parameter effects with context-aware composite score
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['learning_rate'], df['gamma'], 
                             c=df['context_aware_score'], cmap='viridis', 
                             s=50, alpha=0.7)
        plt.colorbar(scatter, label='Context-Aware Score')
        plt.xlabel('Learning Rate')
        plt.ylabel('Gamma (Exploration)')
        plt.title('Effect of Learning Rate and Gamma on Context-Aware Performance')
        plt.savefig('context_aware_param_effects.png')
        
        # Gamma vs power_t colored by context coverage
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['gamma'], df['power_t'], 
                             c=df['context_coverage'], cmap='plasma', 
                             s=50, alpha=0.7)
        plt.colorbar(scatter, label='Context Coverage')
        plt.xlabel('Gamma (Exploration)')
        plt.ylabel('Power T (Decay Exponent)')
        plt.title('Effect of Gamma and Power T on Context Coverage')
        plt.savefig('context_aware_param_effects_coverage.png')
        
        # Learning rate vs initial_t colored by average regret
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['learning_rate'], df['initial_t'], 
                             c=df['average_regret'], cmap='coolwarm_r', 
                             s=50, alpha=0.7)
        plt.colorbar(scatter, label='Average Regret (Lower is Better)')
        plt.xlabel('Learning Rate')
        plt.ylabel('Initial T')
        plt.title('Effect of Learning Rate and Initial T on Average Regret')
        plt.savefig('context_aware_param_effects_regret.png')
    
    # Original visualizations - always show these
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['learning_rate'], df['gamma'], 
                         c=df['final_ctr'], cmap='viridis', 
                         s=50, alpha=0.7)
    plt.colorbar(scatter, label='CTR')
    plt.xlabel('Learning Rate')
    plt.ylabel('Gamma (Exploration)')
    plt.title('Effect of Learning Rate and Gamma on CTR')
    plt.savefig('context_aware_param_effects_ctr.png')
    
    return df

def calculate_context_specific_performance(vw, user_types, times_of_day, actions, num_samples=30):
    """
    Evaluate how well the model performs for each specific context (user_type and time_of_day combination)
    Returns:
    - Per-context performance metrics
    - Average regret across contexts
    - Worst-performing context
    - Context coverage score (% of contexts with good performance)
    """
    # Store results for each context
    context_performance = {}
    context_regret = {}
    
    for user_type in user_types:
        for time in times_of_day:
            context = {'user_type': user_type, 'time_of_day': time}
            
            # Find optimal action for this context (lowest expected cost)
            optimal_action = min(actions, key=lambda a: get_expected_cost(context, a))
            optimal_reward = -get_expected_cost(context, optimal_action)
            
            # Sample model recommendations for this context
            action_counts = {action: 0 for action in actions}
            total_reward = 0
            
            for _ in range(num_samples):
                action, _ = get_action(vw, context, actions)
                action_counts[action] += 1
                reward = -get_expected_cost(context, action)
                total_reward += reward
            
            # Calculate metrics
            avg_reward = total_reward / num_samples
            regret = optimal_reward - avg_reward
            accuracy = action_counts[optimal_action] / num_samples if optimal_action in action_counts else 0
            
            # Store results
            context_key = f"{user_type}_{time}"
            context_performance[context_key] = {
                'optimal_action': optimal_action,
                'avg_reward': avg_reward,
                'optimal_reward': optimal_reward,
                'accuracy': accuracy, 
                'regret': regret
            }
            context_regret[context_key] = regret
    
    # Calculate aggregate metrics
    average_regret = sum(context_regret.values()) / len(context_regret)
    worst_context_key, worst_regret = max(context_regret.items(), key=lambda x: x[1])
    # Context coverage: % of contexts with regret better than 1.5x average
    context_coverage = sum(1 for r in context_regret.values() if r < average_regret * 1.5) / len(context_regret)
    
    return {
        'context_performance': context_performance,
        'average_regret': average_regret,
        'worst_context': (worst_context_key, worst_regret),
        'context_coverage': context_coverage,
        'max_regret': worst_regret
    }

def get_expected_cost(context, action):
    """Calculate expected cost (negative reward) for a given context and action"""
    user_type = context['user_type']
    time_of_day = context['time_of_day']
    base_mu = base_mean_reward[user_type][action]
    multiplier = user_type_time_multiplier[user_type][time_of_day]
    mu = base_mu * multiplier
    return -mu  # Expected cost is negative reward

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter grid search for contextual bandit')
    parser.add_argument('--analyze', action='store_true', help='Only analyze existing results without running experiments')
    parser.add_argument('--processes', type=int, default=2, 
                        help='Number of processes to use for parallel execution (default: 2)')
    parser.add_argument('--results-file', type=str, default=None,
                        help='Specify a custom results file to analyze (only used with --analyze)')
    
    args = parser.parse_args()
    
    if args.analyze:
        if args.results_file:
            # Use specified results file for analysis
            original_results_file = RESULTS_FILE
            RESULTS_FILE = args.results_file
            print(f"Analyzing specified results file: {RESULTS_FILE}")
            analyze_results()
            # Restore original file name
            RESULTS_FILE = original_results_file
        else:
            print(f"Analyzing results from: {RESULTS_FILE}")
            analyze_results()
    else:
        # Set multiprocessing start method
        mp.set_start_method('spawn', force=True)
        print(f"Starting context-aware hyperparameter grid search with {args.processes} processes...")
        run_grid_search(args.processes)
        print("Analyzing results...")
        analyze_results()