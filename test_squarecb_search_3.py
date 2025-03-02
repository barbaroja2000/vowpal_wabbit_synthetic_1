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
import traceback
import argparse

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
    'gamma': [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],          # Expanded to include higher values up to 50.0
    'learning_rate': [ 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],      # Added higher values up to 2.0
    'initial_t': [0.5, 1.0, 3.0, 5.0, 8.0],          # Expanded range from 0.5 to 8.0
    'power_t': [0.1, 0.3, 0.5, 0.7, 0.9],            # Expanded to include 0.3 and more values
    'noise_sigma': [0.05]      # More granular search with additional values
}

# Output file
RESULTS_FILE = 'context_aware_hyperparameter_search_results.csv'
RESULTS_FILE_PHASE2 = 'context_aware_hyperparameter_search_results_phase2.csv'
CHECKPOINT_FILE = 'context_aware_hyperparameter_search_checkpoint.json'
MAX_ITERATIONS = 10000  # Reduced for faster processing with parallelization
EVAL_INTERVAL = 250  # Increased interval for better performance
# Number of repetitions for phase 2 statistical validation
NUM_REPETITIONS = 5  # Number of times to repeat promising experiments
PHASE2_TOP_PERCENT = 0.15  # Top 15% of parameter combinations to validate

# Add a new constant for the context performance results file
CONTEXT_PERFORMANCE_FILE = 'context_performance_best_config.csv'

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

def run_experiment(params, seed=None, run_id=0):
    """Run a single experiment with given hyperparameters with memory optimization
    
    Args:
        params: Dictionary of hyperparameters
        seed: Random seed (if None, a unique seed will be generated)
        run_id: ID to distinguish between multiple runs of the same parameter set (default 0)
    """
    # Set a unique seed if none provided
    if seed is None:
        # Make seed unique for each run_id
        base_seed = int(time.time() * 1000) % 100000 + os.getpid()
        seed = base_seed + run_id * 1000
    
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
    
    # Track metrics for both contextual bandit and A/B testing
    cb_cost_sum = 0.0
    ab_cost_sum = 0.0
    last_cb_ctr = 0.0
    last_ab_ctr = 0.0
    
    # Run iterations
    for i in range(1, MAX_ITERATIONS + 1):
        # Sample context - use same context for both algorithms for fair comparison
        user_type = choose_user(user_types)
        time_of_day = choose_time_of_day(times_of_day)
        context = {'user_type': user_type, 'time_of_day': time_of_day}
        
        # ===== Contextual Bandit Algorithm =====
        # Get action from contextual bandit
        action, prob = get_action(vw, context, actions)
        
        # Get cost and update
        cost = get_cost(context, action, noise_sigma)
        cb_cost_sum += cost
        
        # Learn
        vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)), pyvw.Workspace.lContextualBandit)
        vw.learn(vw_format)
        
        # ===== A/B Testing Baseline =====
        # Random action selection (A/B testing)
        ab_action = random.choice(actions)
        ab_cost = get_cost(context, ab_action, noise_sigma)
        ab_cost_sum += ab_cost
        
        # Calculate final CTRs
        if i == MAX_ITERATIONS:
            last_cb_ctr = -1 * cb_cost_sum / i
            last_ab_ctr = -1 * ab_cost_sum / i
    
    # Calculate time sensitivity at end of experiment
    time_sensitivity = calculate_time_sensitivity(vw, user_types, times_of_day, actions)
    
    # Calculate context-specific performance
    context_performance = calculate_context_specific_performance(vw, user_types, times_of_day, actions)
    
    # Calculate A/B testing context-specific performance for comparison
    ab_context_performance = calculate_ab_context_performance(user_types, times_of_day, actions, noise_sigma)
    
    # Clean up
    vw.finish()
    del vw
    gc.collect()  # Force garbage collection
    
    # Return minimal result structure to save memory, now including A/B testing results
    return {
        'final_ctr': last_cb_ctr,
        'ab_final_ctr': last_ab_ctr,
        'improvement_over_ab': ((last_cb_ctr - last_ab_ctr) / abs(last_ab_ctr) * 100) if last_ab_ctr != 0 else 0,
        'time_sensitivity': time_sensitivity,
        'context_coverage': context_performance['context_coverage'],
        'average_regret': context_performance['average_regret'],
        'max_regret': context_performance['max_regret'],
        'worst_context': context_performance['worst_context'][0],  # Just store the key to save memory
        'context_performance_details': context_performance['context_performance'],  # Store full details
        'ab_context_performance': ab_context_performance,  # Store A/B testing context performance
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
            'ab_final_ctr': result['ab_final_ctr'],
            'improvement_over_ab': result['improvement_over_ab'],
            'time_sensitivity': result['time_sensitivity'],
            'context_coverage': result['context_coverage'],
            'average_regret': result['average_regret'],
            'max_regret': result['max_regret'],
            'worst_context': result['worst_context'],
            'context_performance_details': result['context_performance_details'],
            'ab_context_performance': result['ab_context_performance'],
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

def process_param_combination_phase2(param_tuple, param_keys, completed_runs, result_lock):
    """Process a single parameter combination with multiple runs for phase 2 statistical validation"""
    try:
        # Convert tuple to dict of parameters
        params = {key: param_tuple[i] for i, key in enumerate(param_keys)}
        
        # Skip if already done
        phase2_key = param_tuple + ('phase2',)  # Add marker to distinguish from phase 1
        if phase2_key in completed_runs:
            return None
        
        # Print status with minimal output to avoid buffer issues
        print(f"Phase 2: Running multiple repetitions for gamma={params['gamma']}, lr={params['learning_rate']}")
        
        # Time the experiment
        start_time = time.time()
        
        # Run multiple experiments with the same parameters
        result = run_multiple_experiments(params, num_repetitions=NUM_REPETITIONS)
        
        # Measure elapsed time
        elapsed = time.time() - start_time
        
        # Extract only the necessary results to save memory
        row = {
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'initial_t': params['initial_t'],
            'power_t': params['power_t'],
            'noise_sigma': params['noise_sigma'],
            'final_ctr_mean': result['final_ctr_mean'],
            'final_ctr_std': result['final_ctr_std'],
            'final_ctr_cv': result['final_ctr_cv'],
            'ab_final_ctr_mean': result['ab_final_ctr_mean'],
            'ab_final_ctr_std': result['ab_final_ctr_std'],
            'improvement_over_ab_mean': result['improvement_over_ab_mean'],
            'improvement_over_ab_std': result['improvement_over_ab_std'],
            'improvement_over_ab_cv': result['improvement_over_ab_cv'],
            'time_sensitivity_mean': result['time_sensitivity_mean'],
            'time_sensitivity_std': result['time_sensitivity_std'],
            'context_coverage_mean': result['context_coverage_mean'],
            'context_coverage_std': result['context_coverage_std'],
            'average_regret_mean': result['average_regret_mean'],
            'average_regret_std': result['average_regret_std'],
            'max_regret_mean': result['max_regret_mean'],
            'max_regret_std': result['max_regret_std'],
            'num_repetitions': result['num_repetitions'],
            'timestamp': result['timestamp']
        }
        
        # Write to CSV with lock to prevent race conditions
        with result_lock:
            with open(RESULTS_FILE_PHASE2, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)
        
        print(f"Phase 2 completed: gamma={params['gamma']}, lr={params['learning_rate']} ({NUM_REPETITIONS} runs) in {elapsed:.2f}s")
        
        # Force cleanup
        del result
        gc.collect()
        
        return row
    
    except Exception as e:
        print(f"Error processing parameters in phase 2: {param_tuple}")
        print(f"Exception: {str(e)}")
        traceback.print_exc()
        return None

def run_grid_search(num_processes=None, run_phase2=True):
    """Run hyperparameter grid search with multiprocessing and memory optimization
    
    Args:
        num_processes: Number of processes to use in parallel
        run_phase2: Whether to automatically run phase 2 after phase 1 is completed
    """
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
        'final_ctr', 'ab_final_ctr', 'improvement_over_ab', 'time_sensitivity', 'context_coverage', 
        'average_regret', 'max_regret', 'worst_context', 'context_performance_details', 'ab_context_performance',
        'timestamp'
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

    if run_phase2:
        print("Starting phase 2 grid search...")
        run_phase2_grid_search(num_processes)

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
    has_ab_comparison = 'ab_final_ctr' in df.columns and 'improvement_over_ab' in df.columns
    
    if not has_context_metrics:
        print("\nWARNING: Results file does not contain context-specific metrics.")
        print("Only showing basic CTR and time sensitivity metrics.")
    
    # Top 10 by CTR
    print("\nTop 10 parameter combinations by CTR:")
    top_ctr = df.sort_values('final_ctr', ascending=False).head(10)
    if has_ab_comparison:
        print(top_ctr[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 'final_ctr', 'ab_final_ctr', 'improvement_over_ab']])
    else:
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
    
    # A/B Testing Comparison if available
    if has_ab_comparison:
        print("\nTop 10 parameter combinations by improvement over A/B testing:")
        top_improvement = df.sort_values('improvement_over_ab', ascending=False).head(10)
        print(top_improvement[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 'final_ctr', 'ab_final_ctr', 'improvement_over_ab']])
    
    # Calculate correlation between parameters and metrics
    print("\nCorrelation between parameters and metrics:")
    correlation_cols = ['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 
                         'final_ctr', 'time_sensitivity']
    if has_context_metrics:
        correlation_cols.extend(['context_coverage', 'average_regret', 'max_regret'])
    if has_ab_comparison:
        correlation_cols.extend(['ab_final_ctr', 'improvement_over_ab'])
    
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
        
        # Add A/B comparison component if available
        if has_ab_comparison:
            df['norm_ab_improvement'] = (df['improvement_over_ab'] - df['improvement_over_ab'].min()) / (df['improvement_over_ab'].max() - df['improvement_over_ab'].min())
            
            # Enhanced composite score with A/B comparison
            df['context_aware_score'] = (
                0.25 * df['norm_ctr'] + 
                0.1 * df['norm_time'] + 
                0.15 * df['norm_coverage'] + 
                0.2 * df['norm_avg_regret'] + 
                0.1 * df['norm_max_regret'] +
                0.2 * df['norm_ab_improvement']  # 20% weight on A/B improvement
            )
        else:
            # Original context-aware score
            df['context_aware_score'] = (
                0.3 * df['norm_ctr'] + 
                0.1 * df['norm_time'] + 
                0.2 * df['norm_coverage'] + 
                0.25 * df['norm_avg_regret'] + 
                0.15 * df['norm_max_regret']
            )
        
        print("\nTop 10 parameter combinations by context-aware composite score:")
        if has_ab_comparison:
            print("(25% CTR, 10% time sensitivity, 15% context coverage, 30% regret metrics, 20% A/B improvement)")
            top_context_aware = df.sort_values('context_aware_score', ascending=False).head(10)
            print(top_context_aware[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 
                                 'final_ctr', 'ab_final_ctr', 'improvement_over_ab', 'context_coverage', 'average_regret', 'context_aware_score']])
        else:
            print("(30% CTR, 10% time sensitivity, 20% context coverage, 40% regret metrics)")
            top_context_aware = df.sort_values('context_aware_score', ascending=False).head(10)
            print(top_context_aware[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 
                                 'final_ctr', 'time_sensitivity', 'context_coverage', 'average_regret', 'context_aware_score']])
    
    # Original composite score for comparison
    print("\nTop 10 parameter combinations by original composite score (70% CTR, 30% time sensitivity):")
    top_original = df.sort_values('original_composite_score', ascending=False).head(10)
    print(top_original[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 
                      'final_ctr', 'time_sensitivity', 'original_composite_score']])
    
    # ==========================================
    # Detailed Report on Best Configuration
    # ==========================================
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE REPORT FOR OPTIMAL SOLUTION")
    print("="*80)
    
    # Get best configuration based on context-aware score or CTR if not available
    if 'context_aware_score' in df.columns:
        best_config = df.loc[df['context_aware_score'].idxmax()]
        metric_name = 'context-aware score'
    else:
        best_config = df.loc[df['final_ctr'].idxmax()]
        metric_name = 'CTR'
    
    print(f"\nBest configuration (by {metric_name}):")
    print(f"  Gamma: {best_config['gamma']:.2f}")
    print(f"  Learning Rate: {best_config['learning_rate']:.2f}")
    print(f"  Initial T: {best_config['initial_t']:.2f}")
    print(f"  Power T: {best_config['power_t']:.2f}")
    print(f"  Noise Sigma: {best_config['noise_sigma']:.2f}")
    
    print("\nOverall Performance Metrics:")
    print(f"  Contextual Bandit Final CTR: {best_config['final_ctr']:.4f}")
    
    if has_ab_comparison:
        print(f"  A/B Testing Final CTR: {best_config['ab_final_ctr']:.4f}")
        improvement = best_config['improvement_over_ab']
        print(f"  Improvement over A/B: {improvement:.2f}%")
    
    if has_context_metrics:
        print(f"  Time Sensitivity: {best_config['time_sensitivity']:.4f}")
        print(f"  Context Coverage: {best_config['context_coverage']:.4f} ({best_config['context_coverage']*100:.1f}%)")
        print(f"  Average Regret: {best_config['average_regret']:.4f}")
        print(f"  Max Regret: {best_config['max_regret']:.4f}")
        print(f"  Worst Context: {best_config['worst_context']}")
    
    # Create A/B Comparison Bar Chart
    if has_ab_comparison:
        plt.figure(figsize=(10, 6))
        labels = ['Contextual Bandit', 'A/B Testing']
        values = [best_config['final_ctr'], best_config['ab_final_ctr']]
        colors = ['blue', 'gray']
        
        plt.bar(labels, values, color=colors)
        plt.title('CTR Comparison: Contextual Bandit vs A/B Testing')
        plt.ylabel('Cumulative Reward (CTR)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text labels
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        # Add improvement annotation
        plt.annotate(f"{improvement:.2f}% improvement", 
                   xy=(0, values[0]), 
                   xytext=(0.5, max(values) * 1.1),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        plt.savefig('cb_vs_ab_comparison.png')
        print("\nSaved CTR comparison chart to 'cb_vs_ab_comparison.png'")
    
    # Context-specific performance if available
    if 'context_performance_details' in df.columns and has_context_metrics:
        try:
            # Parse the context performance details from the selected row
            # Use json.loads instead of eval for safer parsing
            # Also handle the case where the data might be already a dictionary
            if isinstance(best_config['context_performance_details'], dict):
                context_perf = best_config['context_performance_details']
            else:
                try:
                    import json
                    context_perf = json.loads(best_config['context_performance_details'].replace("'", '"'))
                except json.JSONDecodeError as json_err:
                    # Fall back to eval with proper error handling
                    try:
                        context_perf = eval(best_config['context_performance_details'])
                    except Exception as e:
                        print(f"Could not parse context performance data using eval: {e}")
                        context_perf = {}
            
            if context_perf:
                print("\nContext-Specific Performance:")
                print("-" * 80)
                print(f"{'Context':<20} {'Optimal Action':<15} {'Avg Reward':<12} {'Optimal Reward':<15} {'Accuracy %':<12} {'Regret':<10}")
                print("-" * 80)
                
                total_accuracy = 0
                num_contexts = len(context_perf)
                
                for context, metrics in sorted(context_perf.items()):
                    # Safely split context using underscore delimiter
                    parts = context.split('_')
                    user_type = parts[0]
                    time = '_'.join(parts[1:]) if len(parts) > 1 else "unknown"
                    
                    # Safely extract metrics values
                    try:
                        optimal_action = metrics.get('optimal_action', 'unknown')
                        avg_reward = metrics.get('avg_reward', 0.0)
                        optimal_reward = metrics.get('optimal_reward', 0.0)
                        accuracy = metrics.get('accuracy', 0.0)
                        regret = metrics.get('regret', 0.0)
                        
                        accuracy_pct = accuracy * 100
                        total_accuracy += accuracy
                        
                        print(f"{context:<20} {optimal_action:<15} {avg_reward:.2f}{'':<8} {optimal_reward:.2f}{'':<10} {accuracy_pct:.1f}%{'':<6} {regret:.2f}")
                    except Exception as e:
                        print(f"Error processing metrics for {context}: {e}")
                
                print("-" * 80)
                print(f"Average Accuracy: {(total_accuracy/num_contexts)*100:.2f}%" if num_contexts else "Average Accuracy: 0.0%")
            else:
                print("\nNo context-specific performance data available.")
                
            # If we have A/B comparison, add that to the context report
            if 'ab_context_performance' in df.columns and has_ab_comparison:
                try:
                    # Parse the A/B performance details using the same safe approach
                    if isinstance(best_config['ab_context_performance'], dict):
                        ab_perf = best_config['ab_context_performance']
                    else:
                        try:
                            import json
                            ab_perf = json.loads(best_config['ab_context_performance'].replace("'", '"'))
                        except json.JSONDecodeError:
                            # Fall back to eval with proper error handling
                            try:
                                ab_perf = eval(best_config['ab_context_performance'])
                            except Exception as e:
                                print(f"Could not parse A/B context performance data using eval: {e}")
                                ab_perf = {'ab_context_performance': {}}
                    
                    # Make sure ab_context_performance exists in the dict
                    if ab_perf and 'ab_context_performance' in ab_perf:
                        print("\nA/B Testing vs SquareCB Performance by Context:")
                        print("-" * 80)
                        print(f"{'Context':<20} {'CB Reward':<12} {'A/B Reward':<12} {'Improvement %':<15}")
                        print("-" * 80)
                        
                        cb_rewards = []
                        ab_rewards = []
                        
                        for context in sorted(context_perf.keys()):
                            if context in ab_perf['ab_context_performance']:
                                try:
                                    # Safely access values
                                    cb_reward = context_perf[context].get('avg_reward', 0.0)
                                    # Get A/B cost (might be stored differently)
                                    ab_metrics = ab_perf['ab_context_performance'][context]
                                    if isinstance(ab_metrics, dict):
                                        ab_cost = ab_metrics.get('ab_cost', 0.0)
                                    else:
                                        ab_cost = 0.0
                                    
                                    ab_reward = -ab_cost  # Reward is negative cost
                                    improvement = ((cb_reward - ab_reward) / abs(ab_reward)) * 100 if ab_reward != 0 else 0
                                    
                                    cb_rewards.append(cb_reward)
                                    ab_rewards.append(ab_reward)
                                    
                                    print(f"{context:<20} {cb_reward:.2f}{'':<8} {ab_reward:.2f}{'':<8} {improvement:.2f}%")
                                except Exception as e:
                                    print(f"Error processing A/B comparison for {context}: {e}")
                        
                        if cb_rewards and ab_rewards:
                            print("-" * 80)
                            avg_cb = sum(cb_rewards) / len(cb_rewards) 
                            avg_ab = sum(ab_rewards) / len(ab_rewards)
                            avg_imp = ((avg_cb - avg_ab) / abs(avg_ab)) * 100 if avg_ab != 0 else 0
                            print(f"Average:           {avg_cb:.2f}{'':<8} {avg_ab:.2f}{'':<8} {avg_imp:.2f}%")
                            
                            # Create contextual performance visualization
                            try:
                                plt.figure(figsize=(12, 8))
                                contexts = sorted(context_perf.keys())
                                contexts_for_chart = [ctx for ctx in contexts if ctx in ab_perf['ab_context_performance']]
                                
                                if contexts_for_chart:
                                    x = np.arange(len(contexts_for_chart))
                                    width = 0.35
                                    
                                    cb_vals = [context_perf[ctx].get('avg_reward', 0.0) for ctx in contexts_for_chart]
                                    ab_vals = [-ab_perf['ab_context_performance'][ctx].get('ab_cost', 0.0) for ctx in contexts_for_chart]
                                    
                                    fig, ax = plt.subplots(figsize=(14, 8))
                                    rects1 = ax.bar(x - width/2, cb_vals, width, label='SquareCB')
                                    rects2 = ax.bar(x + width/2, ab_vals, width, label='A/B Testing')
                                    
                                    ax.set_ylabel('Average Reward')
                                    ax.set_title('SquareCB vs A/B Testing Performance by Context')
                                    ax.set_xticks(x)
                                    ax.set_xticklabels(contexts_for_chart, rotation=45, ha='right')
                                    ax.legend()
                                    
                                    # Add improvement labels
                                    for i, (cb, ab) in enumerate(zip(cb_vals, ab_vals)):
                                        imp = ((cb - ab) / abs(ab)) * 100 if ab != 0 else 0
                                        ax.annotate(f"{imp:.1f}%", 
                                                  xy=(i, max(cb, ab) + 1),
                                                  ha='center', va='bottom',
                                                  color='green' if imp > 0 else 'red',
                                                  weight='bold')
                                    
                                    fig.tight_layout()
                                    plt.savefig('context_performance_comparison.png')
                                    print("\nSaved context performance comparison chart to 'context_performance_comparison.png'")
                                    plt.close()
                            except Exception as e:
                                print(f"\nCould not create performance comparison visualization: {str(e)}")
                        else:
                            print("\nNo A/B testing context performance data available.")
                except Exception as e:
                    print(f"Could not process A/B context performance data: {e}")
                    traceback.print_exc()
        except (ValueError, SyntaxError) as e:
            print(f"Could not parse context performance data: {e}")
    
    # ==========================================
    # Visualizations
    # ==========================================
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

def calculate_ab_context_performance(user_types, times_of_day, actions, noise_sigma, num_samples=30):
    """
    Calculate A/B testing context-specific performance for comparison
    Uses random action selection to simulate A/B testing for each context
    
    Returns:
    - Per-context performance metrics
    - Average regret across contexts
    - Worst-performing context
    - Context coverage score (% of contexts with good performance)
    """
    # Store results for each context
    ab_context_performance = {}
    ab_context_regret = {}
    
    for user_type in user_types:
        for time in times_of_day:
            context = {'user_type': user_type, 'time_of_day': time}
            context_key = f"{user_type}_{time}"
            
            # Find optimal action for this context (lowest expected cost)
            optimal_action = min(actions, key=lambda a: get_expected_cost(context, a))
            optimal_reward = -get_expected_cost(context, optimal_action)
            
            # Sample multiple A/B tests to get average performance
            total_ab_reward = 0
            optimal_action_hits = 0
            avg_cost = 0
            
            for _ in range(num_samples):
                # Random action selection (A/B testing)
                ab_action = random.choice(actions)
                
                # Count how many times A/B testing randomly selects optimal action
                if ab_action == optimal_action:
                    optimal_action_hits += 1
                
                # Get reward for this action
                ab_cost = get_cost(context, ab_action, noise_sigma)
                total_ab_reward += -ab_cost  # Reward is negative cost
                avg_cost += ab_cost
            
            # Calculate average metrics
            avg_ab_reward = total_ab_reward / num_samples
            avg_cost = avg_cost / num_samples
            ab_accuracy = optimal_action_hits / num_samples
            ab_regret = optimal_reward - avg_ab_reward
            
            # Store results
            ab_context_performance[context_key] = {
                'ab_action': 'random',  # A/B testing uses random selection
                'ab_cost': avg_cost,
                'ab_reward': avg_ab_reward,
                'optimal_action': optimal_action,
                'optimal_reward': optimal_reward,
                'ab_accuracy': ab_accuracy,
                'ab_regret': ab_regret
            }
            ab_context_regret[context_key] = ab_regret
    
    # Calculate aggregate metrics
    ab_average_regret = sum(ab_context_regret.values()) / len(ab_context_regret)
    ab_worst_context_key, ab_worst_regret = max(ab_context_regret.items(), key=lambda x: x[1])
    
    # Context coverage: % of contexts with regret better than 1.5x average
    ab_context_coverage = sum(1 for r in ab_context_regret.values() if r < ab_average_regret * 1.5) / len(ab_context_regret)
    
    return {
        'ab_context_performance': ab_context_performance,
        'ab_average_regret': ab_average_regret,
        'ab_worst_context': (ab_worst_context_key, ab_worst_regret),
        'ab_context_coverage': ab_context_coverage,
        'ab_max_regret': ab_worst_regret
    }

def run_multiple_experiments(params, num_repetitions=NUM_REPETITIONS):
    """
    Run multiple repetitions of an experiment with the same parameters to assess variability
    
    Args:
        params: Dictionary of hyperparameters
        num_repetitions: Number of times to repeat the experiment
    
    Returns:
        Dictionary with aggregated results including means and standard deviations
    """
    # Store results from each run
    results = []
    
    # Run experiments with different seeds
    for run_id in range(num_repetitions):
        # Run a single experiment
        result = run_experiment(params, run_id=run_id)
        results.append(result)
        
        # Force garbage collection between runs
        gc.collect()
    
    # Aggregate metrics across runs
    aggregated = {
        # Basic parameters (same for all runs)
        'gamma': params['gamma'],
        'learning_rate': params['learning_rate'],
        'initial_t': params['initial_t'],
        'power_t': params['power_t'],
        'noise_sigma': params['noise_sigma'],
        
        # List of all individual run results
        'individual_runs': results,
        
        # Calculate means for key metrics
        'final_ctr_mean': np.mean([r['final_ctr'] for r in results]),
        'final_ctr_std': np.std([r['final_ctr'] for r in results]),
        'ab_final_ctr_mean': np.mean([r['ab_final_ctr'] for r in results]),
        'ab_final_ctr_std': np.std([r['ab_final_ctr'] for r in results]),
        'improvement_over_ab_mean': np.mean([r['improvement_over_ab'] for r in results]),
        'improvement_over_ab_std': np.std([r['improvement_over_ab'] for r in results]),
        'time_sensitivity_mean': np.mean([r['time_sensitivity'] for r in results]),
        'time_sensitivity_std': np.std([r['time_sensitivity'] for r in results]),
        'context_coverage_mean': np.mean([r['context_coverage'] for r in results]),
        'context_coverage_std': np.std([r['context_coverage'] for r in results]),
        'average_regret_mean': np.mean([r['average_regret'] for r in results]),
        'average_regret_std': np.std([r['average_regret'] for r in results]),
        'max_regret_mean': np.mean([r['max_regret'] for r in results]),
        'max_regret_std': np.std([r['max_regret'] for r in results]),
        
        # Keep the details from the first run for simplicity
        'context_performance_details': results[0]['context_performance_details'],
        'ab_context_performance': results[0]['ab_context_performance'],
        
        # Number of repetitions
        'num_repetitions': num_repetitions,
        
        # Coefficient of variation (CV) for key metrics to measure stability
        # CV = std / mean (lower is more stable)
        'final_ctr_cv': np.std([r['final_ctr'] for r in results]) / 
                         max(abs(np.mean([r['final_ctr'] for r in results])), 1e-10),
        'improvement_over_ab_cv': np.std([r['improvement_over_ab'] for r in results]) / 
                                  max(abs(np.mean([r['improvement_over_ab'] for r in results])), 1e-10),
        
        # Timestamp
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return aggregated

def get_completed_phase2_runs():
    """Get a set of parameter combinations that have already been run in phase 2"""
    if not os.path.exists(RESULTS_FILE_PHASE2):
        return set()
    
    try:
        df = pd.read_csv(RESULTS_FILE_PHASE2)
        # Convert parameters to tuple for hashable comparison
        completed = set()
        for _, row in df.iterrows():
            param_tuple = (row['gamma'], row['learning_rate'], row['initial_t'], 
                          row['power_t'], row['noise_sigma'], 'phase2')  # Add phase2 marker
            completed.add(param_tuple)
        return completed
    except:
        return set()

def run_phase2_grid_search(num_processes=None):
    """Run phase 2 of the grid search with multiple repetitions for top parameter combinations"""
    # First, check if phase 1 has been completed
    if not os.path.exists(RESULTS_FILE):
        print("Error: Phase 1 results file not found. Please run phase 1 first.")
        return
    
    try:
        # Load phase 1 results
        df = pd.read_csv(RESULTS_FILE)
        
        # Check if we have enough data
        if len(df) == 0:
            print("Error: Phase 1 results file is empty. Please run phase 1 first.")
            return
        
        print(f"Loaded {len(df)} parameter combinations from phase 1")
        
        # Determine which metric to use for selecting top combinations
        if 'context_aware_score' in df.columns:
            sort_metric = 'context_aware_score'
            ascending = False  # Higher is better
        else:
            # Fall back to CTR if context_aware_score is not available
            sort_metric = 'final_ctr'
            ascending = False  # Higher is better
        
        print(f"Selecting top {PHASE2_TOP_PERCENT*100:.0f}% of parameter combinations based on {sort_metric}")
        
        # Sort and select top combinations
        df_sorted = df.sort_values(sort_metric, ascending=ascending)
        num_to_select = max(1, int(len(df) * PHASE2_TOP_PERCENT))
        top_combinations = df_sorted.head(num_to_select)
        
        print(f"Selected {len(top_combinations)} top parameter combinations for phase 2")
        
        # Convert top combinations to parameter tuples
        param_keys = ['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma']
        param_combinations = []
        
        for _, row in top_combinations.iterrows():
            param_tuple = tuple(row[key] for key in param_keys)
            param_combinations.append(param_tuple)
        
        # Get already completed phase 2 runs
        completed_phase2_runs = get_completed_phase2_runs()
        print(f"Found {len(completed_phase2_runs)} already completed phase 2 runs")
        
        # Filter out already completed phase 2 runs 
        param_combinations_to_run = []
        for pc in param_combinations:
            phase2_key = pc + ('phase2',)
            if phase2_key not in completed_phase2_runs:
                param_combinations_to_run.append(pc)
        
        print(f"Phase 2 combinations remaining to run: {len(param_combinations_to_run)}")
        
        # Define phase 2 result columns
        phase2_columns = [
            'gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma',
            'final_ctr_mean', 'final_ctr_std', 'final_ctr_cv',
            'ab_final_ctr_mean', 'ab_final_ctr_std',
            'improvement_over_ab_mean', 'improvement_over_ab_std', 'improvement_over_ab_cv',
            'time_sensitivity_mean', 'time_sensitivity_std',
            'context_coverage_mean', 'context_coverage_std',
            'average_regret_mean', 'average_regret_std',
            'max_regret_mean', 'max_regret_std',
            'num_repetitions', 'timestamp'
        ]
        
        # Create phase 2 results file if it doesn't exist
        if not os.path.exists(RESULTS_FILE_PHASE2):
            # Create empty dataframe with columns
            df_phase2 = pd.DataFrame(columns=phase2_columns)
            df_phase2.to_csv(RESULTS_FILE_PHASE2, index=False)
        
        # If no combinations to run, we're done
        if not param_combinations_to_run:
            print("All phase 2 parameter combinations already completed.")
            return
        
        # Determine the number of processes to use (default to 3)
        if num_processes is None:
            num_processes = min(3, len(param_combinations_to_run))
        print(f"Running phase 2 with {num_processes} parallel processes")
        
        # Create a progress tracking mechanism
        progress_lock = threading.Lock()
        progress_count = mp.Value('i', 0)
        
        def track_progress(result):
            if result is not None:  # Skip None results (already completed runs)
                with progress_lock:
                    progress_count.value += 1
                    percent = progress_count.value / len(param_combinations_to_run) * 100
                    print(f"\rPhase 2 Progress: {progress_count.value}/{len(param_combinations_to_run)} ({percent:.1f}%)", 
                          end="", flush=True)
        
        # Process in smaller batches to manage memory better
        BATCH_SIZE = num_processes * 2  # Process 2 batches per process at a time
        
        # Use a Manager object to create a shareable lock
        with mp.Manager() as manager:
            result_lock = manager.Lock()
            
            # Process combinations in batches to control memory usage
            for batch_start in range(0, len(param_combinations_to_run), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(param_combinations_to_run))
                batch = param_combinations_to_run[batch_start:batch_end]
                
                print(f"\nProcessing phase 2 batch {batch_start//BATCH_SIZE + 1}/{(len(param_combinations_to_run)+BATCH_SIZE-1)//BATCH_SIZE}")
                
                # Create a partial function with fixed parameters
                process_func = partial(process_param_combination_phase2, 
                                    param_keys=param_keys, 
                                    completed_runs=completed_phase2_runs,
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
        
        print("\nPhase 2 grid search completed!")
        
    except Exception as e:
        print(f"Error in phase 2 grid search: {str(e)}")
        traceback.print_exc()

def analyze_phase2_results():
    """Analyze statistical results from phase 2 (multiple runs for top parameter combinations)"""
    if not os.path.exists(RESULTS_FILE_PHASE2):
        print("No phase 2 results file found.")
        return
    
    try:
        df = pd.read_csv(RESULTS_FILE_PHASE2)
        
        if len(df) == 0:
            print("Phase 2 results file is empty.")
            return
            
        print(f"Analyzed {len(df)} parameter combinations with multiple runs")
        print(f"Each combination was run {df['num_repetitions'].iloc[0]} times")
        
        # Display key statistics for all parameter combinations
        print("\nPhase 2 Parameter Combinations (ordered by mean CTR):")
        print("="*100)
        print(f"{'Gamma':<8} {'LR':<8} {'Init T':<8} {'Power T':<8} {'CTR Mean':<10} {'CTR Std':<10} {'CTR CV %':<10} {'Improv %':<10} {'Improv Std':<10}")
        print("-"*100)
        
        # Sort by mean CTR descending
        sorted_df = df.sort_values('final_ctr_mean', ascending=False)
        
        for _, row in sorted_df.iterrows():
            gamma = row['gamma']
            lr = row['learning_rate']
            init_t = row['initial_t']
            power_t = row['power_t']
            ctr_mean = row['final_ctr_mean']
            ctr_std = row['final_ctr_std']
            ctr_cv = row['final_ctr_cv'] * 100  # Convert to percentage
            
            # Handle case where improvement columns might be missing
            if 'improvement_over_ab_mean' in row:
                improv_mean = row['improvement_over_ab_mean']
                improv_std = row['improvement_over_ab_std']
            else:
                improv_mean = np.nan
                improv_std = np.nan
                
            print(f"{gamma:<8.2f} {lr:<8.2f} {init_t:<8.2f} {power_t:<8.2f} {ctr_mean:<10.4f} {ctr_std:<10.4f} {ctr_cv:<10.2f} {improv_mean:<10.2f} {improv_std:<10.4f}")
        
        print("\n")
        
        # Check if we have context-specific metrics
        context_cols = ['context_coverage_mean', 'average_regret_mean']
        has_context = all(col in df.columns for col in context_cols)
        
        if has_context:
            # Print results for context metrics
            print("\nPhase 2 Parameter Combinations (ordered by context coverage):")
            print("="*100)
            print(f"{'Gamma':<8} {'LR':<8} {'Init T':<8} {'Power T':<8} {'Ctx Cov %':<10} {'Ctx Cov Std':<12} {'Avg Regret':<10} {'Regret Std':<10}")
            print("-"*100)
            
            # Sort by context coverage descending
            sorted_df = df.sort_values('context_coverage_mean', ascending=False)
            
            for _, row in sorted_df.iterrows():
                gamma = row['gamma']
                lr = row['learning_rate']
                init_t = row['initial_t']
                power_t = row['power_t']
                ctx_cov = row['context_coverage_mean'] * 100  # Convert to percentage
                ctx_cov_std = row['context_coverage_std'] * 100  # Convert to percentage
                avg_regret = row['average_regret_mean']
                avg_regret_std = row['average_regret_std']
                
                print(f"{gamma:<8.2f} {lr:<8.2f} {init_t:<8.2f} {power_t:<8.2f} {ctx_cov:<10.2f} {ctx_cov_std:<12.2f} {avg_regret:<10.4f} {avg_regret_std:<10.4f}")
        
        # Create some visualizations for the variability
        plt.figure(figsize=(10, 6))
        
        # Sort by mean CTR for the chart
        chart_df = df.sort_values('final_ctr_mean', ascending=False).head(10)
        
        # Create parameter labels for x-axis
        param_labels = [f"γ={row['gamma']:.1f},η={row['learning_rate']:.1f}" for _, row in chart_df.iterrows()]
        
        # Plot mean CTR with error bars
        plt.errorbar(range(len(chart_df)), chart_df['final_ctr_mean'], 
                    yerr=chart_df['final_ctr_std'], fmt='o', capsize=5,
                    color='blue', label='CTR with Std Dev')
        
        # Add stability score (lower CV is better)
        stability_scores = 1 - chart_df['final_ctr_cv']  # Invert CV so higher is better
        stability_normalized = (stability_scores - min(stability_scores)) / (max(stability_scores) - min(stability_scores) + 1e-10)
        
        # Add second axis for stability score
        ax2 = plt.gca().twinx()
        ax2.plot(range(len(chart_df)), stability_normalized, 'r-', label='Stability Score')
        ax2.set_ylabel('Stability Score (higher is better)', color='red')
        
        # Set x-ticks
        plt.xticks(range(len(chart_df)), param_labels, rotation=45, ha='right')
        
        plt.title('Top 10 Parameter Combinations - Mean CTR with Std Dev')
        plt.xlabel('Parameter Combination')
        plt.ylabel('CTR')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        # Add legend for both axes
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.savefig('phase2_ctr_variability.png')
        plt.close()
        
        print("\nSaved variability analysis chart to 'phase2_ctr_variability.png'")
        
        # Find most stable configurations (lowest CV)
        print("\nMost Stable Parameter Combinations (lowest coefficient of variation):")
        stable_df = df.sort_values('final_ctr_cv', ascending=True).head(5)
        print(stable_df[['gamma', 'learning_rate', 'initial_t', 'power_t',
                         'final_ctr_mean', 'final_ctr_std', 'final_ctr_cv']])
        
        # Create a composite score considering both performance and stability
        df['stability_score'] = 1 - df['final_ctr_cv']  # Invert CV so higher is better
        
        # Normalize metrics for fair comparison
        df['norm_ctr'] = (df['final_ctr_mean'] - df['final_ctr_mean'].min()) / (df['final_ctr_mean'].max() - df['final_ctr_mean'].min())
        df['norm_stability'] = (df['stability_score'] - df['stability_score'].min()) / (df['stability_score'].max() - df['stability_score'].min())
        
        # Create a composite score (70% performance, 30% stability)
        df['robust_score'] = 0.7 * df['norm_ctr'] + 0.3 * df['norm_stability']
        
        print("\nBest Parameter Combinations (considering both performance and stability):")
        robust_df = df.sort_values('robust_score', ascending=False).head(5)
        print(robust_df[['gamma', 'learning_rate', 'initial_t', 'power_t',
                         'final_ctr_mean', 'final_ctr_std', 'final_ctr_cv', 'robust_score']])
        
        # Identify the overall best configuration
        best_config = df.loc[df['robust_score'].idxmax()]
        
        print("\n" + "="*80)
        print("BEST ROBUST CONFIGURATION FROM PHASE 2")
        print("="*80)
        print(f"Gamma: {best_config['gamma']:.2f}")
        print(f"Learning Rate: {best_config['learning_rate']:.2f}")
        print(f"Initial T: {best_config['initial_t']:.2f}")
        print(f"Power T: {best_config['power_t']:.2f}")
        print(f"Mean CTR: {best_config['final_ctr_mean']:.4f} ± {best_config['final_ctr_std']:.4f}")
        print(f"Coefficient of Variation: {best_config['final_ctr_cv']*100:.2f}%")
        
        if 'improvement_over_ab_mean' in best_config:
            print(f"Mean Improvement over A/B: {best_config['improvement_over_ab_mean']:.2f}% ± {best_config['improvement_over_ab_std']:.2f}%")
        
        if has_context:
            print(f"Mean Context Coverage: {best_config['context_coverage_mean']*100:.2f}% ± {best_config['context_coverage_std']*100:.2f}%")
            print(f"Mean Average Regret: {best_config['average_regret_mean']:.4f} ± {best_config['average_regret_std']:.4f}")
        
        return df
    
    except Exception as e:
        print(f"Error analyzing phase 2 results: {str(e)}")
        traceback.print_exc()
        return None

def run_best_config_with_context_details(num_repetitions=10):
    """
    Run experiments with the best hyperparameter configuration and save detailed
    context-specific performance data with A/B testing comparison.
    
    Args:
        num_repetitions: Number of repetitions to run for statistical significance
    """
    print(f"Running {num_repetitions} repetitions with best hyperparameter configuration...")
    
    # Best configuration from phase 2 results
    best_params = {
        'gamma': 50.0,
        'learning_rate': 2.0,
        'initial_t': 5.0,
        'power_t': 0.1,
        'noise_sigma': 0.05
    }
    
    # Run multiple experiments with this configuration
    results = run_multiple_experiments(best_params, num_repetitions)
    
    # Extract context performance details
    context_perf = results['context_performance_details']
    ab_perf_data = results['ab_context_performance']
    ab_perf = ab_perf_data['ab_context_performance']
    
    # Prepare CSV data for context-specific performance
    context_data = []
    contexts = sorted(context_perf.keys())
    
    for context in contexts:
        if context in ab_perf:
            # Extract metrics
            cb_metrics = context_perf[context]
            ab_metrics = ab_perf[context]
            
            # Get key metrics
            cb_reward = cb_metrics.get('avg_reward', 0.0)
            optimal_action = cb_metrics.get('optimal_action', 'unknown')
            optimal_reward = cb_metrics.get('optimal_reward', 0.0)
            accuracy = cb_metrics.get('accuracy', 0.0) * 100  # Convert to percentage
            regret = cb_metrics.get('regret', 0.0)
            
            # A/B testing metrics
            ab_reward = -ab_metrics.get('ab_cost', 0.0)  # Reward is negative cost
            ab_accuracy = ab_metrics.get('ab_accuracy', 0.0) * 100  # Convert to percentage
            ab_regret = ab_metrics.get('ab_regret', 0.0)
            
            # Calculate improvement over A/B testing
            reward_improvement = ((cb_reward - ab_reward) / abs(ab_reward)) * 100 if ab_reward != 0 else 0
            accuracy_improvement = accuracy - ab_accuracy
            regret_improvement = ((ab_regret - regret) / abs(ab_regret)) * 100 if ab_regret != 0 else 0
            
            # Split context into user_type and time_of_day
            parts = context.split('_')
            user_type = parts[0]
            time_of_day = '_'.join(parts[1:]) if len(parts) > 1 else "unknown"
            
            # Build row
            row = {
                'context': context,
                'user_type': user_type,
                'time_of_day': time_of_day,
                'optimal_action': optimal_action,
                'cb_reward': cb_reward,
                'ab_reward': ab_reward,
                'reward_improvement_pct': reward_improvement,
                'cb_accuracy': accuracy,
                'ab_accuracy': ab_accuracy,
                'accuracy_improvement': accuracy_improvement,
                'cb_regret': regret,
                'ab_regret': ab_regret,
                'regret_improvement_pct': regret_improvement,
                'optimal_reward': optimal_reward
            }
            context_data.append(row)
    
    # Save to CSV
    with open(CONTEXT_PERFORMANCE_FILE, 'w', newline='') as f:
        if context_data:
            writer = csv.DictWriter(f, fieldnames=context_data[0].keys())
            writer.writeheader()
            writer.writerows(context_data)
            print(f"Saved context performance data to {CONTEXT_PERFORMANCE_FILE}")
        else:
            print("No context performance data available to save")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter search for contextual bandits')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing results')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes to use')
    parser.add_argument('--best-config', action='store_true', help='Run best configuration with detailed context analysis')
    parser.add_argument('--phase2', action='store_true', help='Run phase 2 grid search')
    args = parser.parse_args()
    
    if args.analyze:
        analyze_phase2_results()
    elif args.best_config:
        run_best_config_with_context_details()
    elif args.phase2:
        run_phase2_grid_search(args.processes)
    else:
        # Run phase 1 by default
        run_grid_search(args.processes, run_phase2=False)  # Don't automatically run phase 2