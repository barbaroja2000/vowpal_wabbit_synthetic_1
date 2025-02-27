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
    'gamma': [5.0, 15.0, 30.0],          # Reduced from 4 to 3 values
    'learning_rate': [0.1, 0.5, 1.0],    # Removed 1.5 as it's likely too high
    'initial_t': [1.0, 3.0],             # Reduced to 2 key values
    'power_t': [0.4, 0.6],               # Focused on middle range values
    'noise_sigma': [0.05, 0.15]          # Removed middle value for efficiency
}

# Output file
RESULTS_FILE = 'hyperparameter_search_results.csv'
CHECKPOINT_FILE = 'hyperparameter_search_checkpoint.json'
MAX_ITERATIONS = 6000
EVAL_INTERVAL = 200  # Evaluate performance every N iterations

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
def calculate_time_sensitivity(vw, user_types, times_of_day, actions, num_samples=100):
    """
    Measure how differently the model behaves across time periods for the same user_type.
    Higher score means greater time sensitivity.
    """
    time_sensitivity_score = 0
    
    for user_type in user_types:
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
            
    # Normalize by number of comparisons
    return time_sensitivity_score / (len(user_types) * len(list(itertools.combinations(times_of_day, 2))))

def run_experiment(params, seed=42):
    """Run a single experiment with given hyperparameters"""
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
    
    # Track metrics
    cost_sum = 0.0
    cost_history = []
    
    # Metrics storage for efficiency
    metrics = {
        'ctr': [],
        'iteration': []
    }
    
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
        
        # Record metrics at evaluation intervals
        if i % EVAL_INTERVAL == 0 or i == MAX_ITERATIONS:
            ctr = -1 * cost_sum / i  # CTR is negative of average cost
            metrics['ctr'].append(ctr)
            metrics['iteration'].append(i)
    
    # Calculate time sensitivity at end of experiment
    time_sensitivity = calculate_time_sensitivity(vw, user_types, times_of_day, actions)
    
    # Clean up
    vw.finish()
    del vw
    gc.collect()  # Force garbage collection for memory efficiency
    
    return {
        'params': params,
        'final_ctr': metrics['ctr'][-1],
        'time_sensitivity': time_sensitivity,
        'ctr_history': metrics['ctr'],
        'iterations': metrics['iteration']
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

def run_grid_search():
    """Run hyperparameter grid search with checkpointing"""
    # Generate all parameter combinations
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Total parameter combinations: {len(param_combinations)}")
    
    # Get already completed runs
    completed_runs = get_completed_runs()
    print(f"Found {len(completed_runs)} already completed runs")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    start_idx = checkpoint['current_idx']
    
    # Create or open results file
    if not os.path.exists(RESULTS_FILE):
        # Create empty dataframe with columns
        df = pd.DataFrame(columns=[
            'gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma',
            'final_ctr', 'time_sensitivity', 'timestamp'
        ])
        df.to_csv(RESULTS_FILE, index=False)
    
    # Loop through all combinations
    for idx, values in enumerate(param_combinations[start_idx:], start=start_idx):
        params = {key: values[i] for i, key in enumerate(param_keys)}
        
        # Check if this combination has already been run
        param_tuple = tuple(values)
        if param_tuple in completed_runs:
            print(f"Skipping combination {idx+1}/{len(param_combinations)} - already completed")
            continue
        
        print(f"\nRunning combination {idx+1}/{len(param_combinations)}:")
        print(params)
        
        # Time the experiment
        start_time = time.time()
        
        # Run experiment
        result = run_experiment(params)
        
        # Measure elapsed time
        elapsed = time.time() - start_time
        
        # Extract and save results
        row = {
            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'initial_t': params['initial_t'],
            'power_t': params['power_t'],
            'noise_sigma': params['noise_sigma'],
            'final_ctr': result['final_ctr'],
            'time_sensitivity': result['time_sensitivity'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Append to CSV
        df = pd.DataFrame([row])
        df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
        
        # Update checkpoint
        save_checkpoint(idx + 1, param_combinations)
        
        print(f"Completed in {elapsed:.2f} seconds")
        print(f"Final CTR: {result['final_ctr']:.3f}, Time Sensitivity: {result['time_sensitivity']:.3f}")
    
    # Remove checkpoint file when done
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    print("\nGrid search completed!")

def analyze_results():
    """Analyze results and find best hyperparameter combinations"""
    if not os.path.exists(RESULTS_FILE):
        print("No results file found.")
        return
    
    df = pd.read_csv(RESULTS_FILE)
    
    print(f"Analyzed {len(df)} experiment runs")
    
    # Top 10 by CTR
    print("\nTop 10 parameter combinations by CTR:")
    top_ctr = df.sort_values('final_ctr', ascending=False).head(10)
    print(top_ctr[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 'final_ctr']])
    
    # Top 10 by time sensitivity
    print("\nTop 10 parameter combinations by time sensitivity:")
    top_time = df.sort_values('time_sensitivity', ascending=False).head(10)
    print(top_time[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 'time_sensitivity']])
    
    # Calculate correlation between parameters and metrics
    print("\nCorrelation between parameters and metrics:")
    correlation = df[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 
                     'final_ctr', 'time_sensitivity']].corr()
    print(correlation[['final_ctr', 'time_sensitivity']])
    
    # Create a composite score (CTR + time sensitivity)
    # Normalize both scores first
    df['norm_ctr'] = (df['final_ctr'] - df['final_ctr'].min()) / (df['final_ctr'].max() - df['final_ctr'].min())
    df['norm_time'] = (df['time_sensitivity'] - df['time_sensitivity'].min()) / (df['time_sensitivity'].max() - df['time_sensitivity'].min())
    df['composite_score'] = 0.7 * df['norm_ctr'] + 0.3 * df['norm_time']  # Weight CTR higher
    
    print("\nTop 10 parameter combinations by composite score (70% CTR, 30% time sensitivity):")
    top_composite = df.sort_values('composite_score', ascending=False).head(10)
    print(top_composite[['gamma', 'learning_rate', 'initial_t', 'power_t', 'noise_sigma', 
                         'final_ctr', 'time_sensitivity', 'composite_score']])
    
    # Visualize parameter effects
    # Example: learning rate vs gamma colored by CTR
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['learning_rate'], df['gamma'], 
                         c=df['final_ctr'], cmap='viridis', 
                         s=50, alpha=0.7)
    plt.colorbar(scatter, label='CTR')
    plt.xlabel('Learning Rate')
    plt.ylabel('Gamma (Exploration)')
    plt.title('Effect of Learning Rate and Gamma on CTR')
    plt.savefig('param_effects_ctr.png')
    
    # Gamma vs power_t colored by time sensitivity
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['gamma'], df['power_t'], 
                         c=df['time_sensitivity'], cmap='plasma', 
                         s=50, alpha=0.7)
    plt.colorbar(scatter, label='Time Sensitivity')
    plt.xlabel('Gamma (Exploration)')
    plt.ylabel('Power T (Decay Exponent)')
    plt.title('Effect of Gamma and Power T on Time Sensitivity')
    plt.savefig('param_effects_time.png')
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter grid search for contextual bandit')
    parser.add_argument('--analyze', action='store_true', help='Only analyze existing results without running experiments')
    
    args = parser.parse_args()
    
    if args.analyze:
        print("Analyzing existing results...")
        analyze_results()
    else:
        print("Starting hyperparameter grid search...")
        run_grid_search()
        print("Analyzing results...")
        analyze_results()