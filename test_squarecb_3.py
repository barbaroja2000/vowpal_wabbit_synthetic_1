from vowpalwabbit import pyvw
import random
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
import streamlit as st
import time
import plotly.graph_objects as go
from collections import deque
import gc  # Import garbage collector
import os
import psutil  # For memory monitoring
import seaborn as sns

# Set visualization style to match generate_report.py
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams.update({'font.size': 12})

st.set_page_config(layout="wide")

# Initialize session state for memory management
if 'has_run_before' not in st.session_state:
    st.session_state.has_run_before = False

# Define user types, times of day, and actions
user_types = ["high_roller", "casual_player", "sports_enthusiast", "newbie"]
times_of_day = ["morning", "afternoon", "evening"]
actions = ["slots_heavy", "live_casino", "sports_betting", "mixed_games", "promotional"]

# Base mean rewards for each user-type and action combination (mirroring initial code)
base_mean_reward = {
    "high_roller": {
        "slots_heavy": 50,
        "live_casino": 200,
        "sports_betting": 10,
        "mixed_games": 30,  # Reduced from 100
        "promotional": 20
    },
    "casual_player": {
        "slots_heavy": 30,
        "live_casino": 15,
        "sports_betting": 5,
        "mixed_games": 10,  # Reduced from 20
        "promotional": 10
    },
    "sports_enthusiast": {
        "slots_heavy": 10,
        "live_casino": 20,
        "sports_betting": 100,
        "mixed_games": 20,  # Reduced from 50
        "promotional": 5
    },
    "newbie": {
        "slots_heavy": 15,
        "live_casino": 10,
        "sports_betting": 5,
        "mixed_games": 10,  # Reduced from 20
        "promotional": 30
    }
}

user_type_time_multiplier = {
    "high_roller": {"morning": 0.9, "afternoon": 1.0, "evening": 1.5},
    "casual_player": {"morning": 1.0, "afternoon": 0.8, "evening": 1.2},
    "sports_enthusiast": {"morning": 0.7, "afternoon": 1.3, "evening": 1.3},
    "newbie": {"morning": 0.8, "afternoon": 1.0, "evening": 1.4}
}

# Updated get_cost with less restrictive noise clipping and reward capping
def get_cost(context, action, noise_sigma=0.1):
    user_type = context['user_type']
    time_of_day = context['time_of_day']
    base_mu = base_mean_reward[user_type][action]
    multiplier = user_type_time_multiplier[user_type][time_of_day]
    mu = base_mu * multiplier
    sigma = noise_sigma * mu
    noise = np.random.normal(0, 1)
    noise = max(min(noise, 2.0), -2.0)  # Limit noise to ±2.0 SD (less restrictive)
    reward = max(0, min(mu + sigma * noise, mu * 1.3))  # Cap reward at 1.3x mean (less restrictive)
    cost = -reward
    return cost

# Updated get_expected_cost with user-type-specific time multipliers
def get_expected_cost(context, action):
    user_type = context['user_type']
    time_of_day = context['time_of_day']
    base_mu = base_mean_reward[user_type][action]
    multiplier = user_type_time_multiplier[user_type][time_of_day]
    mu = base_mu * multiplier
    expected_cost = -mu  # Expected cost is -mean_reward
    return expected_cost

# This function modifies (context, action, cost, probability) to VW friendly format
def to_vw_example_format(context, actions, cb_label=None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared |UserType user_type={} time_of_day={}\n".format(context["user_type"], context["time_of_day"])
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += "0:{}:{} ".format(cost, prob)
        example_string += "|Action layout={} \n".format(action)
    # Strip the last newline
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

# Display preference matrix
def get_preference_matrix(cost_fun):
    def expand_grid(data_dict):
        rows = itertools.product(*data_dict.values())
        return pd.DataFrame.from_records(rows, columns=data_dict.keys())

    df = expand_grid({'user_types': user_types, 'times_of_day': times_of_day, 'actions': actions})
    df['cost'] = df.apply(lambda r: cost_fun({'user_type': r[0], 'time_of_day': r[1]}, r[2]), axis=1)
    return df.pivot_table(index=['user_types', 'times_of_day'], 
                          columns='actions', 
                          values='cost')

# Run simulation with optional seeding and memory optimization
def run_simulation(vw, num_iterations, user_types, times_of_day, actions, cost_function, do_learn=True, seed=None, downsample_rate=1):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    cost_sum = 0.
    apy = []
    last_yield_iteration = 0

    for i in range(1, num_iterations + 1):
        user_type = choose_user(user_types)
        time_of_day = choose_time_of_day(times_of_day)
        
        context = {'user_type': user_type, 'time_of_day': time_of_day}
        action, prob = get_action(vw, context, actions)
        
        cost = cost_function(context, action)
        cost_sum += cost

        if do_learn:
            vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)), pyvw.Workspace.lContextualBandit)
            vw.learn(vw_format)
            # Free memory by deleting the parsed example
            del vw_format

        current_apy = -1 * cost_sum / i
        
        # Only store APY at downsampled rate to reduce memory usage
        if i % downsample_rate == 0 or i == num_iterations:
            apy.append(current_apy)
            # Yield current iteration and APY for real-time plotting
            yield i, current_apy
            last_yield_iteration = i
        
        # Periodically force garbage collection
        if i % 1000 == 0:
            gc.collect()

def plot_apy(num_iterations, apy_data):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_iterations + 1), apy_data, linewidth=2)
    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel('Average Player Yield (APY)', fontsize=14)
    plt.title('Learning Curve: APY vs Iterations', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_action_distribution_by_user_type(vw, user_types, times_of_day, actions, num_samples=500):
    """Visualize how actions are distributed for each user type and time of day"""
    distributions = {}
    
    for user_type in user_types:
        for time in times_of_day:
            key = f"{user_type}-{time}"
            distributions[key] = {action: 0 for action in actions}
            
            # Sample actions multiple times to get distribution (reduced from 1000 to 500)
            for _ in range(num_samples):
                context = {'user_type': user_type, 'time_of_day': time}
                action, _ = get_action(vw, context, actions)
                distributions[key][action] += 1 / num_samples
    
    # Create a new figure for Streamlit
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(distributions))
    width = 0.8 / len(actions)
    
    # Use better colors matching the report style
    colors = sns.color_palette("deep", len(actions))
    
    for i, action in enumerate(actions):
        values = [dist[action] for dist in distributions.values()]
        ax.bar(x + i * width, values, width, label=action, color=colors[i])
    
    ax.set_ylabel('Selection Probability', fontsize=14)
    ax.set_title('SquareCB vs A/B Testing Performance by Context', fontsize=16)
    ax.set_xticks(x + width * len(actions) / 2)
    ax.set_xticklabels(distributions.keys(), rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig

def plot_reward_heatmap(user_types, times_of_day, actions):
    """Visualize the expected reward structure as a heatmap, showing both base and adjusted rewards"""
    base_rewards = np.zeros((len(user_types) * len(times_of_day), len(actions)))
    adjusted_rewards = np.zeros((len(user_types) * len(times_of_day), len(actions)))
    
    contexts = []
    for i, user_type in enumerate(user_types):
        for j, time in enumerate(times_of_day):
            contexts.append(f"{user_type}-{time}")
            for k, action in enumerate(actions):
                context = {'user_type': user_type, 'time_of_day': time}
                base_mu = base_mean_reward[user_type][action]
                multiplier = user_type_time_multiplier[user_type][time]
                base_rewards[i * len(times_of_day) + j, k] = -base_mu  # Base cost (negative reward)
                adjusted_rewards[i * len(times_of_day) + j, k] = -get_expected_cost(context, action)  # Adjusted cost
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 1]})
    
    # Plot base rewards - using report style
    im1 = ax1.imshow(base_rewards, cmap='YlOrRd')
    ax1.set_xticks(np.arange(len(actions)))
    ax1.set_yticks(np.arange(len(contexts)))
    ax1.set_xticklabels(actions, rotation=45)
    ax1.set_yticklabels(contexts)
    ax1.set_title('Base Reward Structure (Average Player Yield)', fontsize=16)
    plt.colorbar(im1, ax=ax1, label='Base Reward')
    
    # Add text annotations for base rewards
    for i in range(len(contexts)):
        for j in range(len(actions)):
            text = ax1.text(j, i, f'{base_rewards[i, j]:.1f}',
                          ha="center", va="center", color="black")
    
    # Plot adjusted rewards
    im2 = ax2.imshow(adjusted_rewards, cmap='YlOrRd')
    ax2.set_xticks(np.arange(len(actions)))
    ax2.set_yticks(np.arange(len(contexts)))
    ax2.set_xticklabels(actions, rotation=45)
    ax2.set_yticklabels(contexts)
    ax2.set_title('Time-Adjusted Reward Structure (Average Player Yield)', fontsize=16)
    plt.colorbar(im2, ax=ax2, label='Adjusted Reward')
    
    # Add text annotations for adjusted rewards
    for i in range(len(contexts)):
        for j in range(len(actions)):
            text = ax2.text(j, i, f'{adjusted_rewards[i, j]:.1f}',
                          ha="center", va="center", color="black")
    
    plt.tight_layout()
    return fig

def plot_parameter_effects(gamma, learning_rate, apy):
    """Create a scatter plot showing parameter effects on APY"""
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter([learning_rate], [gamma], c=[apy], cmap='viridis', s=150, alpha=0.8)
    plt.colorbar(scatter, label='APY')
    ax.set_xlabel('Learning Rate', fontsize=14)
    ax.set_ylabel('Gamma (Exploration)', fontsize=14)
    ax.set_title('Effect of Learning Rate and Gamma on APY', fontsize=16)
    plt.tight_layout()
    return fig

def plot_apy_comparison(squarecb_apy, ab_apy, improvement):
    """Create a bar chart comparing SquareCB and A/B Testing APY"""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['SquareCB (Contextual)', 'A/B Testing (Random)']
    values = [squarecb_apy, ab_apy]
    colors = ['#3498db', '#95a5a6']  # Blue for SquareCB, Gray for A/B Testing
    
    ax.bar(labels, values, color=colors)
    ax.set_title('Average Player Yield (APY) Comparison', fontsize=16)
    ax.set_ylabel('APY (Average Reward)', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    
    # Add improvement annotation
    ax.annotate(f"{improvement:.2f}% improvement", 
               xy=(0, values[0]), 
               xytext=(0.5, max(values) * 1.1),
               arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
               fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Memory usage monitoring function
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def main():
    st.title("Contextual Bandit Learning Visualization")
    
    # Clean up previous run if exists
    if st.session_state.has_run_before:
        gc.collect()
        plt.close('all')  # Close all matplotlib figures
    
    # Add explanation in an expander
    with st.expander("ℹ️ How this simulation works", expanded=False):
        st.markdown("""
        ### About this Simulation
        
        This demo shows how a Contextual Bandit algorithm learns to personalize content layout for different types of users.
        
        #### Key Components:
        - **Users**: Four types of users (High Roller, Casual Player, Sports Enthusiast, Newbie)
        - **Context**: User type and time of day (morning/afternoon/evening) with user-specific reward adjustments
        - **Actions**: Different layout options (slots_heavy, live_casino, etc.)
        - **Rewards**: Simulated staking amounts based on user preferences, adjusted by user type and time of day, with adjustable noise
        
        #### Learning Process:
        1. For each iteration, we simulate a user visit
        2. The algorithm considers the context (user type & time)
        3. It selects a layout based on past learning
        4. We simulate user feedback based on predefined preferences with noise
        5. The algorithm updates its model based on the feedback
        
        #### Visualizations:
        - **Learning Curve**: Shows how the algorithm improves over time
        - **Action Distribution**: Shows how the algorithm learns to match layouts to users
        - **Reward Heatmap**: Displays both base and adjusted expected reward structures
        
        The algorithm is compared against random A/B testing to demonstrate its effectiveness.

        #### Tuning Parameters:
        - **Number of Iterations**: Controls simulation length; higher values allow more learning (default: 5000).
        - **Gamma (Exploration)**: Controls exploration vs. exploitation; higher values mean more exploration (range: 0.1–50.0, default: 0.1).
        - **Learning Rate**: Controls how quickly the model learns from rewards; higher values mean faster learning but more sensitivity to noise (range: 0.01–2.0, default: 1.0).
        - **Initial T (Learning Rate Decay)**: Starting learning rate for decay; higher values allow rapid initial learning (range: 0.1–5.0, default: 0.5).
        - **Power T (Decay Exponent)**: Controls decay speed; lower values slow decay (range: 0.1–1.0, default: 0.7).
        - **Reward Noise (Sigma % of Mean)**: Adds variability to rewards; lower values reduce noise (range: 0.05–0.3, default: 0.1).
        - **Set Random Seed**: Ensures consistent runs for debugging (default: 42, optional).
        """)
    
    # Add sliders and inputs for tuning with improved defaults for better convergence
    num_iterations = st.slider("Number of Iterations", min_value=100, max_value=10000, value=5000, step=100)
    gamma = st.slider("Gamma (Exploration)", min_value=0.1, max_value=50.0, value=50.0, step=0.1)
    learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=2.0, value=1.5, step=0.01)
    initial_t = st.slider("Initial T (Learning Rate Decay)", min_value=0.1, max_value=10.0, value=5.0, step=0.1)
    power_t = st.slider("Power T (Decay Exponent)", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    noise_sigma = st.slider("Reward Noise (Sigma % of Mean)", min_value=0.05, max_value=0.3, value=0.05, step=0.01)
    use_seed = st.checkbox("Set Random Seed", value=True)
    seed = st.number_input("Random Seed (if checked)", min_value=0, value=42) if use_seed else None
    
    # Add memory optimization options
    st.subheader("Memory Optimization Settings")
    downsample_rate = st.slider("Data Downsampling Rate", min_value=1, max_value=50, value=10, 
                               help="Only store 1 out of N data points to reduce memory usage. Higher values use less memory but show less detail.")
    plot_update_frequency = st.slider("Plot Update Frequency", min_value=50, max_value=1000, value=200, 
                                     help="How often to update the plots (in iterations). Higher values use less memory.")
    visualization_frequency = st.slider("Visualization Update Frequency", min_value=1000, max_value=5000, value=2000, 
                                       help="How often to update the distribution and heatmap visualizations (in iterations).")
    
    memory_info = st.empty()
    memory_info.info(f"Current memory usage: {get_memory_usage():.1f} MB")
    
    if st.button("Start Learning"):
        # Mark that we've run the simulation
        st.session_state.has_run_before = True
        
        # Create placeholders
        fig_placeholder = st.empty()
        metrics_placeholder = st.empty()
        dist_placeholder = st.empty()  # For action distribution
        heat_placeholder = st.empty()  # For reward heatmap
        
        # Initialize Plotly figure with improved styling
        fig = go.Figure()
        fig.update_layout(
            title="Learning Curve: APY Over Time",
            xaxis_title="Iterations",
            yaxis_title="Average Player Yield (APY)",
            xaxis_range=[0, num_iterations],
            showlegend=True,
            template="plotly_white",
            font=dict(size=14),
            legend=dict(font=dict(size=12))
        )
        
        # Data storage - use lists with pre-allocated size to reduce memory reallocation
        # Estimate size based on downsampling rate
        expected_points = num_iterations // downsample_rate + 1
        x_data = []
        y_data = []
        moving_avg = deque(maxlen=min(100, expected_points))  # 100-iteration moving average
        ab_y_data = []  # A/B testing baseline
        bandit_total_reward = 0.0  # Track total reward for SquareCB
        ab_total_reward = 0.0      # Track total reward for A/B
        
        # Initialize VW with SquareCB, using optimized parameters for better convergence
        vw_args = f"--cb_explore_adf -q UA --quiet --squarecb --gamma {gamma} -l {learning_rate} --initial_t {initial_t} --power_t {power_t} --cb_type mtr --normalize "
        
        st.info(f"Using VW command: {vw_args}")
        vw = pyvw.Workspace(vw_args)
        
        # Create columns for visualizations
        col1, col2 = st.columns([1, 1])
        
        # Additional placeholders for new visualizations from report
        with st.container():
            param_effects_placeholder = st.empty()
            apy_comparison_placeholder = st.empty()
        
        # Show progress bar
        progress = st.progress(0)
        
        # Track last plot update to avoid too frequent updates
        last_plot_update = 0
        last_viz_update = 0
        
        try:
            for iteration, apy in run_simulation(vw, num_iterations, user_types, times_of_day, actions, 
                                               lambda context, action: get_cost(context, action, noise_sigma), 
                                               do_learn=True, seed=seed, downsample_rate=downsample_rate):
                x_data.append(iteration)
                y_data.append(apy)
                moving_avg.append(apy)

                # Accumulate bandit total reward using raw cost - only sample periodically to save computation
                if iteration % downsample_rate == 0:
                    user_type = choose_user(user_types)
                    time_of_day = choose_time_of_day(times_of_day)
                    context = {'user_type': user_type, 'time_of_day': time_of_day}
                    action, _ = get_action(vw, context, actions)
                    cost = get_cost(context, action, noise_sigma)
                    bandit_total_reward += -cost  # Reward = -cost
                    
                    # A/B baseline
                    ab_context = {'user_type': choose_user(user_types), 'time_of_day': choose_time_of_day(times_of_day)}
                    ab_action = random.choice(actions)
                    ab_cost = get_cost(ab_context, ab_action, noise_sigma)
                    ab_total_reward += -ab_cost
                    
                    # Calculate running average for A/B testing
                    if len(ab_y_data) == 0:
                        ab_y_data.append(-ab_cost)
                    else:
                        # Use a more memory-efficient calculation that doesn't require storing all values
                        prev_avg = ab_y_data[-1]
                        new_avg = prev_avg * (len(ab_y_data)/(len(ab_y_data)+1)) - ab_cost/(len(ab_y_data)+1)
                        ab_y_data.append(new_avg)
                
                # Update plots less frequently to save memory
                if iteration - last_plot_update >= plot_update_frequency or iteration == num_iterations:
                    last_plot_update = iteration
                    
                    # Update memory usage info
                    memory_info.info(f"Current memory usage: {get_memory_usage():.1f} MB")
                    
                    # Update Plotly traces - reuse the figure object instead of creating new ones
                    fig.data = []
                    
                    # Use more efficient plotting by downsampling large datasets
                    plot_downsample = max(1, len(x_data) // 1000) if len(x_data) > 1000 else 1
                    
                    # Add SquareCB trace with improved styling
                    fig.add_trace(go.Scatter(
                        x=x_data[::plot_downsample], 
                        y=y_data[::plot_downsample], 
                        mode='lines', 
                        name='SquareCB APY', 
                        line=dict(color='#3498db', width=2)
                    ))
                    
                    # Calculate moving average more efficiently
                    avg_value = sum(moving_avg) / len(moving_avg) if moving_avg else 0
                    
                    fig.add_trace(go.Scatter(
                        x=[x_data[0], x_data[-1]], 
                        y=[avg_value, avg_value], 
                        mode='lines', 
                        name='Moving Avg', 
                        line=dict(color='orange', dash='dash', width=2)
                    ))
                    
                    # Plot A/B testing data with same downsampling
                    if ab_y_data:
                        ab_plot_data = ab_y_data[::plot_downsample]
                        fig.add_trace(go.Scatter(
                            x=x_data[:len(ab_plot_data)][::plot_downsample], 
                            y=ab_plot_data, 
                            mode='lines', 
                            name='A/B Testing', 
                            line=dict(color='#95a5a6', dash='dot', width=2)
                        ))
                    
                    fig_placeholder.plotly_chart(fig, use_container_width=True, key=f"learning_curve_{iteration}")
                    
                    metrics_placeholder.markdown(f"""
                        ### Current Metrics
                        - Iteration: {iteration}
                        - Current APY: {apy:.3f}
                        - Moving Avg APY: {avg_value:.3f}
                        - Memory Usage: {get_memory_usage():.1f} MB
                    """)
                    
                    # Update progress bar
                    progress.progress(iteration / num_iterations)
                
                # Update visualizations less frequently
                if iteration - last_viz_update >= visualization_frequency or iteration == num_iterations:
                    last_viz_update = iteration
                    
                    # First row of visualizations (standard from original code)
                    with st.container():
                        with col1:
                            # Render Action Distribution with fewer samples
                            dist_fig = plot_action_distribution_by_user_type(vw, user_types, times_of_day, actions, num_samples=500)
                            dist_placeholder.pyplot(dist_fig)
                            plt.close(dist_fig)  # Ensure figure is closed
                        
                        with col2:
                            # Render Reward Heatmap
                            heat_fig = plot_reward_heatmap(user_types, times_of_day, actions)
                            heat_placeholder.pyplot(heat_fig)
                            plt.close(heat_fig)  # Ensure figure is closed
                    
                    # Second row of visualizations (new from generate_report.py)
                    with st.container():
                        col3, col4 = st.columns([1, 1])
                        
                        with col3:
                            # Parameter effects visualization (simplified version for real-time)
                            param_fig = plot_parameter_effects(gamma, learning_rate, apy)
                            param_effects_placeholder.pyplot(param_fig)
                            plt.close(param_fig)
                        
                        with col4:
                            if ab_y_data:
                                # APY comparison visualization
                                improvement = ((y_data[-1] - ab_y_data[-1]) / abs(ab_y_data[-1])) * 100 if ab_y_data[-1] != 0 else 0
                                apy_comp_fig = plot_apy_comparison(y_data[-1], ab_y_data[-1], improvement)
                                apy_comparison_placeholder.pyplot(apy_comp_fig)
                                plt.close(apy_comp_fig)
                    
                    # Force garbage collection after visualization updates
                    gc.collect()
            
            # Final update
            fig_placeholder.plotly_chart(fig, use_container_width=True, key="final_learning_curve")
            progress.empty()  # Clear progress bar
            
            st.success("Learning completed!")
            
            # Summary Table
            st.write("### Final Metrics Summary")
            final_bandit_apy = y_data[-1]
            final_ab_apy = ab_y_data[-1] if ab_y_data else 0
            improvement = ((final_bandit_apy - final_ab_apy) / abs(final_ab_apy)) * 100 if final_ab_apy != 0 else 0
            
            summary_data = {
                "Method": ["A/B Testing", "SquareCB"],
                "Final APY": [f"{final_ab_apy:.3f}", f"{final_bandit_apy:.3f}"],
                "Total Reward": [f"{ab_total_reward:.1f}", f"{bandit_total_reward:.1f}"],
                "Relative Improvement (%)": ["-", f"{improvement:.1f}"]
            }
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
            
            # Context-specific performance analysis
            st.write("### Context-Specific Performance")
            
            # Collect context-specific data
            context_performance = {}
            for user_type in user_types:
                for time in times_of_day:
                    context = {'user_type': user_type, 'time_of_day': time}
                    context_key = f"{user_type}-{time}"
                    
                    # Find optimal action
                    best_action = None
                    best_reward = float('-inf')
                    for action in actions:
                        reward = -get_expected_cost(context, action)
                        if reward > best_reward:
                            best_reward = reward
                            best_action = action
                    
                    # Sample actual action selection
                    action_counts = {a: 0 for a in actions}
                    total_reward = 0
                    num_samples = 200  # Use fewer samples to save memory
                    
                    for _ in range(num_samples):
                        chosen_action, _ = get_action(vw, context, actions)
                        action_counts[chosen_action] += 1
                        reward = -get_cost(context, chosen_action, noise_sigma)
                        total_reward += reward
                    
                    avg_reward = total_reward / num_samples
                    accuracy = action_counts[best_action] / num_samples if best_action else 0
                    
                    context_performance[context_key] = {
                        'optimal_action': best_action,
                        'optimal_reward': best_reward,
                        'avg_reward': avg_reward,
                        'accuracy': accuracy,
                        'regret': best_reward - avg_reward
                    }
            
            # Context performance table
            headers = ["Context", "Optimal Action", "Avg Reward", "Optimal Reward", "Accuracy", "Regret"]
            context_data = []
            
            for context, metrics in sorted(context_performance.items()):
                context_data.append([
                    context,
                    metrics['optimal_action'],
                    f"{metrics['avg_reward']:.2f}",
                    f"{metrics['optimal_reward']:.2f}",
                    f"{metrics['accuracy']*100:.1f}%",
                    f"{metrics['regret']:.2f}"
                ])
            
            # Calculate coverage metrics
            context_coverage = sum(1 for m in context_performance.values() if m['accuracy'] > 0.5) / len(context_performance)
            worst_context = max(context_performance.items(), key=lambda x: x[1]['regret'])[0]
            time_sensitivity = max(context_performance.values(), key=lambda x: x['accuracy'])['accuracy'] - min(context_performance.values(), key=lambda x: x['accuracy'])['accuracy']
            
            # Display context metrics
            col5, col6, col7 = st.columns(3)
            col5.metric("Context Coverage", f"{context_coverage*100:.1f}%")
            col6.metric("Time Sensitivity", f"{time_sensitivity:.3f}")
            col7.metric("Worst Context", worst_context)
            
            # Display context data table
            st.write("#### Context Performance Details")
            st.dataframe(pd.DataFrame(context_data, columns=headers))
            
            # A/B comparison plot - use downsampled data for efficiency
            st.write("### SquareCB vs A/B Testing Comparison")
            plot_downsample = max(1, len(x_data) // 500) if len(x_data) > 500 else 1
            
            fig_compare, ax_compare = plt.subplots(figsize=(12, 6))
            ax_compare.plot(x_data[::plot_downsample], y_data[::plot_downsample], label='SquareCB', linewidth=2, color='#3498db')
            
            if ab_y_data:
                ax_compare.plot(x_data[:len(ab_y_data)][::plot_downsample], ab_y_data[::plot_downsample], label='A/B Testing', linewidth=2, color='#95a5a6', linestyle='--')
            
            ax_compare.set_xlabel('Number of Iterations', fontsize=14)
            ax_compare.set_ylabel('Average Player Yield (APY)', fontsize=14)
            ax_compare.set_title('SquareCB vs A/B Testing Performance Comparison', fontsize=16)
            ax_compare.legend(fontsize=12)
            ax_compare.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_compare)
            plt.close(fig_compare)  # Ensure figure is closed
            
            # Show final parameter effects and APY comparison
            col8, col9 = st.columns([1, 1])
            
            with col8:
                final_param_fig = plot_parameter_effects(gamma, learning_rate, final_bandit_apy)
                st.pyplot(final_param_fig)
                plt.close(final_param_fig)
                
            with col9:
                final_apy_fig = plot_apy_comparison(final_bandit_apy, final_ab_apy, improvement)
                st.pyplot(final_apy_fig)
                plt.close(final_apy_fig)
            
        finally:
            # Clean up resources
            if 'vw' in locals():
                vw.finish()
                del vw
            
            # Force garbage collection
            gc.collect()
            
            # Report final memory usage
            memory_info.info(f"Final memory usage: {get_memory_usage():.1f} MB")

if __name__ == "__main__":
    main()