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

st.set_page_config(layout="wide")

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

# Run simulation with optional seeding
def run_simulation(vw, num_iterations, user_types, times_of_day, actions, cost_function, do_learn=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    cost_sum = 0.
    ctr = []

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

        ctr.append(-1 * cost_sum / i)
        # Yield current iteration and CTR for real-time plotting
        yield i, ctr[-1]

def plot_ctr(num_iterations, ctr_data):
    plt.plot(range(1, num_iterations + 1), ctr_data)
    plt.xlabel('num_iterations', fontsize=14)
    plt.ylabel('ctr', fontsize=14)
    plt.show()

def plot_action_distribution_by_user_type(vw, user_types, times_of_day, actions, num_samples=1000):
    """Visualize how actions are distributed for each user type and time of day"""
    distributions = {}
    
    for user_type in user_types:
        for time in times_of_day:
            key = f"{user_type}-{time}"
            distributions[key] = {action: 0 for action in actions}
            
            # Sample actions multiple times to get distribution
            for _ in range(num_samples):
                context = {'user_type': user_type, 'time_of_day': time}
                action, _ = get_action(vw, context, actions)
                distributions[key][action] += 1 / num_samples
    
    # Create a new figure for Streamlit
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(distributions))
    width = 0.8 / len(actions)
    
    for i, action in enumerate(actions):
        values = [dist[action] for dist in distributions.values()]
        ax.bar(x + i * width, values, width, label=action)
    
    ax.set_ylabel('Selection Probability')
    ax.set_title('Action Distribution by User Type and Time')
    ax.set_xticks(x + width * len(actions) / 2)
    ax.set_xticklabels(distributions.keys(), rotation=45)
    ax.legend()
    plt.tight_layout()
    
    return fig

def plot_reward_heatmap(user_types, times_of_day, actions):
    """Visualize the expected reward structure as a heatmap, showing both base and adjusted rewards"""
    base_rewards = np.zeros((len(user_types) * len(times_of_day), len(actions)))
    adjusted_rewards = np.zeros((len(user_types) * len(times_of_day), len(actions)))
    
    for i, user_type in enumerate(user_types):
        for j, time in enumerate(times_of_day):
            for k, action in enumerate(actions):
                context = {'user_type': user_type, 'time_of_day': time}
                base_mu = base_mean_reward[user_type][action]
                multiplier = user_type_time_multiplier[user_type][time]
                base_rewards[i * len(times_of_day) + j, k] = -base_mu  # Base cost (negative reward)
                adjusted_rewards[i * len(times_of_day) + j, k] = -get_expected_cost(context, action)  # Adjusted cost
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 1]})
    
    # Plot base rewards
    im1 = ax1.imshow(base_rewards, cmap='YlOrRd')
    ax1.set_xticks(np.arange(len(actions)))
    ax1.set_yticks(np.arange(len(user_types) * len(times_of_day)))
    ax1.set_xticklabels(actions, rotation=45)
    ax1.set_yticklabels([f"{ut}-{tod}" for ut in user_types for tod in times_of_day])
    ax1.set_title('Base Reward Structure (Negative Cost)')
    plt.colorbar(im1, ax=ax1, label='Base Reward (Negative)')
    
    # Add text annotations for base rewards
    for i in range(len(user_types) * len(times_of_day)):
        for j in range(len(actions)):
            text = ax1.text(j, i, f'{base_rewards[i, j]:.1f}',
                           ha="center", va="center", color="black")
    
    # Plot adjusted rewards
    im2 = ax2.imshow(adjusted_rewards, cmap='YlOrRd')
    ax2.set_xticks(np.arange(len(actions)))
    ax2.set_yticks(np.arange(len(user_types) * len(times_of_day)))
    ax2.set_xticklabels(actions, rotation=45)
    ax2.set_yticklabels([f"{ut}-{tod}" for ut in user_types for tod in times_of_day])
    ax2.set_title('Adjusted Reward Structure (Negative Cost)')
    plt.colorbar(im2, ax=ax2, label='Adjusted Reward (Negative)')
    
    # Add text annotations for adjusted rewards
    for i in range(len(user_types) * len(times_of_day)):
        for j in range(len(actions)):
            text = ax2.text(j, i, f'{adjusted_rewards[i, j]:.1f}',
                           ha="center", va="center", color="black")
    
    plt.tight_layout()
    return fig

def main():
    st.title("Contextual Bandit Learning Visualization")
    
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
    num_iterations = st.slider("Number of Iterations", min_value=100, max_value=10000, value=10000, step=100)
    gamma = st.slider("Gamma (Exploration)", min_value=0.1, max_value=50.0, value=50.0, step=0.1)
    learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=2.0, value=1.5, step=0.01)
    initial_t = st.slider("Initial T (Learning Rate Decay)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    power_t = st.slider("Power T (Decay Exponent)", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    noise_sigma = st.slider("Reward Noise (Sigma % of Mean)", min_value=0.05, max_value=0.3, value=0.05, step=0.01)
    use_seed = st.checkbox("Set Random Seed", value=True)
    seed = st.number_input("Random Seed (if checked)", min_value=0, value=42) if use_seed else None
    
    if st.button("Start Learning"):
        fig_placeholder = st.empty()
        metrics_placeholder = st.empty()
        dist_placeholder = st.empty()  # For action distribution
        heat_placeholder = st.empty()  # For reward heatmap
        
        # Initialize Plotly figure without y-axis limit
        fig = go.Figure()
        fig.update_layout(
            xaxis_title="Iterations",
            yaxis_title="Cumulative Reward (CTR)",
            xaxis_range=[0, num_iterations],
            showlegend=True,
            template="plotly_white"
        )
        
        # Data storage
        x_data, y_data = [], []
        moving_avg = deque(maxlen=100)  # 100-iteration moving average
        ab_y_data = []  # A/B testing baseline
        bandit_total_reward = 0.0  # Track total reward for SquareCB
        ab_total_reward = 0.0      # Track total reward for A/B
        
        # Initialize VW with SquareCB, using optimized parameters for better convergence
        vw_args = f"--cb_explore_adf -q UA --quiet --squarecb --gamma {gamma} -l {learning_rate} --initial_t {initial_t} --power_t {power_t} --cb_type mtr --normalize "
        
        st.info(f"Using VW command: {vw_args}")
        vw = pyvw.Workspace(vw_args)
        
        col1, col2 = st.columns([1, 1])
        
        # Show progress bar
        progress = st.progress(0)
        
        for iteration, ctr in run_simulation(vw, num_iterations, user_types, times_of_day, actions, 
                                           lambda context, action: get_cost(context, action, noise_sigma), 
                                           do_learn=True, seed=seed):
            x_data.append(iteration)
            y_data.append(ctr)
            moving_avg.append(ctr)

            # Accumulate bandit total reward using raw cost
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
            ab_y_data.append(-ab_cost if len(ab_y_data) == 0 else ab_y_data[-1] * (iteration-1)/iteration - ab_cost/iteration)
            
            if iteration % 50 == 0:
                # Update Plotly traces
                fig.data = []
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name='SquareCB CTR', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=x_data, y=[np.mean(moving_avg)]*len(x_data), mode='lines', 
                                        name='100-Iter Moving Avg', line=dict(color='orange', dash='dash')))
                fig.add_trace(go.Scatter(x=x_data, y=ab_y_data, mode='lines', name='A/B Testing', 
                                        line=dict(color='gray', dash='dot')))
                
                fig_placeholder.plotly_chart(fig, use_container_width=True, key=f"ctr_plot_{iteration}")
                
                metrics_placeholder.markdown(f"""
                    ### Current Metrics
                    - Iteration: {iteration}
                    - Current CTR: {ctr:.3f}
                    - Moving Avg CTR: {np.mean(moving_avg):.3f}
                """)
                
                # Update progress bar
                progress.progress(iteration / num_iterations)
                
                if iteration % 1000 == 0:
                    with st.container():
                        with col1:
                            # Render Action Distribution
                            dist_fig = plot_action_distribution_by_user_type(vw, user_types, times_of_day, actions)
                            dist_placeholder.pyplot(dist_fig)
                            plt.close(dist_fig)  # Ensure figure is closed
                        
                        with col2:
                            # Render Reward Heatmap
                            heat_fig = plot_reward_heatmap(user_types, times_of_day, actions)
                            heat_placeholder.pyplot(heat_fig)
                            plt.close(heat_fig)  # Ensure figure is closed
        
        # Final update
        fig_placeholder.plotly_chart(fig, use_container_width=True, key="ctr_plot_final")
        progress.empty()  # Clear progress bar
        
        st.success("Learning completed!")
        
        # Summary Table
        st.write("### Final Metrics Summary")
        final_bandit_ctr = y_data[-1]
        final_ab_ctr = ab_y_data[-1]
        improvement = ((final_bandit_ctr - final_ab_ctr) / abs(final_ab_ctr)) * 100 if final_ab_ctr != 0 else 0
        
        summary_data = {
            "Method": ["A/B Testing", "SquareCB"],
            "Final CTR": [f"{final_ab_ctr:.3f}", f"{final_bandit_ctr:.3f}"],
            "Total Reward": [f"{ab_total_reward:.1f}", f"{bandit_total_reward:.1f}"],
            "Relative Improvement (%)": ["-", f"{improvement:.1f}"]
        }
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        # A/B comparison plot
        st.write("### Comparing with A/B Testing")
        fig_compare, ax_compare = plt.subplots(figsize=(10, 6))
        ax_compare.plot(range(1, num_iterations + 1), y_data, label='SquareCB')
        ax_compare.plot(range(1, num_iterations + 1), ab_y_data, label='A/B Testing')
        ax_compare.set_xlabel('Number of Iterations')
        ax_compare.set_ylabel('Cumulative Reward')
        ax_compare.set_title('A/B Testing vs Contextual Bandit Performance Comparison')
        ax_compare.legend()
        ax_compare.grid(True)
        st.pyplot(fig_compare)

if __name__ == "__main__":
    main()