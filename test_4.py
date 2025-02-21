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
import numpy as np

# VW tries to minimize loss/cost, therefore we will pass cost as -reward
USER_LIKED_LAYOUT = -1.0
USER_DISLIKED_LAYOUT = 0.0

def get_cost(context,action):
    if context['user_type'] == "Sports":
        if context['time_of_day'] == "morning" and action == 'in_play':
            return USER_LIKED_LAYOUT
        elif context['time_of_day'] == "afternoon" and action == 'pre_match':
            return USER_LIKED_LAYOUT
        else:
            return USER_DISLIKED_LAYOUT
    elif context['user_type'] == "Mixed":
        if context['time_of_day'] == "morning" and action == 'bingo':
            return USER_LIKED_LAYOUT
        elif context['time_of_day'] == "afternoon" and action == 'pre_match':
            return USER_LIKED_LAYOUT
        else:
            return USER_DISLIKED_LAYOUT
    elif context['user_type'] == "Casino":
        if context['time_of_day'] == "morning" and action == 'bingo':
            return USER_LIKED_LAYOUT
        elif context['time_of_day'] == "afternoon" and action == 'slots':
            return USER_LIKED_LAYOUT
        else:
            return USER_DISLIKED_LAYOUT
        
# This function modifies (context, action, cost, probability) to VW friendly format
def to_vw_example_format(context, actions, cb_label = None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared |UserType user_type={} time_of_day={}\n".format(context["user_type"], context["time_of_day"])
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += "0:{}:{} ".format(cost, prob)
        example_string += "|Action layout={} \n".format(action)
    #Strip the last newline
    return example_string[:-1]


context = {"user_type":"Casino","time_of_day":"morning"}
actions = ["bingo", "slots", "mixed", "live_casino", "pre_match"]

print(to_vw_example_format(context,actions))

def sample_custom_pmf(pmf):
    total = sum(pmf)
    scale = 1/total
    pmf = [x * scale for x in pmf]
    draw = random.random()
    sum_prob = 0.0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if(sum_prob > draw):
            return index, prob
        
def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context,actions)
    pmf = vw.predict(vw_text_example)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob

user_types = ['Mixed', 'Casino', 'Sports']
times_of_day = ['morning', 'afternoon']
actions = ["bingo", "mixed", "slots", "sports", "in_play", "live_casino", "virtuals", "pre_match"]

def choose_user(users):
    return random.choice(users)

def choose_time_of_day(times_of_day):
    return random.choice(times_of_day)

# display preference matrix
def get_preference_matrix(cost_fun):
    def expand_grid(data_dict):
        rows = itertools.product(*data_dict.values())
        return pd.DataFrame.from_records(rows, columns=data_dict.keys())

    df = expand_grid({'user_types':user_types, 'times_of_day': times_of_day, 'actions': actions})
    df['cost'] = df.apply(lambda r: cost_fun({'user_type': r[0], 'time_of_day': r[1]}, r[2]), axis=1)

    return df.pivot_table(index=['user_types', 'times_of_day'], 
            columns='actions', 
            values='cost')

get_preference_matrix(get_cost)


def run_simulation(vw, num_iterations, user_types, times_of_day, actions, cost_function, do_learn = True):
    cost_sum = 0.
    ctr = []

    for i in range(1, num_iterations+1):
        user_type = choose_user(user_types)
        time_of_day = choose_time_of_day(times_of_day)
        
        context = {'user_type': user_type, 'time_of_day': time_of_day}
        action, prob = get_action(vw, context, actions)
        
        cost = cost_function(context, action)
        cost_sum += cost

        if do_learn:
            vw_format = vw.parse(to_vw_example_format(context, actions, (action, cost, prob)),pyvw.Workspace.lContextualBandit)
            vw.learn(vw_format)

        ctr.append(-1*cost_sum/i)
        # Yield current iteration and CTR for real-time plotting
        yield i, ctr[-1]

def plot_ctr(num_iterations, ctr_data):
    plt.plot(range(1, num_iterations+1), ctr_data)
    plt.xlabel('num_iterations', fontsize=14)
    plt.ylabel('ctr', fontsize=14)
    plt.ylim([0,1])
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
                distributions[key][action] += 1/num_samples
    
    # Create a new figure for Streamlit
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(distributions))
    width = 0.8 / len(actions)
    
    for i, action in enumerate(actions):
        values = [dist[action] for dist in distributions.values()]
        ax.bar(x + i*width, values, width, label=action)
    
    ax.set_ylabel('Selection Probability')
    ax.set_title('Action Distribution by User Type and Time')
    ax.set_xticks(x + width * len(actions)/2)
    ax.set_xticklabels(distributions.keys(), rotation=45)
    ax.legend()
    plt.tight_layout()
    
    return fig

def plot_reward_heatmap(user_types, times_of_day, actions):
    """Visualize the reward structure as a heatmap"""
    rewards = np.zeros((len(user_types) * len(times_of_day), len(actions)))
    
    for i, user_type in enumerate(user_types):
        for j, time in enumerate(times_of_day):
            for k, action in enumerate(actions):
                context = {'user_type': user_type, 'time_of_day': time}
                rewards[i*len(times_of_day) + j, k] = -get_cost(context, action)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rewards, cmap='YlOrRd')
    
    # Label axes
    ax.set_xticks(np.arange(len(actions)))
    ax.set_yticks(np.arange(len(user_types) * len(times_of_day)))
    ax.set_xticklabels(actions, rotation=45)
    ax.set_yticklabels([f"{ut}-{tod}" for ut in user_types for tod in times_of_day])
    
    # Add colorbar
    plt.colorbar(im)
    ax.set_title('Reward Structure Heatmap')
    
    # Add text annotations
    for i in range(len(user_types) * len(times_of_day)):
        for j in range(len(actions)):
            text = ax.text(j, i, f'{rewards[i, j]:.1f}',
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
        - **Users**: Three types of users (Sports, Casino, Mixed)
        - **Context**: User type and time of day (morning/afternoon)
        - **Actions**: Different layout options (bingo, slots, sports, etc.)
        - **Rewards**: The algorithm receives feedback (-1 for liked, 0 for disliked)
        
        #### Learning Process:
        1. For each iteration, we simulate a user visit
        2. The algorithm considers the context (user type & time)
        3. It selects a layout based on past learning
        4. We simulate user feedback based on predefined preferences
        5. The algorithm updates its model based on the feedback
        
        #### Visualizations:
        - **Learning Curve**: Shows how the algorithm improves over time
        - **Action Distribution**: Shows how the algorithm learns to match layouts to users
        - **Reward Heatmap**: Displays the underlying reward structure
        
        The algorithm is compared against random A/B testing to demonstrate its effectiveness.
        """)
    
    num_iterations = st.slider("Number of Iterations", min_value=100, max_value=3000, value=1000, step=100)
    
    if st.button("Start Learning"):
        fig_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Initialize Plotly figure
        fig = go.Figure()
        fig.update_layout(
            xaxis_title="Iterations",
            yaxis_title="Cumulative Reward (CTR)",
            yaxis_range=[0, 1],
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
        
        # Initialize VW with SquareCB
        vw = pyvw.Workspace("--cb_explore_adf -q UA --quiet --squarecb")
        
        col1, col2 = st.columns(2)
        
        for iteration, ctr in run_simulation(vw, num_iterations, user_types, times_of_day, actions, get_cost):
            x_data.append(iteration)
            y_data.append(ctr)
            moving_avg.append(ctr)

            # Changed: Correctly accumulate bandit_total_reward using raw cost
            user_type = choose_user(user_types)
            time_of_day = choose_time_of_day(times_of_day)
            context = {'user_type': user_type, 'time_of_day': time_of_day}
            action, _ = get_action(vw, context, actions)
            cost = get_cost(context, action)
            bandit_total_reward += -cost  # Reward = -costt_total_reward += -ctr * iteration if iteration == 1 else -ctr * iteration + y_data[-2] * (iteration - 1)
            
            # A/B baseline
            ab_context = {'user_type': choose_user(user_types), 'time_of_day': choose_time_of_day(times_of_day)}
            ab_action = random.choice(actions)
            ab_cost = get_cost(ab_context, ab_action)
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
                
                if iteration % 1000 == 0:
                    with col1:
                        st.write("Action Distribution")
                        dist_fig = plot_action_distribution_by_user_type(vw, user_types, times_of_day, actions)
                        st.pyplot(dist_fig)
                        plt.close(dist_fig)
                    with col2:
                        st.write("Reward Heatmap")
                        heat_fig = plot_reward_heatmap(user_types, times_of_day, actions)
                        st.pyplot(heat_fig)
                        plt.close(heat_fig)
        
        # Final update
        fig_placeholder.plotly_chart(fig, use_container_width=True, key="ctr_plot_final")
        
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
        
        # Optional: Keep the A/B comparison plot if desired
        st.write("### Comparing with A/B Testing")
        fig_compare, ax_compare = plt.subplots(figsize=(10, 6))
        ax_compare.plot(range(1, num_iterations+1), y_data, label='SquareCB')
        ax_compare.plot(range(1, num_iterations+1), ab_y_data, label='A/B Testing')
        ax_compare.set_xlabel('Number of Iterations')
        ax_compare.set_ylabel('Cumulative Reward')
        ax_compare.set_title('A/B Testing vs Contextual Bandit Performance Comparison')
        ax_compare.legend()
        ax_compare.grid(True)
        st.pyplot(fig_compare)


def run_ab_test_simulation(num_iterations, user_types, times_of_day, actions, cost_function):
    """Simulate A/B testing by randomly selecting actions with equal probability"""
    cost_sum = 0.
    ctr = []
    
    for i in range(1, num_iterations+1):
        # 1. Choose a random user
        user_type = choose_user(user_types)
        # 2. Choose time of day
        time_of_day = choose_time_of_day(times_of_day)
        
        # 3. In A/B testing, we randomly select an action with equal probability
        action = random.choice(actions)
        
        # 4. Get cost/reward for the selected action
        context = {'user_type': user_type, 'time_of_day': time_of_day}
        cost = cost_function(context, action)
        cost_sum += cost
        
        # Track cumulative reward (negative cost)
        ctr.append(-1*cost_sum/i)
    
    return ctr

if __name__ == "__main__":
    main()