from vowpalwabbit import pyvw
import random
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import numpy as np
import streamlit as st
import time

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













# Create Streamlit app
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
    
    # Add slider for num_iterations
    num_iterations = st.slider("Number of Iterations", min_value=100, max_value=20000, value=5000, step=100)
    
    # Add start button
    if st.button("Start Learning"):
        # Initialize plot
        fig_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [])
        ax.set_xlim(0, num_iterations)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cumulative Reward')
        ax.grid(True)
        
        # Initialize VW
        vw = pyvw.Workspace("--cb_explore_adf -q UA --quiet --epsilon 0.2")
        
        # Run simulation with real-time updates
        x_data, y_data = [], []
        
        # Create two columns for the visualizations
        col1, col2 = st.columns(2)
        
        for iteration, ctr in run_simulation(vw, num_iterations, user_types, times_of_day, actions, get_cost):
            x_data.append(iteration)
            y_data.append(ctr)
            
            if iteration % 50 == 0:  # Update plot every 50 iterations for performance
                line.set_data(x_data, y_data)
                fig_placeholder.pyplot(fig)
                
                # Update metrics
                metrics_placeholder.markdown(f"""
                    ### Current Metrics
                    - Iteration: {iteration}
                    - Current CTR: {ctr:.3f}
                """)
                
                # Show distribution and heatmap every 1000 iterations
                if iteration % 1000 == 0:
                    with col1:
                        st.write("Action Distribution")
                        dist_fig = plot_action_distribution_by_user_type(vw, user_types, times_of_day, actions)
                        st.pyplot(dist_fig)
                    
                    with col2:
                        st.write("Reward Heatmap")
                        heat_fig = plot_reward_heatmap(user_types, times_of_day, actions)
                        st.pyplot(heat_fig)
        
        # Final update
        line.set_data(x_data, y_data)
        fig_placeholder.pyplot(fig)
        
        st.success("Learning completed!")
        
        # Compare with A/B testing
        st.write("### Comparing with A/B Testing")
        ab_ctr = run_ab_test_simulation(num_iterations, user_types, times_of_day, actions, get_cost)
        
        fig_compare, ax_compare = plt.subplots(figsize=(10, 6))
        ax_compare.plot(range(1, num_iterations+1), y_data, label='Contextual Bandit')
        ax_compare.plot(range(1, num_iterations+1), ab_ctr, label='A/B Testing')
        ax_compare.set_xlabel('Number of Iterations')
        ax_compare.set_ylabel('Cumulative Reward')
        ax_compare.set_title('A/B Testing vs Contextual Bandit Performance Comparison')
        ax_compare.legend()
        ax_compare.grid(True)
        
        st.pyplot(fig_compare)
        
        improvement = ((y_data[-1] - ab_ctr[-1]) / abs(ab_ctr[-1]) * 100)
        st.markdown(f"""
            ### Final Results
            - A/B Testing final CTR: {ab_ctr[-1]:.3f}
            - Contextual Bandit final CTR: {y_data[-1]:.3f}
            - Relative improvement: {improvement:.1f}%
        """)

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