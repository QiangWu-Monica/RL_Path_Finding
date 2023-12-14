# Q Learning with $\varepsilon$-greedy

import numpy as np
import gym
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.table import Table
np.set_printoptions(precision=3)
np.random.seed(1234)

"""
For tabular Q-learning with eps-greedy exploration LN14
"""

# Comment this assert to run in other gym version.
#assert gym.__version__ == '0.17.3', 'Gym version 0.17.3 is preferred  rather than %s.' % ( gym.__version__)


def eps_greedy_policy(q_table, eps, state):
    """
    Exploration-exploitation tradeoff using Epsilon Greedy

    Parameter:
    ----------
    q_table: Q table to be exploited
    eps: exploration with probability epsilon

    Returns:
    ----------
    actions: return the action considering exporation and exploitation tradeoff

    """
    assert eps != None
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    # TODO Implement Epsilon-greedy Policy
    if np.random.uniform(0, 1) < eps:
        action = env.action_space.sample() 
    else:
        action = np.argmax(q_table[state,:])
    ############################
    return action


def tabular_q_learning(env, gamma=0.9, learning_rate=0.1, max_steps_train=1000, eps=None):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    env: Gym Environment
    exploration_policy: 
        eps_greedy_policy or boltzmann_policy
    gamma:
        Discount factor. Number in range [0, 1)
    learning_rate: float
        Learning rate for Q-learning

    max_steps_train: int
        Terminate Q-Learning when exceeding max_steps_train
    eps: float
        for epsilon greedy exploration policy
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """
    episode = 0
    max_episode = 1e5
    tol = 1e-12
    learning_rate = 0.1

    # Define lists or dictionary for plotting metrics
    discounted_return_history = []

    # Initialize the value and policy
    value_function = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    # TODO Initialize the Q_TABLE
    ############################
    q_table = np.zeros((env.nS,env.action_space.n))
    # interact with environment
    while episode < max_episode:
        episode += 1
        # Reset Environment to initial distribution
        state = env.reset()
        last_table = q_table.copy()
        discounted_return = 0
        for t in range(max_steps_train):

            ############################
            # YOUR IMPLEMENTATION HERE #
            # TODO Select Action using Epsilon-greedy
            action = eps_greedy_policy(q_table, eps,state)
            ############################
            # perform action in env
            next_state, reward, done, info = env.step(action)
            discounted_return += gamma**t*reward

            ############################
            # YOUR IMPLEMENTATION HERE #
            # TODO Update Q-table
            old_q_value = q_table[state,action]
            # Check if next_state has q values already
            # Maximum q_value for the actions in next state
            next_max = max(q_table[next_state,:])
            # Calculate the new q_value
            new_q_value = (1 - learning_rate) * old_q_value + learning_rate * (reward + gamma * next_max)
            # Finally, update the q_value
            q_table[state,action] = new_q_value
            ############################
            # environment enter into the next state
            state = next_state
            if done:
                break

        # store your cumulative reward for episode
        discounted_return_history.append(discounted_return)
        if episode % 100 == 0:
            # Log details
            template = "Discounted Return: {:.2f} at episode {}"
            print(template.format(discounted_return, episode))
        if np.max(np.abs(q_table-last_table)) < tol:
            break
    print("Training finished.\n")
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    # TODO Convert Q_table to state value function and Policy
    for state in range(env.nS):
        policy[state] = np.argmax(q_table[state,:])
        value_function[state] = eps*q_table[state,policy[state]]+(1-eps)*(q_table[state].sum()-q_table[state,policy[state]])/(env.action_space.n-1)
    ############################
    return value_function, policy, discounted_return_history


# Test and run your code. Plot is stored into img/Q1_tabular_q_learning.png
# You may change the parameters in the functions below
if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    env.seed(1234)
    gamma = 1
    V_q, policy_q, total_reward_history = tabular_q_learning(
        env, gamma=gamma, learning_rate=0.1, max_steps_train=1000, eps=1e-1)
    plt.plot(total_reward_history)
    plt.xlabel('Training Episode')
    plt.ylabel('Discounted Return')
    plt.savefig('img/Q1_tabular_q_learning.png')
    # Test your policy
    return_list = []
    for episode in range(100):
        current_state = env.reset()
        discounted_return = 0
        for t in range(1000):
            action = policy_q[current_state]
            current_state, reward, done, info = env.step(action)
            discounted_return += gamma**t*reward
            if done:
                break
        return_list.append(discounted_return)
    print('Average Discounted Return: ', np.mean(return_list))
