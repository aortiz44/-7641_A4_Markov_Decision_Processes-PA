import os
import time
import timeit
import pandas as pd
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym.envs.toy_text.frozen_lake import generate_random_map

from MDPClass import MarkovDecisionProcess as MDP
from Agent import Agent
from OtherExample import OtherExample
from QLearn import QLearner, RandomPolicy
from numpy import genfromtxt

absolute_path = os.path.dirname(__file__)
SAVE_FIGURE_DIRECTORY = "SaveFigures/"


def part0():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    env.render()
    mdp = MDP(env.observation_space.n, env.action_space.n, env.unwrapped.P)

    print("Number of states ", mdp.num_states)
    print("Number of actions ", mdp.num_actions)

    sample_actions = {'LEFT': 0, 'UP': 3}
    sample_states = [0, 11, 15]

    for s in sample_states:
        for a in sample_actions.keys():
            print(f"Transitions for state {s} and action {a} are\n ", mdp.P[s][sample_actions[a]])
            # 4-tuple: (probability_of_transistion, next_state, reward, is_episode_end)

    return


def run_PI_VI(env, agent, num_runs=1, render=False):
    tot_reward = [0]
    total_time = []
    start_time = time.time()
    for _ in range(num_runs):
        observation = 0
        done = False
        env.reset()
        # render and env.render()
        reward_per_run = 0
        while not done:
            action = agent.get_action(observation)
            observation, reward, done, info = env.step(action)
            reward_per_run += reward
            # render and env.render()
        # env.close()
        # if done:
        #     break
        tot_reward.append(reward_per_run + tot_reward[-1])
        total_time.append((time.time() - start_time))
    return tot_reward, total_time


def experiment(num_runs=10000):
    map_size = "4x4"
    env = gym.make('FrozenLake-v1', desc=None, map_name=map_size, is_slippery=True)
    # env = gym.make('Taxi-v3')
    # env = gym.make('CliffWalking-v0')

    env.reset()
    env.render()
    # state = env.reset()
    mdp = MDP(env.observation_space.n, env.action_space.n, env.unwrapped.P)
    # Gamma is discount_rate
    dis_rate_list = [0.1, 0.4, 0.6, 0.9]
    rewards_by_discount_1 = []
    time_list_1 = []
    rewards_by_discount_2 = []
    time_list_2 = []
    title = map_size + " Frozen-Lake " + str(num_runs) + " iterations "
    for g in dis_rate_list:
        agent1 = Agent(mdp, discount_rate=g, theta=0.000001)
        agent1.policy_iteration()
        cumulative_rewards1, total_time1 = run_PI_VI(env, agent1, num_runs)
        rewards_by_discount_1.append(cumulative_rewards1)
        time_list_1.append(total_time1)

    for ga in dis_rate_list:
        agent2 = Agent(mdp, discount_rate=ga, theta=0.000001)
        agent2.value_iteration()
        cumulative_rewards2, total_time2 = run_PI_VI(env, agent2, num_runs)
        rewards_by_discount_2.append(cumulative_rewards2)
        time_list_2.append(total_time2)

    plot_rewards(rewards_by_discount_1, num_runs, title=title + " Reward (Policy)", sub_directory="")
    plot_rewards(rewards_by_discount_2, num_runs, title=title + " Reward (Value)", sub_directory="")

    plot_time(time_list_1, num_runs, title=title + " Time (Policy)", sub_directory="")
    plot_time(time_list_2, num_runs, title=title + " Time (Value)", sub_directory="")

    OtherExample.run_test(num_runs)
    # plot_reward_iteration(map_size, num_runs, cumulative_rewards1, cumulative_rewards2)
    # plot_optimal_value(agent1, (4, 4), "Policy Iteration Optimal Value ")
    # plot_optimal_value(agent2, (4, 4), "Value Iteration Optimal Value ")


def iteration_time_PI_VI():
    policy_iteration_setup = '''
    import gym
    from MarkovDecisionProcess import MarkovDecisionProcess as MDP
    from Agent import Agent
    env = gym.make('FrozenLake-v0')
    env.reset()
    mdp = MDP(env.observation_space.n, env.action_space.n, env.unwrapped.P)
    agent = Agent(mdp, 1, 0.000001)
    '''
    policy_iteration_code = "agent.policy_iteration()"
    value_iteration_setup = '''
    import gym
    from MarkovDecisionProcess import MarkovDecisionProcess as MDP
    from Agent import Agent
    env = gym.make('FrozenLake-v0')
    env.reset()
    mdp = MDP(env.observation_space.n, env.action_space.n, env.unwrapped.P)
    agent = Agent(mdp, 1, 0.000001)
    '''
    value_iteration_code = "agent.value_iteration()"

    num_runs_options = [1, 10, 50, 100, 500, 1000]
    pi_time = []
    vi_time = []
    for num_runs in num_runs_options:
        pi_time.append(timeit.timeit(setup=policy_iteration_setup, stmt=policy_iteration_code, number=num_runs))
        vi_time.append(timeit.timeit(setup=value_iteration_setup, stmt=value_iteration_code, number=num_runs))

    print(pi_time, vi_time)
    plot_time_iter(num_runs_options, pi_time, vi_time)


def plot_time_iter(num_runs_options, pi_time, vi_time):
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and an axes.
    ax.plot(num_runs_options, pi_time, label="Policy iteration:")
    ax.plot(num_runs_options, vi_time, label="Value iteration:")
    ax.grid(False)
    ax.set_xlabel('Runs')  # Add an x-label to the axes.
    ax.set_ylabel('Time taken')  # Add a y-label to the axes.
    ax.set_title("Execution Time: Policy Iteration vs Value Iteration")  # Add a title to the axes.
    ax.legend()  # Add a legend.


def plot_time(convergence_list, num_runs, title='Template Title', sub_directory=""):
    x_range = range(0, num_runs)
    plt.plot(x_range, convergence_list[0], marker='.', label='Gamma 0.9')
    plt.plot(x_range, convergence_list[1], label='Gamma 0.6')
    plt.plot(x_range, convergence_list[2], label='Gamma 0.4')
    plt.plot(x_range, convergence_list[3], label='Gamma 0.1')
    plt.xlabel('# Iterations')
    plt.title(title)
    plt.ylabel('Time')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    relative_path = SAVE_FIGURE_DIRECTORY + sub_directory + title + ".png"

    full_path = os.path.join(absolute_path, relative_path)
    plt.savefig(full_path)
    plt.close()
    # plt.show()


def plot_rewards(convergence_list, num_runs, title='Template Title', sub_directory=""):
    x_range = range(0, num_runs + 1)
    plt.plot(x_range, convergence_list[0], marker='.', label='Gamma 0.9')
    plt.plot(x_range, convergence_list[1], label='Gamma 0.6')
    plt.plot(x_range, convergence_list[2], label='Gamma 0.4')
    plt.plot(x_range, convergence_list[3], label='Gamma 0.1')
    plt.xlabel('# Iterations')
    plt.title(title)
    plt.ylabel('Rewards')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    relative_path = SAVE_FIGURE_DIRECTORY + sub_directory + title + ".png"

    full_path = os.path.join(absolute_path, relative_path)
    plt.savefig(full_path)
    plt.close()
    # plt.show()


def part1():
    print("Part1")
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    # env = gym.make('Taxi-v3')
    # env = gym.make('CliffWalking-v0')

    env.reset()
    env.render()
    # state = env.reset()
    mdp = MDP(env.observation_space.n, env.action_space.n, env.unwrapped.P)

    print("Number of states ", mdp.num_states)
    print("Number of actions ", mdp.num_actions)

    value_agent = Agent(mdp, 1.0, 0.000001)
    policy_agent = Agent(mdp, 1.0, 0.000001)
    print("####################  Policy Agent ####################")
    policy_agent.policy_iteration()
    print("policy_agent.mdp: ", policy_agent.mdp)
    print("policy_agent.value_fn: ", policy_agent.value_fn)
    print("policy_agent.theta: ", policy_agent.theta)
    print("policy_agent.policy: ", policy_agent.policy)
    print("policy_agent.discount_rate: ", policy_agent.discount_rate)
    policy_agent.print_agent_info()

    print("####################  #################### ####################")

    print("####################  Value Agent ####################")
    value_agent.value_iteration()
    value_agent.print_agent_info()
    print("####################  #################### ####################")

    # agent.policy[0] = [1, 0, 0, 0]
    # assert agent.get_action(0) == 0
    # agent.policy[14] = [0, 0, 1, 0]
    # assert agent.get_action(14) == 2
    # agent.policy[2] = [0.5, 0.5, 0, 0]
    # assert agent.get_action(2) in [0, 1]
    return


def basicloop():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    s_t = env.reset()
    for t in range(1000):
        # no policy defined, just randomly sample actions
        a_t = env.action_space.sample()
        s_t, r_t, d_t, _ = env.step(a_t)
        if d_t:
            s_t = env.reset()
        env.step(env.action_space.sample())
        env.render()
    pass


def read_csv_nparray():
    num_runs = 100 - 1
    map_size = "4x4"
    title = "QL " + map_size + " Frozen-Lake " + str(num_runs) + " iterations "
    my_data = genfromtxt('Convergence_list.csv', delimiter=',')
    print(my_data)
    plot_rewards(my_data, num_runs, title=title + " Reward (Policy)", sub_directory="")


def run_experiment():
    num_runs = 100000
    experiment(num_runs)


# test0()
# part1()
run_experiment()

# read_csv_nparray()
# basicloop()
