import os
import time

import numpy as np
import gym
import random

from matplotlib import pyplot as plt
from numpy import savetxt

absolute_path = os.path.dirname(__file__)
SAVE_FIGURE_DIRECTORY = "SaveFigures/"


def plot_time(convergence_list, x_range, title='Template Title', sub_directory=""):
    plt.plot(x_range[0], convergence_list[0], marker='.', label='Epsilon 0.9')
    plt.plot(x_range[1], convergence_list[1], label='Epsilon 0.6')
    plt.plot(x_range[2], convergence_list[2], label='Epsilon 0.4')
    plt.plot(x_range[3], convergence_list[3], label='Epsilon 0.1')
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


def plot_rewards(convergence_list, x_range, title='Template Title', sub_directory="", y_axis_label="Rewards"):
    plt.plot(x_range[0], convergence_list[0], marker='.', label='Epsilon 0.9')
    plt.plot(x_range[1], convergence_list[1], label='Epsilon 0.6')
    plt.plot(x_range[2], convergence_list[2], label='Epsilon 0.4')
    plt.plot(x_range[3], convergence_list[3], label='Epsilon 0.1')
    plt.xlabel('# Iterations')
    plt.title(title)
    plt.ylabel(y_axis_label)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    relative_path = SAVE_FIGURE_DIRECTORY + sub_directory + title + ".png"

    full_path = os.path.join(absolute_path, relative_path)
    plt.savefig(full_path)
    plt.close()
    # plt.show()


def random_agent(env):
    num_steps = 99
    for s in range(num_steps + 1):
        print(f"step: {s} out of {num_steps}")

        # sample a random action from the list of available actions
        action = env.action_space.sample()

        # perform this action on the environment
        env.step(action)

        # print the new state
        env.render()

    # end this instance of the taxi environment
    env.close()


def train_ql(env, num_episodes, max_steps, epsilon, qtable, learning_rate, discount_rate, decay_rate):
    reward_list = []
    rolling_avg = []
    total_time = []
    # training
    for episode in range(num_episodes):
        old_qtable = qtable.copy()

        # reset the environment
        state = env.reset()
        done = False
        max_step_reward = []
        start_time = time.time()

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            # print("epsilon: ", epsilon)
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state, :])

            # take action and observe reward
            new_state, reward, done, info = env.step(action)

            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            # Update to our new state
            state = new_state
            max_step_reward.append(reward)
            # if done, finish episode
            if done:
                break

        reward_list.append(max_step_reward[-1])
        total_time.append((time.time() - start_time))
        # print("qtable - old_qtable")
        # print(np.argmax(qtable, axis=1))
        # print(np.round(qtable, 3))
        # if len(reward_list) >= 100:
        #     print(f"average score: {np.average(reward_list[-100])}")

        # print("Episode: ", episode)
        r_avg = np.average(reward_list[-100:])
        # print("Percentage of reaching goal",r_avg )
        rolling_avg.append(r_avg)

        # print(qtable - old_qtable)
        # print(np.sum(np.abs(qtable - old_qtable)))
        # Decrease epsilon
        # epsilon = np.exp(-decay_rate * episode)
        epsilon = (1 - decay_rate) * epsilon
        epsilon = max(epsilon, .001)

    return qtable, reward_list, rolling_avg, total_time


def run_ql(env, map_size, num_runs=100000):
    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.9
    epsilon = 1.0
    decay_rate = 0.001
    discount_rate_list = [0.1, 0.4, 0.6, 0.9]
    epsilon_rate_list = [0.9, 0.6, 0.4, 0.1]
    # epsilon_rate_list = [0.9]

    # training variables
    num_episodes = 100000
    max_train_steps = 1000  # per episode
    max_test_steps = num_runs
    reward_l = []

    x_axis_l = []
    x_t_axis_l = []
    total_time_l = []
    training_r_list = []
    x_train_axis_l = []
    rolling_avg_r_list = []
    x_train_rolling_axis_l = []
    train_time_list = []
    train_time_x_axis_list = []

    for e in epsilon_rate_list:
        qtable = np.zeros((state_size, action_size))

        # training

        qtable, training_reward_list, rolling_avg, train_time = train_ql(env, num_episodes, max_train_steps, e, qtable,
                                                                         learning_rate,
                                                                         discount_rate, decay_rate)
        training_r_list.append(training_reward_list)
        x_train_axis_l.append(range(0, len(training_reward_list), 1))

        rolling_avg_r_list.append(rolling_avg)
        x_train_rolling_axis_l.append((range(0, len(rolling_avg), 1)))

        train_time_list.append(train_time)
        train_time_x_axis_list.append((range(0, len(train_time), 1)))

        # print(f"Training completed over {num_episodes} episodes")
        # input("Press Enter to watch trained agent...")
        tot_reward, total_time = test_ql(env, qtable, max_test_steps)

        reward_l.append(tot_reward)
        x_axis_l.append(range(0, len(tot_reward), 1))
        x_t_axis_l.append(range(0, len(total_time), 1))
        total_time_l.append(total_time)

    title_t = map_size + " Frozen-Lake " + str(num_episodes) + " Training " + str(
        max_test_steps) + "Testing iterations "
    plot_rewards(reward_l, x_axis_l, title=title_t + ' Reward (QL)', y_axis_label="Cumulative Rewards")
    plot_time(total_time_l, x_t_axis_l, title=title_t + ' Time (QL)')

    savetxt(title_t + ' Reward (QL)' + '.csv', np.asarray(reward_l), delimiter=',')
    savetxt(title_t + ' Time (QL)' + '.csv', np.asarray(total_time_l), delimiter=',')

    title_t = map_size + " Frozen-Lake " + str(num_episodes) + " iterations "
    plot_rewards(training_r_list, x_train_axis_l, title=title_t + ' Reward (QL) Training', y_axis_label="Rewards")
    plot_rewards(rolling_avg_r_list, x_train_rolling_axis_l, title=title_t + ' Rolling Avg (QL) Training Rolling Avg',
                 y_axis_label="Rolling Avg")
    plot_time(train_time_list, train_time_x_axis_list, title=title_t + ' Time (QL) Training')

    savetxt(title_t + ' Reward (QL) Training' + '.csv', np.asarray(training_r_list), delimiter=',')
    savetxt(title_t + ' Rolling Avg (QL) Training Rolling Avg' + '.csv', np.asarray(rolling_avg_r_list), delimiter=',')
    savetxt(title_t + ' Time (QL) Training' + '.csv', np.asarray(train_time_list), delimiter=',')

    env.render()
    env.close()

    pass


def test_ql(env, qtable, max_test_steps):
    # watch trained agent
    state = env.reset()
    step_list = range(max_test_steps)

    total_time = []
    tot_reward = [0]
    start_time = time.time()

    for s in step_list:
        # print(f"TRAINED AGENT")
        # print("Step {}".format(s + 1))

        observation = 0
        done = False
        env.reset()
        # render and env.render()
        reward_per_run = 0

        while not done:
            action = np.argmax(qtable[state, :])
            new_state, reward, done, info = env.step(action)
            reward_per_run += reward
            state = new_state
            # render and env.render()
        # env.render()
        # print(f"score: {reward_per_run}")

        tot_reward.append(reward_per_run + tot_reward[-1])
        # tot_reward.append(reward_per_run)
        total_time.append((time.time() - start_time))

        #
        # if done:
        #     break

    return tot_reward, total_time


def run_test(num_runs=100000):
    # map_size = "8x8"
    map_size = "4x4"
    # env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), map_name=map_size, is_slippery=True)
    env = gym.make('FrozenLake-v1', desc=None, map_name=map_size, is_slippery=True)

    # env = gym.make('Taxi-v3')
    # create a new instance of taxi, and get the initial state
    state = env.reset()
    env.render()
    # random_agent(env)
    run_ql(env, map_size, num_runs)

    # Convergence criteria for step function
    # target_reward = reward + self.gamma * np.max(self.Q[next_state])
    # self.Q[state][action] += self.alpha * (target_reward - self.Q[state][action])
    # if done:
    #     self.episode += 1
    #     self.epsilon = 1 / self.episode

    return


run_test()
