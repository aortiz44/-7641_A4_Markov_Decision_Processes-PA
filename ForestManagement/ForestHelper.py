import os

import mdptoolbox.example
import numpy as np
from hiive.mdptoolbox.mdp import QLearning
import pandas as pd
from matplotlib import pyplot as plt

absolute_path = os.path.dirname(__file__)
SAVE_FIGURE_DIRECTORY = "SaveFigures/"
CSV_DIRECTORY = "CSVfiles/"
SUB_DIRECTORY = ["QL_EpsilonState/", "QL_ForestManagement/", "mdpPI/", "mdpVI/", "mdpQL/"]
BURN_PROB = 0.3


def evaluate_policy(P, R, policy, test_count=100, gamma=0.9):
    num_state = P.shape[-1]
    total_episode = num_state * test_count
    # start in each state
    total_reward = 0
    for state in range(num_state):
        state_reward = 0
        for state_episode in range(test_count):
            episode_reward = 0
            disc_rate = 1
            while True:
                # take step
                action = policy[state]
                # get next step using P
                probs = P[action][state]
                candidates = list(range(len(P[action][state])))
                next_state = np.random.choice(candidates, 1, p=probs)[0]
                # get the reward
                reward = R[state][action] * disc_rate
                episode_reward += reward
                # when go back to 0 ended
                disc_rate *= gamma
                if next_state == 0:
                    break
            state_reward += episode_reward
        total_reward += state_reward
    print("total_reward: ", total_reward)
    return total_reward / test_count


def trainQL_DataFrame(P, R, test_count=1, discount=0.9, alpha_dec=[0.99], alpha_min=[0.001],
                      epsilon=[1.0], epsilon_decay=[0.99], n_iter=[10000]):
    q_df = pd.DataFrame(columns=["Iterations", "Alpha Decay", "Alpha Min",
                                 "Epsilon", "Epsilon Decay", "Reward",
                                 "Time", "Policy", "Value Function",
                                 "Training Rewards"])

    count = 0
    rolling_avg_list = []
    x_train_rolling_axis_l = []
    roll_avg_value = 100
    # states = 2500
    # prob = 0.5

    for eps in epsilon:

        rewards_list = []
        rolling_avg = []
        for i in n_iter:
            # P, R = mdptoolbox.example.forest(S=states, r1=4, r2=2, p=prob)
            q = QLearning(P, R, discount, alpha_decay=alpha_dec[0],
                          alpha_min=alpha_min[0], epsilon=eps,
                          epsilon_decay=epsilon_decay[0], n_iter=i, run_stat_frequency=1)
            # TODO: Look at the run funciton there might be more infromation that we are not using
            run_stats = q.run()
            reward = evaluate_policy(P, R, q.policy)
            count += 1
            print("{}: {}".format(count, reward))
            st = q.run_stats
            rews = [s['Reward'] for s in st]
            print("Sum(rewards): ", np.sum(rews))
            info = [i, alpha_dec[0], alpha_min[0], eps, epsilon_decay[0], reward,
                    q.time, q.policy, q.V, rews]
            # rewards_list.append(rews)
            df_length = len(q_df.index)
            q_df.loc[df_length] = info
            # print(len(rews))
            for r in range(0, len(rews)):
                # if rews[r] > 1:
                #     print(rews[r])
                if r >= roll_avg_value and r % roll_avg_value == 0:
                    r_avg = np.average(rews[-roll_avg_value:])
                    rolling_avg.append(r_avg)

        rolling_avg_list.append(rolling_avg)
        x_train_rolling_axis_l.append((range(0, len(rolling_avg), 1)))
    return q_df, rolling_avg_list, x_train_rolling_axis_l


def mdpVI(gamma, states):
    value_f = []
    iters = []
    times = []
    P, R = mdptoolbox.example.forest(S=states,r1=4, r2=2, p=BURN_PROB)
    pi = mdptoolbox.mdp.ValueIteration(P, R, gamma)
    pi.run()
    value_f.append(evaluate_policy(P, R, pi.policy,
                                   gamma=gamma))  # valuate_policy(P, R, pi.policy, test_count=100, gamma=gamma)
    policy = pi.policy
    iters.append(pi.iter)
    times.append(pi.time)
    return value_f, times, iters, policy


def mdpPI(gamma, states):
    value_f = []
    iters = []
    times = []
    P, R = mdptoolbox.example.forest(S=states,r1=4, r2=2, p=BURN_PROB)
    pi = mdptoolbox.mdp.PolicyIteration(P, R, gamma)
    pi.run()

    value_f.append(evaluate_policy(P, R, pi.policy,
                                   gamma=gamma))  # valuate_policy(P, R, pi.policy, test_count=100, gamma=gamma)
    policy = pi.policy
    iters.append(pi.iter)
    times.append(pi.time)
    return value_f, times, iters, policy


def mdpQL(gamma, states):
    value_f = []
    iters = []
    times = []
    P, R = mdptoolbox.example.forest(S=states,r1=4, r2=2, p=BURN_PROB)
    pi = mdptoolbox.mdp.QLearning(P, R, gamma)
    pi.run()
    value_f.append(evaluate_policy(P, R, pi.policy,
                                   gamma=gamma))  # valuate_policy(P, R, pi.policy, test_count=100, gamma=gamma)
    policy = pi.policy
    # iters.append(pi.max_iter)
    iters.append(0)
    times.append(pi.time)
    return value_f, times, iters, policy


def run_gamma_diff(state_range, gamma_value, mdp_function):
    convergence_iters = []
    convergence_time = []
    convergence_rewards = []
    policy_list = []
    for i in state_range:
        training_rewards, training_times, training_iters, policy = mdp_function(gamma_value, i)
        convergence_iters.append(training_iters)
        convergence_time.append(training_times)
        convergence_rewards.append(training_rewards)
        policy_list.append(policy)

    return convergence_iters, convergence_time, convergence_rewards, policy_list


def runPI_VI(mdp_function, title, state_range=range(2, 50, 5), sub_directory=""):
    convergence_reward_list = []
    convergence_list = []
    convergence_time_list = []
    policy_list = []

    gamma_list = [0.9, 0.6, 0.4, 0.1]
    for g_value in gamma_list:
        convergence_iters, convergence_time, convergence_rewards, policy = run_gamma_diff(state_range, g_value,
                                                                                          mdp_function)
        convergence_list.append(convergence_iters)
        convergence_reward_list.append(convergence_rewards)
        convergence_time_list.append(convergence_time)
        policy_list.append(policy)

    # print(convergence_list)
    # print(len(convergence_list))
    # print(convergence_list[0])
    # print(len(convergence_list[0]))
    #
    # print(policy_list)
    # print(len(policy_list))
    # print(policy_list[0])
    # print(len(policy_list[0]))

    plot_convergence(convergence_list, state_range, title=title + " Convergence", sub_directory=sub_directory)
    plot_rewards(convergence_reward_list, state_range, title=title + " Reward", sub_directory=sub_directory)
    plot_time(convergence_time_list, state_range, title=title + " Time", sub_directory=sub_directory)


def trainQL(P, R, states=10):
    # n_iter = [25000, 50000, 100000, 125000]
    n_iter = [1000000]
    discount = 0.9
    alpha_dec = [0.99]
    alpha_min = [0.001]
    epsilon = [0.1, 0.4, 0.6, 0.9]
    # epsilon = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95]
    epsilon_decay = [0.99]

    q_df = trainQL_DataFrame(P, R, 1, discount=discount, epsilon=epsilon, n_iter=n_iter)

    print('Forest Management ' + str(states) + ' states - Q-learning Table')
    print(q_df)

    title = 'Highest Reward Epsilon ' + str(states) + ' states - QL ' + str(n_iter[-1]) + "- Max Iterations "
    title2 = "Epsilon Time for " + str(n_iter[-1]) + "- Iterations " + str(states) + " states - QL"

    relative_path = SAVE_FIGURE_DIRECTORY + SUB_DIRECTORY[0] + title + ".png"
    full_path = os.path.join(absolute_path, relative_path)

    save_pd_csv(q_df, title)
    plot_QL_df(q_df, epsilon, title, full_path)

    # plt.show()
    relative_path = SAVE_FIGURE_DIRECTORY + SUB_DIRECTORY[0] + title2 + ".png"
    full_path = os.path.join(absolute_path, relative_path)
    plot_time_EpsilonIter(q_df, n_iter, title2, full_path)


def run_Forest(states, P, R, test_count=1):
    # New way
    print('Q LEARNING WITH FOREST MANAGEMENT ' + str(states) + ' STATES ')

    # n_iter = [10000, 100000, 1000000, 10000000]
    # n_iter = [100000, 250000, 500000, 1000000]
    # n_iter = [100000, 130000, 160000, 190000]
    n_iter = [20000, 30000, 40000, 10000000]
    # n_iter = [10000]
    discount = 0.9
    alpha_dec = [0.99]
    alpha_min = [0.001]
    # epsilon = [0.05, 0.15, 0.25, 0.5, 0.75, 0.95]
    epsilon = [0.9, 0.6, 0.4, 0.1]
    epsilon_decay = [0.99]

    q_df, rolling_avg_list, x_train_rolling_axis_l = trainQL_DataFrame(P, R, discount=0.9, epsilon=epsilon,
                                                                       n_iter=n_iter)

    print("rolling_avg_list")
    print(rolling_avg_list)
    print("x_train_rolling_axis_l")
    print(x_train_rolling_axis_l)
    title_t = + str(states) + 'States Forest Management Max' + str(n_iter[-1]) + ' training '
    plot_rollin_avg_rewards(rolling_avg_list, x_train_rolling_axis_l, title=title_t + 'Rolling Avg (QL) ',
                            y_axis_label="Rolling Avg")

    # print('Forest Management ' + str(states) + ' states - Q-learning Table')
    # print(q_df)
    title = 'Forest Management ' + str(states) + ' states - Q Learning - Reward Vs Iteration'
    title2 = "Forest Management " + str(states) + " - Q Learning Time Vs Epsilon"

    relative_path = SAVE_FIGURE_DIRECTORY + SUB_DIRECTORY[1] + title + ".png"
    full_path = os.path.join(absolute_path, relative_path)
    save_pd_csv(q_df, title)
    plot_QL_df(q_df, epsilon, title, full_path)

    # plt.show()
    relative_path = SAVE_FIGURE_DIRECTORY + SUB_DIRECTORY[1] + title2 + ".png"
    full_path = os.path.join(absolute_path, relative_path)
    plot_time_EpsilonIter(q_df, n_iter, title2, full_path)

    return


def plot_rollin_avg_rewards(convergence_list, x_range, title='Template Title', sub_directory="",
                            y_axis_label="Rewards"):
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


def plot_convergence(convergence_list, state_range, title='Template Title', sub_directory=""):
    plt.plot(state_range, convergence_list[0], marker='o', label='Gamma 0.9')
    plt.plot(state_range, convergence_list[1], marker='o', label='Gamma 0.6')
    plt.plot(state_range, convergence_list[2], marker='o', label='Gamma 0.4')
    plt.plot(state_range, convergence_list[3], marker='o', label='Gamma 0.1')
    plt.xlabel('State')
    plt.title(title)
    plt.ylabel('#iterations for convergence')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(12, 6)

    relative_path = SAVE_FIGURE_DIRECTORY + sub_directory + title + ".png"
    full_path = os.path.join(absolute_path, relative_path)
    plt.savefig(full_path)

    plt.close()
    # plt.show()


def plot_rewards(convergence_list, state_range, title='Template Title', sub_directory=""):
    plt.plot(state_range, convergence_list[0], marker='o', label='Gamma 0.9')
    plt.plot(state_range, convergence_list[1], marker='o', label='Gamma 0.6')
    plt.plot(state_range, convergence_list[2], marker='o', label='Gamma 0.4')
    plt.plot(state_range, convergence_list[3], marker='o', label='Gamma 0.1')
    plt.xlabel('# States')
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


def plot_time(convergence_list, state_range, title='Template Title', sub_directory=""):
    plt.plot(state_range, convergence_list[0], marker='o', label='Gamma 0.9')
    plt.plot(state_range, convergence_list[1], marker='o', label='Gamma 0.6')
    plt.plot(state_range, convergence_list[2], marker='o', label='Gamma 0.4')
    plt.plot(state_range, convergence_list[3], marker='o', label='Gamma 0.1')
    plt.xlabel('# States')
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


def plot_time_EpsilonIter(q_df, n_iter, title2, full_path):
    fig = plt.figure()
    for frame in [q_df[(q_df.Iterations == n_iter[-1])]]:
        plt.plot(frame['Epsilon'], frame['Time'], marker='o')

    labels = ["iter=" + str(x) for x in [n_iter[-1]]]
    plt.title(title2)
    plt.legend(labels, loc='lower right')
    plt.xlabel('Epsilon')
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    plt.ylabel('Time')

    plt.savefig(full_path)
    plt.close()


def plot_QL_df(q_df, epsilon, title, full_path):
    fig = plt.figure()
    for frame in [q_df[(q_df.Epsilon == epsilon[0])], q_df[(q_df.Epsilon == epsilon[1])],
                  q_df[(q_df.Epsilon == epsilon[2])],
                  q_df[(q_df.Epsilon == epsilon[3])]]:
        plt.plot(frame['Iterations'], frame['Reward'], marker='o')

    # for frame in [q_df[(q_df.Epsilon == epsilon[0])], q_df[(q_df.Epsilon == epsilon[1])], q_df[(q_df.Epsilon ==
    # epsilon[2])], q_df[(q_df.Epsilon == epsilon[3])], q_df[(q_df.Epsilon == epsilon[4])], q_df[(q_df.Epsilon ==
    # epsilon[5])]]: plt.plot(frame['Iterations'], frame['Reward'], marker='o')

    labels = ["epsilon=" + str(x) for x in epsilon]
    plt.title(title)
    plt.legend(labels, loc='lower right')
    plt.xlabel('Iterations')
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    plt.ylabel('Reward')

    plt.savefig(full_path)


def save_pd_csv(df, title):
    relative_path = SAVE_FIGURE_DIRECTORY + CSV_DIRECTORY + title + ".csv"
    full_path = os.path.join(absolute_path, relative_path)
    df.to_csv(title + ".csv", index=False)
    return
