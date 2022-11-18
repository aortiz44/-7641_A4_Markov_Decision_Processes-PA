import numpy as np


class QLearner:
    def __init__(self, env, epsilon, alpha, gamma):
        self.N = env.observation_space.n
        self.M = env.action_space.n
        self.Q = np.zeros((self.N, self.M))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def act(self, s_t):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.M)
        else:
            return np.argmax(self.Q[s_t])

    def learn(self, s_t, a_t, r_t, s_t_next, d_t):
        a_t_next = np.argmax(self.Q[s_t_next])
        Q_target = r_t + self.gamma*(1-d_t)*self.Q[s_t_next, a_t_next]
        self.Q[s_t, a_t] = (1-self.alpha)*self.Q[s_t, a_t] + self.alpha*(Q_target)


class RandomPolicy:
    def __init__(self, env):
        self.epsilon = 0
        self.N = env.action_space.n

    def act(self, s_t):
        return np.random.choice(self.N)
