
class MarkovDecisionProcess():
    def __init__(self, num_states, num_actions, dynamics_fn):
        """
        Initializes an object representing a Markov Decision Process.
        Assumes the reward is deterministic for a given state.
        """
        assert num_states > 0 and num_actions > 0
        self.num_states = num_states
        self.num_actions = num_actions
        # P[s][a] represents a list of possible transistions given state s and a.
        # each transistion is expected as a list/tuple:
        # [prob_next_state, next_state, reward, is_terminal]
        self.P = dynamics_fn
        # sanity checks
        self.__verify()

    def __verify(self):
        assert len(self.P) == self.num_states
        for s in self.P.keys():
            assert len(self.P[s]) == self.num_actions
        for s in self.P.keys():
            for a in self.P[s].keys():
                transitions = self.P[s][a]
                p_sum = sum([t[0] for t in transitions])
                assert 1 >= p_sum > 0.99
