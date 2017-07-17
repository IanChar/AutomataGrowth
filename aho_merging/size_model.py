"""File for exploring stoachastic process to model growth of the minimal DFA."""

from __future__ import division

from random import random

class SizeModel(object):
    """Class for modeling the growth of the minimal DFA."""

    def __init__(self, new_thread_prob, offspring_dist):
        """Constructor for SizeModel.
        Args:
            new_thread_prob: The probability of seeing a new thread at a
                particular level.
            offspring_dist: Probability vector for how many children one thread
                will produce after each level.
        """
        if abs(sum(offspring_dist) - 1) > 1e-6:
            raise ValueError("Not given valid probability vector.")
        self.new_thread_prob = new_thread_prob
        self.offspring_dist = offspring_dist

    def do_sims(self, num_levels, trials):
        """Simulate the number of states at each level, average over trials.
        Args:
            num_levels: The number of levels to simulate for.
            trials: The number of trials to perform.
        Returns: List with average number of states at each level.
        """
        to_return = [0 for _ in xrange(num_levels)]
        for _ in xrange(trials):
            curr = 0
            for level in xrange(1, num_levels):
                curr = self._sim_next(curr)
                to_return[level] += curr
        return [val / trials for val in to_return]

    def get_offspring_mean(self):
        """Get the expected value for the offspring distribution.
        Returns: Expected value of the offspring distribution.
        """
        return sum([i * val for i, val in enumerate(self.offspring_dist)])

    def get_asymptotic_state_mean(self):
        """Get the expected value of how many states we expect to be in a level
        in the limit.
        Returns: The expected value.
        """
        offspring_mean = self.get_offspring_mean()
        return self.new_thread_prob * (1 / (1 - offspring_mean))

    def _sim_next(self, last):
        """Simulate the next level given the last one.
        Args:
            last: The number of states in the last level.
        Returns: The number of states in the next level.
        """
        total_states = 0
        total_states = (total_states + 1 if random() <= self.new_thread_prob
                        else total_states)
        for _ in xrange(last):
            rand = random()
            num_to_add = 0
            prob_accum = self.offspring_dist[num_to_add]
            while rand > prob_accum:
                num_to_add += 1
                prob_accum += self.offspring_dist[num_to_add]
            total_states += num_to_add
        return total_states
