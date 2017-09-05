"""Markov chain model for the number of links in a failure chain."""

from __future__ import division
from random import random

class FailureChain(object):
    """Class representing the MC that models Failure Chain length."""

    def __init__(self, probs):
        """Args:
            probs: The probabilities of seeing each letter in G.
        """
        # Alpha is the probability that an arbitrary letter is in an arbitrary
        # set.
        self._alpha = _get_alpha(probs)
        # Cache for the geometric distribution probabilities of falling down
        # the failure chain. (This is L in my notes).
        self._geom_cache = []
        self._transitions = [[1 - self._alpha, self._alpha]]

    def get_stationary_dist(self, depth, num_realizations):
        """Get the stationary distribution.
        Args:
            depth: The depth to go to before checking.
            num_realizations: The number of times to run the MC.
        Returns: Probability vector represented as a list.
        """
        pass

    def get_realization_path(self, depth):
        """Get the history of where one realization went to.
        Args:
            depth: The depth to go to.
        Returns: List of the states visited.
        """
        pass

    def print_transition_mat(self):
        """Print out probability transition matrix."""
        print '---------------PROBABILITY TRANSITION MATRIX--------------'
        for row in self._transitions:
            for col in row:
                print ('%f ' %col),
            print ''

    def _transition(self, curr):
        """Randomly transition from current state to next state.
        Args:
            curr: Current state's number.
        Returns: Number of next state.
        """
        if len(self._transitions) < curr + 1:
            self._compute_next_transition_row()
        dist = self._transitions[curr - 1]
        unif = random()
        accum = 0
        for index, prob in enumerate(dist):
            accum += prob
            if unif <= accum:
                return index + 1
        # If we didn't ever return, it is probably because of machine error?
        return curr + 1

    def _compute_next_transition_row(self):
        """Computes the next needed row in the transition matrix."""
        row_num = len(self._transitions) + 1
        self._transitions.append([self._compute_entry(nxt_ind + 1, row_num)
                                  for nxt_ind in range(row_num + 1)])

    def _compute_entry(self, nxt, curr):
        """Compute the entry of the transition matrix.
        Args:
            nxt/curr: Probability of going to next state from curr.
            geom_dist: Distribution of falling index + 1 in the chain.
        Returns: The probability at that entry.
        """
        # Check if makes sense.
        if nxt > curr + 1:
            return 0
        # Check if cached.
        if curr > 0 and curr <= len(self._transitions):
            return self._transitions[curr - 1][nxt - 1]
        # Boundary conditions.
        if curr == 0:
            return (1 - self._alpha) if nxt == 1 else self._alpha
        if nxt == 1:
            return (1 - self._alpha) ** curr

        to_return = 0
        geom_dist = self._get_geom_dist(curr)
        for fall, fall_prob in enumerate(geom_dist):
            to_return += fall_prob * self._compute_entry(nxt - 1,
                                                         curr - (fall + 1))
        return to_return


    def _get_geom_dist(self, up_to):
        """Get the finite geometric distribution from the cache.
        Args:
            up_to: The amount we can fall in the distribution.
        Returns: probability vector as a list where [0] is fell once. Last index
            is where we fell all the way down.
        """
        curr_cache_size = len(self._geom_cache)
        while curr_cache_size < up_to - 1:
            self._geom_cache.append((1 - self._alpha) ** curr_cache_size
                                    * self._alpha)
            curr_cache_size += 1
        to_return = self._geom_cache[:up_to - 1]
        to_return.append(1 - sum(to_return))
        return to_return

def _get_alpha(probs):
    """Get probability that our randomly selected letter is in another set.
    Args:
        probs: The list of probabilities.
    Returns: The probability alpha.
    """
    to_return = 0
    for letter_ind, letter_prob in enumerate(probs):
        to_return += letter_prob * _get_prob_letter_in_set(probs, letter_ind)
    return to_return

def _get_prob_letter_in_set(probs, letter_index, accum=None, nxt_ind=None,
                            set_size=None):
    """Get the probability of selecting some letter from an arbitrary set.
    Args:
        probs: The probability of seeing each letter in G.
        letter_index: The index of the letter we want to the prob for.
    Returns: The probability.
    """
    if accum is None:
        accum, nxt_ind, set_size = 1, 0, 0
    print accum, nxt_ind, set_size
    if nxt_ind == letter_index:
        accum *= probs[letter_index]
        nxt_ind += 1
        set_size += 1
    if nxt_ind == len(probs):
        prob_empty = 1
        for prob in probs:
            prob_empty *= (1 - prob)
        return accum / (set_size * (1 - prob_empty))
    else:
        is_in = _get_prob_letter_in_set(probs, letter_index,
                                        accum * probs[nxt_ind], nxt_ind + 1,
                                        set_size + 1)
        not_in = _get_prob_letter_in_set(probs, letter_index,
                                         accum * (1 - probs[nxt_ind]),
                                         nxt_ind + 1, set_size)
        return is_in + not_in

if __name__ == '__main__':
    FC = FailureChain([0.5 for _ in range(4)])
    for _ in range(3):
        FC._compute_next_transition_row()
    FC.print_transition_mat()
