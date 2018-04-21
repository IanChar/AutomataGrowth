"""Experimental code for size models relating to signatures."""

from __future__ import division
from itertools import izip
from numpy.random import binomial

import sys
sys.path.append('../common')
import arbitrary_probs_util as string_util
import arbitrary_probs_calcs as calcs

class SignatureSizeModel(object):
    """Size model that works based on the states signatures."""

    def __init__(self, probs, length):
        """Constructor.
        Args:
            probs: The probabilities for the generalized strings.
            length: The length of the genrealized string to test.
        """
        self.probs = probs
        self.length = length

    def run_simulation(self, debug=False):
        """Run simulations on the size of the depths.
        Returns: List of the sizes at each state.
        """
        states = [1, 1]
        curr_signatures = set([(0,)])
        for _ in xrange(self.length - 1):
            if debug:
                print curr_signatures
            curr_signatures = self._get_child_signatures(curr_signatures)
            states.append(len(curr_signatures))
        return states

    def get_average_depth_sizes(self, num_sims):
        """Run the simulations multiple times and average over the sizes.
        Args:
            num_sims: The number of simulations to run.
        Returns: List of the average sizes at each depth.
        """
        avg = self.run_simulation()
        for _ in xrange(num_sims - 1):
            avg = [avg_s + s for avg_s, s in izip(avg, self.run_simulation())]
        return [a / num_sims for a in avg]

    def _get_child_signatures(self, sigs):
        """Get realization of random child signatures given parent signatures.
        Args:
            sigs: Set of parent signatures.
        Returns: Set of child signatures.
        """
        child_transitions = string_util.create_random_string(self.probs, 1)[0]
        child_signatures = set()

        # Collect all entries of parent's signature into one place.
        parent_ints = {}
        for sig in sigs:
            for entry in sig:
                parent_ints[entry] = 0
        # This dictionary maps letter -> parent sig entry -> child sig entry.
        letter_sig_map = {letter: dict(parent_ints)
                          for letter in child_transitions}
        # Populate what the child sig entries should be.
        for entry in parent_ints.keys():
            prev_letter = string_util.create_random_string(self.probs, 1)[0]
            for letter in child_transitions:
                if letter in prev_letter:
                    letter_sig_map[letter][entry] = entry + 1
        # Create the child signatures using this map.
        for sig in sigs:
            for letter in child_transitions:
                new_sig = set([0])
                for entry in sig:
                    new_sig.add(letter_sig_map[letter][entry])
                child_signatures.add(tuple(new_sig))
        return child_signatures

def signature_mc_sim(probs, depth):
    """Simulate the size of a signature at the depth with a Markov Chain.
    Args:
        probs: The probabilities of seeing letters.
        depth: The depth to simulate to.
    Returns: The size of the signature at the depth.
    """
    curr = 1
    c_2 = calcs.theoretical_c2(probs)
    for _ in xrange(depth):
        curr = binomial(curr, c_2) + 1
    return curr + 1

if __name__ == '__main__':
    PROBS = [0.5 for _ in range(4)]
    LENGTH = 30
    SSM = SignatureSizeModel(PROBS, LENGTH)
    SSM.run_simulation()
