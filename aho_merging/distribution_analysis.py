"""Code to analyze the distribution of number of states for DFA."""
from __future__ import division
import pandas as pd

import sys
sys.path.append('..')

import merge_alg
import arbitrary_probs_util as string_util

class DistributionGroup(object):
    """Object containing multiple StateDistributions."""

    def __init__(self, params):
        """Constructor for DistributionGroup.
        Args:
            params: List of tuples of (length, probs) for each of the
                StateDistributions.
        """
        self.distributions = [StateDistribution(length, probs)
                              for length, probs in params]

    def draw_samples(self, num_samples):
        """Draws samples for each of the distributions.
        Args:
            num_samples: The number of samples to draw for each distribution.
        """
        for dist in self.distributions:
            dist.draw_samples(num_samples)

    def get_means_by_level(self):
        """Gets means for each distribution, sorted by increasing level.
        Returns: List of means sorted by increasing level.
        """
        pairs = [(dist.length, dist.mean) for dist in self.distributions]
        pairs.sort()
        return [pair[1] for pair in pairs]

    def get_all_samples(self):
        """Returns all samples in a pandas dataframe.
        Returns: Samples with length, probabilities, and number of states.
        """
        length_list = []
        probs_list = []
        num_states_list = []
        for dists in self.distributions:
            for samp in dists.samples:
                length_list.append(dists.length)
                probs_list.append(dists.probs)
                num_states_list.append(samp)
        return pd.DataFrame({'length': length_list, 'probs': probs_list,
                             'num_states': num_states_list})


class StateDistribution(object):
    """Object representing distribution for number of states given particular
    string length and probabilities of letters."""

    def __init__(self, length, probs):
        """Constructor for StateDistribution.
        Args:
            length: Length of the generalized strings.
            probs: The probabilities of seeing a particular letter.
        """
        self.length = length
        self.probs = probs
        self.samples = []
        self.mean = None
        self.conf_interval_map = {}
        self.approx_dist = None

    def draw_samples(self, num_samples):
        """Draws random samples from the distribution by running the algorithm.
        Args:
            num_samples: Number of samples to draw from the distribution.
        """
        new_samples = []
        for _ in xrange(num_samples):
            alph = string_util.get_default_alphabet(len(self.probs))
            gen_string = string_util.create_random_string(self.probs,
                                                          self.length)
            new_samples.append(merge_alg.aho_merge(gen_string, alph)
                               .get_num_states())
        self._update_mean(new_samples)
        self.samples.extend(new_samples)

    def _update_mean(self, new_samples):
        """Updates the mean after new samples have been drawn.
        Args:
            new_samples: The new samples that were just draw.
        """
        if self.mean is None:
            self.mean = sum(new_samples) / len(new_samples)
        else:
            self.mean = ((sum(new_samples) + self.mean * len(self.samples))
                         / (len(new_samples) + len(self.samples)))

if __name__ == '__main__':
    probs = [0.5, 0.5, 0.5, 0.5]
    group = DistributionGroup([(3, probs), (4, probs), (5, probs)])
    group.draw_samples(20)
    print group.get_means_by_level()
    print group.get_all_samples()
