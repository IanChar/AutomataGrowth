"""Analyze clashes in relation to last level's size."""

from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter

import aho_construction
import aho_test
import clash_analysis

def get_conditioned_clash(clash_level, samples, alphabet, max_trials):
    """Get the clash distribution given that the last level was a specific size.

    Args:
        clash_level: The level at which to analyze the clash distributions.
        samples: The number of samples that each conditioned value should have.
        alphabet: Size of the alphabet to use for construction.
        max_trials: The maximum number of samples to run to get the desired
            number of samples.
    """
    seen_amount = {}
    clash_data = {}

    # Continuously loop until we have enough data.
    running = True
    trial_num = 0
    while running:
        rand_string = aho_test.build_random_string(clash_level, alphabet)
        root = aho_construction.construct_dfa(rand_string, alphabet)
        lvl_sizes = clash_analysis.get_level_sizes(root, clash_level)
        # Check if we need anymore samples
        if lvl_sizes[-2] not in seen_amount:
            seen_amount[lvl_sizes[-2]] = 1
            clash_data[lvl_sizes[-2]] = []
        elif seen_amount[lvl_sizes[-2]] < samples:
            seen_amount[lvl_sizes[-2]] += 1
            # Compute the clashes and add just the first clash number.
            clashes = clash_analysis.get_clashes(root)
            clash_data[lvl_sizes[-2]].append(clashes[-1][0])
        # See if we need to continue to run.
        trial_num += 1
        running = trial_num < max_trials
        if running:
            for amount in seen_amount.values():
                if amount < samples:
                    running = True
                    break
    # Trim out the data that didn't meet the sample threshold.
    for size, amount in seen_amount.iteritems():
        if amount < samples:
            del clash_data[size]
    return clash_data

def plot_conditioned_clash(clash_level, samples, alphabet, max_trials):
    """Plot the clash distribution given that the last level was some size.

    Args:
        clash_level: The level at which to analyze the clash distributions.
        samples: The number of samples that each conditioned value should have.
        alphabet: Size of the alphabet to use for construction.
        max_trials: The maximum number of samples to run to get the desired
            number of samples.
    """
    clash_data = get_conditioned_clash(clash_level, samples, alphabet,
                                       max_trials)
    for conditioned_size, data in clash_data.iteritems():
        max_clash = max(data)
        _, _, _ = plt.hist(data, 50, alpha=0.75)
        plt.xlabel('Clash Number')
        plt.xlim((1, max_clash + 1))
        plt.ylabel('Frequency')
        plt.title('Clash Number at Level ' + str(clash_level)
                  + '; Last level Size: ' + str(conditioned_size)
                  + '; Samples: ' + str(samples))
        plt.grid(True)
        plt.show()

def _compute_chisquare(observed, expected):
    """Computes the Chi-Square score for two given histograms.
       i.e. sum((O_i - E_i)^2/E_i)

    Args:
        observed: The observed histogram to analyze represented as a dictionary.
            The dictionary maps size -> frequency.
        expected: The expected histogram to analyze represented as a dictionary.
            The dictionary maps size -> frequency.
    Returns: A numeric value representing the Chi-Square score.
    """
    score = 0
    scaling_factor = sum(observed.values()) / sum(expected.values())
    for size, exp_freq in expected.iteritems():
        obs_freq = observed[size] if size in observed else 0
        score += ((obs_freq - exp_freq * scaling_factor) ** 2
                  / (exp_freq * scaling_factor))
    return score

def get_chisquare_mat(clash_level, samples, alphabet, max_trials):
    """Get a matrix with the Chi-Square scores from the histograms.
       sum((O_i - E_i)^2/E_i)

    Args:
        clash_level: The level at which to analyze the clash distributions.
        samples: The number of samples that each conditioned value should have.
        alphabet: Size of the alphabet to use for construction.
        max_trials: The maximum number of samples to run to get the desired
            number of samples.
    Returns: A tuple containing two things. The firs is a numpy matrix of the
        chi-square scores where m_i,j is the score with the ith and jth size of
        the previous level. The second is a dictionary with:
        dict[conditioned_size] -> Number of Clashes observed.
    """
    # Get the data and convert data to show frequency of item.
    clash_data = get_conditioned_clash(clash_level, samples, alphabet,
                                       max_trials)
    hists = {size: dict(Counter(data)) for size, data in clash_data.iteritems()}

    # Initialize matrix and create the mappings to the level sizes.
    mat_mappings = hists.keys()
    mat_mappings.sort()
    mat = np.zeros((len(hists), len(hists)))

    # Go through the matrices and compute the chi square values.
    for row in range(len(hists)):
        for col in range(len(hists)):
            if row != col:
                score = _compute_chisquare(hists[mat_mappings[row]],
                                           hists[mat_mappings[col]])
                mat[row][col] = score
    # Turn mat_mappings into a dictionary with number of possible clashes.
    mat_mappings = {size: len(hists[size].keys()) for size in mat_mappings}
    return mat, mat_mappings

if __name__ == '__main__':
    print get_chisquare_mat(15, 1000, clash_analysis.DNA_ALPH, 100000)
