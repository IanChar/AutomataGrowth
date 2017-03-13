""" Analyzes matching substrings with prefixes for failure function. """
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

import aho_test

COLORS = ['b', 'g', 'r', 'c', 'm', 'y']

def get_theoretical_ratio_disjoint(alph):
    """ Returns the theoretical disjoint value given the alphabet size.

    Args:
        alph: The size of the alphabet.
    Returns:
        Value between 0 and 1 representing the ratio of disjoint observed.
    """
    to_return = 3 ** alph - 2 ** (alph + 1) + 1
    return to_return / ((2 ** alph - 1) ** 2)

def get_theoretical_substring_prob(sub_len, alph, string_len):
    """ Returns the theoretical prob of having a substring of given length.

    Args:
        sub_len: The length of the matching substring.
        alph: The size of the alphabet.
        string_len: The length of the string.
    Returns: Probability of having a matching substring of length sub_len.
    """
    to_return = (1 - get_theoretical_ratio_disjoint(alph)) ** sub_len
    to_return = 1 - (1 - to_return) ** (string_len - sub_len)
    return to_return


def ratio_disjoint(trials, alphabet_size):
    """Finds the ratio of disjoint generalized string letters.

    Args:
        trials: The number of trials to perform.
        alphabet_size: The size of the alphabet.
    Returns:
        The ratio of number of times two generalized strings were disjoint.
    """
    # Create a alphabet of appropriate size.
    alphabet = [chr(ord('A') + i) for i in range(alphabet_size)]

    num_disjoint = 0
    for _ in xrange(trials):
        # Generate two different potential generalized string letters.
        letters = [set(aho_test.build_random_string(1, alphabet)[0])
                   for _ in range(2)]
        if len(letters[0].intersection(letters[1])) == 0:
            num_disjoint += 1
    return num_disjoint / trials

def num_overalapping_substrings(sub_len, string_len, alph_size):
    """Find the number of overlapping substrings match with the prefix.

    Args:
        sub_len: The length of the substring to check.
        string_len: The size of the string to test on.
        alph_size: The size of the alphabet the string is built with.
    Returns:
        The number of substrings of length sub_len match the prefix.
    """
    # Create a alphabet of appropriate size.
    alphabet = [chr(ord('A') + i) for i in range(alph_size)]
    # Create a string to analyze and convert to list of sets.
    string = aho_test.build_random_string(string_len, alphabet)
    string = [set(letter) for letter in string]

    num_substrings = 0
    for start_index in range(1, string_len - sub_len):
        is_substring = True
        for sub_index in range(sub_len):
            if len(string[sub_index].intersection(string[start_index
                                                         + sub_index])) == 0:
                is_substring = False
                break
        if is_substring:
            num_substrings += 1
    return num_substrings

def matching_substring_hist(trials, sub_len, string_len, alph_size):
    """Plot a histogram of number of substrings against theoretical.

    Args:
        trials: Number of trials for the histogram.
        sub_len: The length of the matching substring to look at.
        string_len: The length of strings to look at.
        alph_size: The size of the alphabet to use.
    """
    # Generate observed values.
    observed = []
    for _ in xrange(trials):
        observed.append(num_overalapping_substrings(sub_len, string_len,
                                                    alph_size))
    # Generate values from theoretical dist.
    num_trials = string_len - sub_len
    success_prob = (1 - get_theoretical_ratio_disjoint(alph_size)) ** sub_len
    theoretical = []
    for _ in xrange(trials):
        theoretical.append(np.random.binomial(num_trials, success_prob))

    # Make a histogram of the two
    bins = range(num_trials)
    plt.hist(observed, bins, alpha=0.5, label='Observed')
    plt.hist(theoretical, bins, alpha=0.5, label='Theoretical')
    plt.legend(loc='upper right')
    plt.title('Distribution of Number Matching Substrings'
              + '; String Length: ' + str(string_len)
              + '; Substring Length: ' + str(sub_len)
              + '; Alphabet Size: ' + str(alph_size)
              + '; Trials: ' + str(trials))
    plt.xlabel('Number Substrings Matching')
    plt.ylabel('Frequency')
    plt.show()

def test_substring_length_prob(sub_len, alph, string_len, trials):
    """ Tests theoretical probability against simulations for prob of substring.

    Args:
        sub_len: The size of the substring.
        alph: The size of the alphabet.
        string_len: The size of the string to test.
    Returns:
        Tuple of (theoretical, observed)
    """
    theoretical = get_theoretical_substring_prob(sub_len, alph, string_len)
    # Find ratio from real life sim.
    num_successes = [1 if num_overalapping_substrings(sub_len, string_len, alph)
                     else 0 for _ in xrange(trials)]
    sim_ratio = sum(num_successes) / trials
    return (theoretical, sim_ratio)

def plot_substring_prob_error(sub_lens, alphs, string_len, trials):
    """ Plots the error between theoretical prob and observed highest substring.

    Args:
        sub_lens: A list of the substring lengths to plot.
        alphs: A list of the alphabet sizes to try.
        string_len: The total length of the string.
        trials: The total number of trials to perform for each data point.
    """
    data_by_alph = [[] for _ in range(len(alphs))]
    for alph_index, alph_size in enumerate(alphs):
        for sub_len in sub_lens:
            found_values = test_substring_length_prob(sub_len, alph_size,
                                                      string_len, trials)
            data_by_alph[alph_index].append((found_values[0]
                                             - found_values[1]))
    for alph_index, alph_size in enumerate(alphs):
        plt.plot(sub_lens, data_by_alph[alph_index], label=('Alphabet '
                                                            + str(alph_size)))
    plt.legend(loc='upper right')
    plt.title('Theoretical - Observed Probability for Matching Substring'
              + '; String Length: ' + str(string_len)
              + '; Trials: ' + str(trials))
    plt.xlabel('Substring Size')
    plt.ylabel('Probability')
    plt.show()

if __name__ == '__main__':
    # test_substring_length_prob(35, 10, 50, 1000)
    plot_substring_prob_error([10, 20, 30, 40, 50], [4, 6, 8, 10], 100, 100)
