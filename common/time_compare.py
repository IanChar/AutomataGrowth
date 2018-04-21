"""Tool to generate graphs showing comparison of dfa construction algorithms."""

import time
import random
import matplotlib.pyplot as plt
import sys

sys.path.append('../marschall')
sys.path.append('../experimental')
import single_dfa_construction
import aho_construction

def time_method(string, technique):
    """Times how long the technique takes to make a corresponding DFA.

    Args:
        string: The generalized string to pass to the technique.
        technique: The function of the technique to use.
    Returns: The time to compute the DFA.
    """
    start_time = time.time()
    technique(string)
    return time.time() - start_time

def find_average_time(strings, technique):
    """Find the average time for calculating a DFA of a certain length.

    Args:
        strings: Nested list of random strings.
        technique: The function of the technique to use.
    Returns: The average time it took to compute the DFAs.
    """
    total_time = 0
    for string in strings:
        # Time for the random string.
        total_time += time_method(string, technique)
    return total_time / len(strings)

"""
Requires refactoring.
def plot_comparison(trials, max_string_length):
    Compute averages the techniques and display results on same plot.

    Args:
        trials: The number of trials that should be performed for each of the
            string lengths.
        max_string_length: The maximum length of the string to test for.

    # Calculate the times.
    word_lengths = range(3, max_string_length + 1)
    sub_constructs = [find_average_time(x, trials,
                      single_dfa_construction.subset_construction)
                      for x in word_lengths]
    sub_binary = [find_average_time(x, trials,
                  single_dfa_construction.binary_subset_construction)
                  for x in word_lengths]
    intersections = [find_average_time(x, trials,
                     single_dfa_construction.intersection_construction)
                     for x in word_lengths]
    # Plot the times.
    plt.plot(word_lengths, sub_constructs, 'ro', label='Subset Construction')
    plt.plot(word_lengths, sub_binary, 'go', label='Binary Subset '
             'Construction')
    plt.plot(word_lengths, intersections, 'bo',
             label='Intersection Construction')
    plt.xlabel('Word Length')
    plt.xlim([2.5, max_string_length + 0.5])
    plt.ylabel('Average Time (s)')
    plt.title('Average Time for Construction vs Word Length'
              ' (%d trials per word)' % trials)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.show()
"""

def plot_aho_comparison(trials, max_string_length):
    """Compute averages the techniques and display results on same plot.

    Args:
        trials: The number of trials that should be performed for each of the
            string lengths.
        max_string_length: The maximum length of the string to test for.
    """
    # Calculate the times.
    word_lengths = range(3, max_string_length + 1)
    sub_binary = []
    ahos = []
    def aho_wrapper(string):
        """Wraps the aho_construction function call."""
        return aho_construction.construct_dfa(string,
                                              single_dfa_construction.ALPHABET)
    for length in word_lengths:
        rand_strings = [build_random_string(length) for _ in range(trials)]
        sub_binary.append(find_average_time(rand_strings,
                          single_dfa_construction.binary_subset_construction))
        ahos.append(find_average_time(rand_strings, aho_wrapper))

    # Plot the times.
    plt.plot(word_lengths, ahos, 'ro', label='Aho Construction')
    plt.plot(word_lengths, sub_binary, 'go', label='Binary Subset '
             'Construction')
    plt.xlabel('Word Length')
    plt.xlim([2.5, max_string_length + 0.5])
    plt.ylabel('Average Time (s)')
    plt.title('Average Time for Construction vs Word Length'
              ' (%d trials per word)' % trials)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.show()

def test_accuracy(trials, string_length):
    """Compares the two constructions to make sure they are the same.

    Args:
        trials: The amount of tests to perform.
        string_lenth: The length of the random strings to construct.
    """
    failures = 0
    for _ in range(trials):
        test_string = build_random_string(string_length)
        sub = single_dfa_construction.subset_construction(list(test_string))
        inter = single_dfa_construction.intersection_construction(test_string)
        if sub != inter:
            print '--------------------FAILED-----------------------'
            print 'String:', test_string
            print '\nSubset Construction:'
            print sub
            print '\nIntersection Construction:'
            print inter
            failures += 1
    print '-----------------------------------------------------------'
    print ('Tests complete: %d Successes, %d Failures'
           % (trials - failures, failures))


def build_random_string(length):
    """Build a random string to use.

    Args:
        length: The length of the string to construct.
    Returns:
        The generalized random string.
    """
    random_string = []
    for _ in range(length):
        rand_code = random.randint(0,
                                   2 ** len(single_dfa_construction.ALPHABET)
                                   - 2) + 1
        rand_letters = []
        for digit in range(len(single_dfa_construction.ALPHABET)):
            if rand_code & (1 << digit) > 0:
                rand_letters.append(single_dfa_construction
                                    .ALPHABET[digit])
        random_string.append(rand_letters)
    return random_string

if __name__ == '__main__':
    plot_aho_comparison(1000, 10)
