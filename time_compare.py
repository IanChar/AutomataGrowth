"""Tool to generate graphs showing comparison of dfa construction algorithms."""

import time
import random
import matplotlib.pyplot as plt

import single_dfa_construction

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

def find_average_time(string_length, trials, technique):
    """Find the average time for calculating a DFA of a certain length.

    Args:
        string_length: The length of the strings that should be considered.
        trials: The number of trials to average over.
        technique: The function of the technique to use.
    Returns: The average time it took to compute the DFAs.
    """
    total_time = 0
    for _ in range(trials):
        # Build the random string.
        random_string = []
        for _ in range(string_length):
            rand_code = random.randint(0, 14) + 1
            rand_letters = []
            for digit in range(3):
                if rand_code & (1 << digit) > 0:
                    rand_letters.append(single_dfa_construction
                                        .ALPHABET[digit])
            random_string.append(rand_letters)
        # Time for the random string.
        total_time += time_method(random_string, technique)
    return total_time / trials

def plot_comparison(trials, max_string_length):
    """Compute averages the techniques and display results on same plot.

    Args:
        trials: The number of trials that should be performed for each of the
            string lengths.
        max_string_length: The maximum length of the string to test for.
    """
    # Calculate the times.
    word_lengths = range(3, max_string_length + 1)
    sub_constructs = [find_average_time(x, trials,
                      single_dfa_construction.subset_construction)
                      for x in word_lengths]
    intersections = [find_average_time(x, trials,
                     single_dfa_construction.intersection_construction)
                     for x in word_lengths]
    # Plot the times.
    plt.plot(word_lengths, sub_constructs, 'ro', label='Subset Construction')
    plt.plot(word_lengths, intersections, 'bo',
             label='Intersection Construction')
    plt.xlabel('Word Length')
    plt.xlim([2.5, max_string_length + 0.5])
    plt.ylabel('Average Time (s)')
    plt.title('Average Time for Construction vs Word Length'
              ' (%d trials per word)' % trials)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.show()



if __name__ == '__main__':
    plot_comparison(1000, 16)
