"""Perform analysis on the clash number distribution."""

from __future__ import division
from collections import deque
from matplotlib import pyplot as plt
import numpy as np

import aho_construction
import aho_test

DNA_ALPH = ['A', 'C', 'G', 'T']

def create_alphabet(size):
    """Creates arbitrary alphabet to use of specified size.

    Args:
        size: The size of the alphabet.
    Returns:
        An alphabet represented as a list of letters.
    """
    return [chr(ordinal) for ordinal in range(65, 65 + size)]

def get_clashes(root):
    """Get the clash numbers for all nodes in the DFA.

    Args:
        root: The root of the DFA.
    Returns:
        A nested list of the clash numbers e.g. return_val[lvl][elem] is an int.
    """
    # Keep a list of dictionaries where each index of the list is a level
    # and each dictionary keeps track of how many times we have seen a forward
    # link leading to some node.
    levels = []
    queue = deque()
    seen = []

    # Do a BFS to fill out data structure.
    queue.appendleft(root)
    seen.append(root.unique_id)
    while len(queue) > 0:
        curr_node = queue.pop()

        for _, child in curr_node.transitions.iteritems():
            # Check all forward transtionsd
            if child.level == curr_node.level + 1:
                if len(levels) < child.level:
                    levels.append({})
                if child.unique_id in levels[child.level - 1]:
                    levels[child.level - 1][child.unique_id] += 1
                else:
                    levels[child.level - 1][child.unique_id] = 1
            # Add to the queue if we haven't seen before
            if child.unique_id not in seen:
                queue.appendleft(child)
                seen.append(child.unique_id)

    # Collapse dictionaries to only have frequency.
    levels = [levels[lvl].values() for lvl in range(len(levels))]
    to_return = []
    for lvl in levels:
        full_clash = []
        for elem in lvl:
            full_clash += [elem] * elem
        to_return.append(full_clash)
    return to_return

def get_level_sizes(root, string_length):
    """Get the sizes of the levels.

    Args:
        root: The root to the automaton.
        string_length: The length of the string being analyzed.
    Returns:
        List of the sizes of the levels starting from level 1 rather than 0.
    """
    queue = deque()
    seen = {}
    lvl_sizes = [0 for _ in range(string_length)]

    queue.appendleft(root)
    seen[root.unique_id] = True
    while len(queue) > 0:
        curr_node = queue.pop()
        # Add children and add to lvl_sizes
        for _, child in curr_node.transitions.iteritems():
            if child.unique_id not in seen:
                lvl_sizes[child.level - 1] += 1
                seen[child.unique_id] = True
                queue.appendleft(child)
    return lvl_sizes

def get_expected_vals(trials, string_length, alphabet):
    """Find the expected value of 1/clash and number of nodes for each level.
    (E[1/T_1,n], E[W_n])

    Args:
        trials: The number of trials to perform.
        string_length: The length of the strings to look at.
        alphabet: The alphabet to use for string construction.
    Returns:
        A list of tulples (expected 1/clash, expected nodes) for each level.
    """
    running_sum = [0 for _ in range(string_length)]
    num_expanded = [0 for _ in range(string_length)]
    # Sum up all of the 1/clash values
    for _ in xrange(trials):
        rand_string = aho_test.build_random_string(string_length, alphabet)
        root = aho_construction.construct_dfa(rand_string, alphabet)
        clashes = get_clashes(root)
        for lvl, vals in enumerate(clashes):
            for val in vals:
                running_sum[lvl] += 1 / val
            num_expanded[lvl] += len(vals)
    # Find the sample means
    to_return = []
    for lvl, accum in enumerate(running_sum):
        to_return.append((accum / num_expanded[lvl], accum / trials))
    return to_return

def sim_next_size(expected_vals, alphabet):
    """Calculate the expected next level size from previous information.

    Args:
        expected_vals: Expected data as a list of tuples
            (expected 1/clash, expected nodes) for each level.
    Returns:
        A list of the calculated level size based on the above information.
    """
    # 0 for first index because we have no informatino for the first level
    calcd = [0]
    growth_coef = len(alphabet) / (2 * (1 - (1 / 2) ** len(alphabet)))
    for lvl in range(len(expected_vals) - 1):
        calcd.append(expected_vals[lvl + 1][0] * growth_coef
            * expected_vals[lvl][1])
    return calcd

def plot_comparison(true_vals, predicted_vals, string_len, trials):
    """Plot the two expected values side-by-side.

    Args:
        true_vals: The sample mean of number of nodes in level n.
        predicted_vals: The predicted number of nodes in level n.
        string_len: The length of the string that was looked at.
        trials: The number of trials performed.
    """
    # Plot the two together on the same plot.
    true, = plt.plot(range(string_len), true_vals, 'bo',
            label='True Values')
    pred, = plt.plot(range(string_len), predicted_vals, 'ro',
            label='Predicted Values')
    plt.xlabel('Level')
    plt.ylabel('Number of nodes in Level')
    plt.title('Comparison of True and Predicted Number of Nodes in Level, '
            '%d trials, %d length string' % (trials, string_len))
    plt.grid(True)
    plt.legend([true, pred])
    plt.show()

def run_analysis(trials, string_length, alphabet, make_hists=False):
    """Run analysis between truth and predicted for nodes in level.

    Args:
        trials: Number of trials to perform.
        string_len: The length of the string to consider.
        alphabet: The alphabet to use for string construction.
        make_hists: Whether to plot histograms.
    Returns:
        List of differences.
    """
    data = get_expected_vals(trials, string_length, alphabet)
    predicted = sim_next_size(data, alphabet)
    level_sizes = [lvl[1] for lvl in data]
    if make_hists:
        plot_comparison(level_sizes, predicted, string_length, trials)
    differences = []
    for lvl in range(len(level_sizes)):
        differences.append(level_sizes[lvl] - predicted[lvl])
    return differences

def plot_clash_hist(trials, string_length, alphabet):
    """Plot histograms of clash numbers for each level.

    Args:
        trials: Number of trials to perform.
        string_length: The length of the string to consider.
        alphabet: The alphabet to use in string construction.
    """
    # Gather the data.
    data = [[] for _ in range(string_length)]
    for _ in xrange(trials):
        rand_string = aho_test.build_random_string(string_length, alphabet)
        root = aho_construction.construct_dfa(rand_string, alphabet)
        clashes = get_clashes(root)
        for lvl, clash_data in enumerate(clashes):
            data[lvl].append(clash_data[0])

    # Histogram the data.
    for lvl, lvl_data in enumerate(data):
        aho_test.analyze_data(lvl_data, make_histogram=True,
            title=' '.join(['Clashes for Level', str(lvl + 1), 'Samples:',
            str(len(lvl_data))]))

def plot_lvlsize_hist(trials, string_length, alphabet):
    """Plots the histograms of level sizes.

    Args:
        trials: Number of trials to perform.
        string_length: The length of the string to consider.
        alphabet: The alphabet to use in string construction.
    """
    # Gather the data.
    data = [[] for _ in range(string_length)]
    for _ in xrange(trials):
        rand_string = aho_test.build_random_string(string_length, alphabet)
        root = aho_construction.construct_dfa(rand_string, alphabet)
        lvl_sizes = get_level_sizes(root, string_length)
        for lvl, lvl_size in enumerate(lvl_sizes):
            data[lvl].append(lvl_size)

    # Histrogram the data.
    for lvl, lvl_size in enumerate(data):
        aho_test.analyze_data(lvl_size, make_histogram=True,
            title=' '.join(['Size of Level', str(lvl + 1), 'Samples:',
            str(len(lvl_size))]))

def plot_lvlsize_trend(trials, string_length, alphabet):
    """Plots the trend of average level size with variance included.

    Args:
        trials: Number of trials to be performed.
        string_length: The length of the string to consider.
        alphabet: The alphabet to use in string construction.
    """
    # Gather the data.
    data = [[] for _ in range(string_length)]
    for _ in xrange(trials):
        rand_string = aho_test.build_random_string(string_length, alphabet)
        root = aho_construction.construct_dfa(rand_string, alphabet)
        lvl_sizes = get_level_sizes(root, string_length)
        for lvl, lvl_size in enumerate(lvl_sizes):
            data[lvl].append(lvl_size)
    # Get the averages and variances from the data.
    averages = [np.average(lvl_data) for lvl_data in data]
    stddev = [np.std(lvl_data) for lvl_data in data]

    # Plot the data as a trent.
    plt.errorbar(range(1, string_length + 1), averages, yerr=stddev)
    plt.title('Average Level Size vs Level; Trials: ' + str(trials)
              + '; Alphabet: ' + str(len(alphabet)))
    plt.xlabel('Level')
    plt.ylabel('Level Size')
    plt.yticks(np.arange(-5, 20, 1.0))
    plt.ylim((-5, 20))
    plt.grid(True)
    plt.show()


def get_level_size_ratio(trials, string_length, alphabet):
    """Gets the ratio between the size of levels.

    Args:
        trials: Number of trials to perform.
        string_length: The length of the string to consider.
        alphabet: The alphabet to use for string construction.
    Returns:
        A list of the average ratios between the levels.
    """
    ratios = [0 for _ in range(string_length - 1)]
    for _ in xrange(trials):
        # Get the clash numbers.
        rand_string = aho_test.build_random_string(string_length, alphabet)
        root = aho_construction.construct_dfa(rand_string, alphabet)
        level_sizes = get_level_sizes(root, string_length)
        # Add the ratio of the level sizes.
        for lvl in range(string_length - 1):
            ratios[lvl] += level_sizes[lvl + 1] / level_sizes[lvl]
    # Get average of the ratios
    for lvl in range(string_length - 1):
        ratios[lvl] /= trials
    return ratios

def get_total_size_ratio(trials, string_length, alphabet):
    """Gets the ratio between the size of levels. E[X_n+1/X_n]

    Args:
        trials: Number of trials to perform.
        string_length: The length of the string to consider.
        alphabet: The alphabet to use for string construction.
    Returns:
        A list of the average ratios for total size between the levels.
    """
    ratios = [0 for _ in range(string_length - 1)]
    for _ in xrange(trials):
        # Get the clash numbers.
        rand_string = aho_test.build_random_string(string_length, alphabet)
        root = aho_construction.construct_dfa(rand_string, alphabet)
        clashes = get_clashes(root)
        # Get the total sizes
        total_sizes = [0 for _ in range(string_length)]
        for lvl, lvl_clashes in enumerate(clashes):
            for clash in lvl_clashes:
                total_sizes[lvl] += 1 / clash
        for lvl in range(len(total_sizes))[::-1]:
            for prev_lvl in range(lvl):
                total_sizes[lvl] += total_sizes[prev_lvl]
        # Add the ratio of the total sizes.
        for lvl in range(string_length - 1):
            ratios[lvl] += total_sizes[lvl + 1] / total_sizes[lvl]
    # Get average of the ratios
    for lvl in range(string_length - 1):
        ratios[lvl] /= trials
    return ratios

def get_average_dfa_size(trials, string_length, alphabet):
    """Gets the average size of automaton E[X_n]

    Args:
        trials: Number of trials to perform.
        string_length: The length of the string to consider.
        alphabet: The alphabet to use for string construction.
    Returns:
        A list of the dfa sizes by increasing level.
    """
    sizes = [0 for _ in range(string_length)]
    for _ in xrange(trials):
        # Get a random string and get the size of each level.
        rand_string = aho_test.build_random_string(string_length, alphabet)
        root = aho_construction.construct_dfa(rand_string, alphabet)
        lvl_sizes = get_level_sizes(root, string_length)
        # Add the sizes to our list
        curr_size = 1
        for lvl, lvl_size in enumerate(lvl_sizes):
            curr_size += lvl_size
            sizes[lvl] = curr_size
    # Get the sample mean for level size.
    for lvl in range(string_length):
        sizes[lvl] /= trials

    return sizes

def get_dfa_growth_of_expected(trials, string_length, alphabet):
    """Gets the growth of the expected size E[X_n+1]/E[X_n]

    Args:
        trials: Number of trials to perform.
        string_length: The length of the string to consider.
        alphabet: The alphabet to use for string construction.
    Returns:
        A list of the growths by increasing level.
    """
    sizes = get_average_dfa_size(trials, string_length, alphabet)
    # Compute the ratios.
    ratios = []
    for lvl in range(len(sizes) - 1):
        ratios.append(sizes[lvl + 1] / sizes[lvl])
    return ratios

def automaton_growth_ratio(string_length, alphabet):
    """Find the growth ratios for the size of the automaton. X_n+1/X_n

    Args:
        string_length: The length of the string to analyze.
        alphabet: Size of the alphabet to use for construction.
    Returns:
        List of ratios where each is the new size over the old size.
    """
    # Get a random string of the size and get the clash numbers.
    rand_string = aho_test.build_random_string(string_length, alphabet)
    root = aho_construction.construct_dfa(rand_string, alphabet)
    lvl_sizes = get_level_sizes(root, string_length)

    curr_size = 0
    ratios = []
    for last_lvl in range(len(lvl_sizes) - 1):
        curr_size += lvl_sizes[last_lvl]
        ratios.append((curr_size + lvl_sizes[last_lvl + 1]) / curr_size)
    return ratios

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
        lvl_sizes = get_level_sizes(root, clash_level)
        # Check if we need anymore samples
        if lvl_sizes[-2] not in seen_amount:
            seen_amount[lvl_sizes[-2]] = 1
            clash_data[lvl_sizes[-2]] = []
        elif seen_amount[lvl_sizes[-2]] < samples:
            seen_amount[lvl_sizes[-2]] += 1
            # Compute the clashes and add just the first clash number.
            clashes = get_clashes(root)
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

def compare_expected_growths(trials, string_length, alphabet_range):
    """Compare the simulated growth rate E[X_n+1]/E[X_n] w/ expected.

    Args:
        string_length: The maximum length to check.
        trials: The number of trials to perform.
        alphabet_range: The range of the alphabet lengths to use as a list.
    Returns:
        List of tuples (simulated, predicted, (predicted - simulated)/simulated)
    """
    results = []
    for alph in [create_alphabet(alph_size) for alph_size in alphabet_range]:
        # Get what we need for prediction.
        expected_split = len(alph) / (2 * (1 - (1 / 2) ** len(alph)))
        asym_inv_clash = get_expected_vals(trials, string_length, alph)[-1][0]
        predicted = expected_split * asym_inv_clash
        # Get the simulated value.
        sim = get_dfa_growth_of_expected(trials, string_length, alph)[-1]
        # Add information to the results.
        results.append((sim, predicted, (predicted - sim) / sim))
    return results

if __name__ == '__main__':
    plot_lvlsize_trend(10000, 100, create_alphabet(2))
