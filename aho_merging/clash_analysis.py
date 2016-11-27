"""Perform analysis on the clash number distribution."""

from __future__ import division
from collections import deque
from matplotlib import pyplot as plt
import sys

sys.path.append('./..')
sys.path.append('./../automata')
import aho_construction
import time_compare
import aho_test

ALPHABET = ['A', 'C', 'G', 'T']

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

def get_expected_vals(trials, string_length):
    """Find the expected value of 1/clash and number of nodes for each level.

    Args:
        trials: The number of trials to perform.
        string_length: The length of the strings to look at.
    Returns:
        A list of tulples (expected 1/clash, expected nodes) for each level.
    """
    running_sum = [0 for _ in range(string_length)]
    num_expanded = [0 for _ in range(string_length)]
    # Sum up all of the 1/clash values
    for _ in xrange(trials):
        rand_string = time_compare.build_random_string(string_length)
        root = aho_construction.construct_dfa(rand_string, ALPHABET)
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

def sim_next_size(expected_vals):
    """Calculate the expected next level size from previous information.

    Args:
        expected_vals: Expected data as a list of tuples
            (expected 1/clash, expected nodes) for each level.
    Returns:
        A list of the calculated level size based on the above information.
    """
    # 0 for first index because we have no informatino for the first level
    calcd = [0]
    growth_coef = len(ALPHABET) / (2 * (1 - (1 / 2) ** len(ALPHABET)))
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

def run_analysis(trials, string_length):
    """Run analysis between truth and predicted for nodes in level.

    Args:
        trials: Number of trials to perform.
        string_len: The length of the string to consider.
    """
    data = get_expected_vals(trials, string_length)
    predicted = sim_next_size(data)
    level_sizes = [lvl[1] for lvl in data]
    plot_comparison(level_sizes, predicted, string_length, trials)

def plot_clash_hist(trials, string_length):
    """Plot histograms of clash numbers for each level.

    Args:
        trials: Number of trials to perform.
        string_length: The length of the string to consider.
    """
    # Gather the data.
    data = [[] for _ in range(string_length)]
    for _ in xrange(trials):
        rand_string = time_compare.build_random_string(string_length)
        root = aho_construction.construct_dfa(rand_string, ALPHABET)
        clashes = get_clashes(root)
        for lvl, clash_data in enumerate(clashes):
            data[lvl] += clash_data

    # Histogram the data.
    for lvl, lvl_data in enumerate(data):
        aho_test.analyze_data(lvl_data, make_histogram=True,
            title=' '.join(['Clashes for Level', str(lvl + 1), 'Samples:',
            str(len(lvl_data))]))


if __name__ == '__main__':
    plot_clash_hist(10000, 10)
