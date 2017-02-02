"""Analyze the nature of the fallbacks for the automata."""

from __future__ import division
from collections import deque
from matplotlib import pyplot as plt
from random import randint
from numpy.random import geometric

import aho_construction
import aho_test

DNA_ALPH = ['A', 'C', 'G', 'T']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y']

def get_fall_levels(root):
    """Gets the levels that the fallbacks lead to for each level.

    Args:
        root: The root for the automaton to analyze.
    Returns: A nested list formatted as...
        ret[lvl][lvl_node] = <level fallen back to>
    """
    to_return = []

    # Do BFS to analyze each node.
    seen = [root.unique_id]
    queue = deque()
    queue.appendleft(root)
    while len(queue) > 0:
        curr_node = queue.pop()
        # Add queue location of where this node falls to our list.
        if len(to_return) <= curr_node.level:
            to_return.append([])
        to_return[curr_node.level].append(curr_node.fallback.level)
        # Add the children to the queue to explore.
        for child in curr_node.transitions.values():
            if child.unique_id not in seen:
                queue.appendleft(child)
                seen.append(child.unique_id)
    return to_return

def compute_num_fallbacks(root, level_cap):
    """For each level, finds the number of failures pointing to that level.

    Args:
        root: Root for the constructed DFA object in question.
        level_cap: The maximum level to consider finding failures to.
    Returns:
        List of number of failures going to each level.
        e.g. result[lvl] = <num failures pointing to lvl>
    """
    to_return = [0 for _ in range(level_cap + 1)]

    # Do BFS to analyze each node.
    seen = [root.unique_id]
    queue = deque()
    queue.appendleft(root)
    while len(queue) > 0:
        curr_node = queue.pop()
        # Check if the fallback goes to a level we are considering.
        fall_lvl = curr_node.fallback.level
        if fall_lvl <= level_cap:
            to_return[fall_lvl] += 1
        # Add the children to the queue to explore.
        for child in curr_node.transitions.values():
            if child.unique_id not in seen:
                queue.appendleft(child)
                seen.append(child.unique_id)
    return to_return

def compute_fall_percentages(trials, str_length, alphabet, level_cap,
        lower_than=False):
    """Finds the percentages of fallbacks that go to a certain level.

    Args:
        trials: The number of trials to perform.
        str_length: The length of strings to consider.
        alphabet: The alphabet to use for string construction.
        level_cap: The highest level to consider when looking at where fallbacks
            lead to.
        lower_than: Whether to consider fallbacks less than or equal to a
            some level in question. e.g. with this mode enabled if we see nodes
            going back to levels 0, 1, 2, and 3, for considering level 2 we
            would compute 75 percent rather than 25 percent.
    Returns: A list nested list of the percentages given in the format...
        ret[lvl][fallback_to_lvl_i] = <some_percentage>
        The last index of the inner list is the percent of nodes that fell
        outside of the level cap (i.e. ret[lvl][-1]).
    """
    fall_counts = [[0 for _ in range(level_cap + 2)]
                   for _ in range(str_length + 1)]

    for _ in xrange(trials):
        # Build a random string and construct the dfa
        rand_string = aho_test.build_random_string(str_length, alphabet)
        root = aho_construction.construct_dfa(rand_string, alphabet)
        # Find where the falls lead to and add to our fall_counts
        falls = get_fall_levels(root)
        for curr_lvl, fall_list in enumerate(falls):
            fall_lvl = fall_list[randint(0, len(fall_list) - 1)]
            if fall_lvl <= level_cap:
                fall_counts[curr_lvl][fall_lvl] += 1
            else:
                fall_counts[curr_lvl][-1] += 1

    # Divide by totals to convert counts to percentages
    for curr_lvl in fall_counts:
        curr_total = sum(curr_lvl)
        for fall_lvl in range(level_cap + 2):
            curr_lvl[fall_lvl] /= curr_total
            if lower_than and fall_lvl > 0:
                curr_lvl[fall_lvl] += curr_lvl[fall_lvl - 1]
    return fall_counts

def plot_fall_percentages(trials, str_length, alphabet, level_cap,
        lower_than=False):
    """Plots the percentages of fallbacks that go to a certain level.

    Args:
        trials: The number of trials to perform.
        str_length: The length of strings to consider.
        alphabet: The alphabet to use for string construction.
        level_cap: The highest level to consider when looking at where fallbacks
            lead to.
        lower_than: Whether to consider fallbacks less than or equal to a
            some level in question. e.g. with this mode enabled if we see nodes
            going back to levels 0, 1, 2, and 3, for considering level 2 we
            would compute 75 percent rather than 25 percent.
    """
    percents = compute_fall_percentages(trials, str_length, alphabet, level_cap,
                                        lower_than)
    lvls = range(str_length + 1)
    legends = []
    for fall_lvl in range(level_cap + 2):
        fall_percents = [percents[lvl][fall_lvl] for lvl in lvls]
        if fall_lvl <= level_cap:
            fall_label = str(fall_lvl)
            if lower_than:
                fall_label += ' or Below'
        else:
            fall_label = 'Other'
        line_label, = plt.plot(lvls, fall_percents,
                               COLORS[fall_lvl % len(COLORS)], lw=2,
                               label='Fall to Level ' + str(fall_label))
        if lower_than:
            plt.fill_between(lvls, 0, fall_percents, alpha=0.25,
                             facecolor=COLORS[fall_lvl % len(COLORS)])
        legends.append(line_label)
    plt.ylabel('Percent of Fallen')
    plt.ylim((0, 1))
    plt.xlabel('Level')
    plt.title('Breakdown of Fall Destinations vs Level; Trials: ' + str(trials)
              + '; Alphabet Size: ' + str(len(alphabet)))
    plt.legend()
    plt.show()

def plot_falldest_dist(trials, alphabet, level, level_cap):
    """Plot the percent of falls observed vs level fallback leads to.

    Args:
        trials: The number of strings to build and analyze.
        alphabet: The alphabet to use to build or random strings.
        level: The level at which we analyze the fallbacks.
        level_cap: The highest level we want to observe fallbacks to.
    """
    # Get the data for the last level's observed fallback destenations.
    percents = compute_fall_percentages(trials, level, alphabet, level_cap)
    dest_dist = percents[-1][:-1]
    # Geometric data clean up later.
    geom = geometric(p=0.18, size=trials)
    compare_data = []
    for lvl in range(level_cap + 1):
        compare_data.append((geom == lvl + 1).sum() / trials)
    # Plot the data.
    sim, = plt.plot(range(len(dest_dist)), dest_dist, 'bo', label='Simulated')
    geo, = plt.plot(range(len(dest_dist)), compare_data, 'go', label='Geometric p=0.18')
    plt.ylim((0, 1))
    plt.xlabel('Destination Level')
    plt.ylabel('Percent of Fallen')
    plt.title('Percent of Fallen vs Destination Level; Trials: ' + str(trials)
              + '; Observed at level: ' + str(level)
              + '; Alphabet: ' + str(len(alphabet)))
    plt.legend()
    plt.show()

def plot_numfall_single(alphabet, str_length, level_cap):
    """Plot the number of failures pointing to a level vs level for one string.

    Args:
        alphabet: The alphabet to use to construct the string.
        str_length: The length of the string.
        level_cap: The maximum level to consider on the plot.
    """
    # Build a random string, construct the dfa, and get the data.
    rand_string = aho_test.build_random_string(str_length, alphabet)
    root = aho_construction.construct_dfa(rand_string, alphabet)
    data = compute_num_fallbacks(root, level_cap)

    # Plot the data.
    plt.plot(range(level_cap + 1), data)
    plt.xlabel('Level')
    plt.ylabel('Number of Failures Pointing to Level')
    plt.title('Number of Failures Pointing to Level vs. Level'
              + '; String Length: ' + str(str_length)
              + '; Alphabet Size: ' + str(len(alphabet)))
    plt.show()

def plot_numfall_avg(trials, alphabet, str_length, level_cap):
    """Plot the average number of failures pointing to a level vs level.

    Args:
        trials: The number of strings to consider for the average value.
        alphabet: The alphabet to use to construct the string.
        str_length: The length of the string.
        level_cap: The maximum level to consider on the plot.
    """
    averages = [0 for _ in range(level_cap + 1)]

    # Construct random string and add the data to averages.
    for _ in xrange(trials):
        rand_string = aho_test.build_random_string(str_length, alphabet)
        root = aho_construction.construct_dfa(rand_string, alphabet)
        data = compute_num_fallbacks(root, level_cap)
        for lvl, num_fall in enumerate(data):
            averages[lvl] += num_fall

    averages = [avg / trials for avg in averages]
    # Plot the averages.
    plt.plot(range(level_cap + 1), averages)
    plt.xlabel('Level')
    plt.ylabel('Number of Failures Pointing to Level')
    plt.title('Number of Failures Pointing to Level vs. Level'
              + '; Trials: ' + str(trials)
              + '; String Length: ' + str(str_length)
              + '; Alphabet Size: ' + str(len(alphabet)))
    plt.show()


if __name__ == '__main__':
    plot_numfall_avg(1000, [chr(ord('A') + i) for i in range(10)], 100, 100)
