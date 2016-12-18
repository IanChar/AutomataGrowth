"""Analyze the nature of the fallbacks for the automata."""

from __future__ import division
from collections import deque
from matplotlib import pyplot as plt

import aho_construction
import aho_test

DNA_ALPH = ['A', 'C', 'G', 'T']

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
            for fall_lvl in fall_list:
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

if __name__ == '__main__':
    print compute_fall_percentages(10, 5, DNA_ALPH, 2)
