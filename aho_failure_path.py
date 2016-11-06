"""Code to test aho_construction against subset construction."""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from automata import Automata
import aho_construction
import time_compare

ALPHABET = ['A', 'C', 'G', 'T']

def deduce_path_lengths(string, alphabet):
    """Finds the number of nodes used in fallback paths

    Args:
        string: The generalized string to compare.
        alphabet: The alphabet of the string.
    Returns:
        Integer of how many failure computations were made.
    """
    root = aho_construction.construct_dfa(string, alphabet)
    # Structure sorting the states to the level number.
    level_ptrs = {l:[] for l in range(len(string) + 1)}
    # Data structure representing the number of states by level.
    num_failure_path = [0 for _ in range(len(string) + 1)]
    # The names of the states we have seen so far.
    seen = [0]

    # Do a BFS of the graph that we have made.
    queue = deque()
    queue.appendleft(root)
    while len(queue) > 0:
        curr_node = queue.pop()
        # Add children to the queue.
        for linked_node in curr_node.transitions.values():
            if linked_node.unique_id not in seen:
                queue.appendleft(linked_node)
                seen.append(linked_node.unique_id)
                level_ptrs[linked_node.level].append(linked_node)

    # Fill out the number of failure paths used.
    for lvl in range(len(string) + 1):
        seen = []
        nxt_node = None
        for child in level_ptrs[lvl]:
            nxt_node = child.fallback
            while nxt_node.unique_id not in seen:
                seen.append(nxt_node.unique_id)
                nxt_node = nxt_node.fallback
        num_failure_path[lvl] = len(seen)
    return num_failure_path

def sim_path_growth(trials, str_length):
    """Simulates and finds the growth rate of path size.

    Args:
        trials: NUmber of trials to be performed.
        str_length: The length of strings to be simulated.
    Returns:
        List of growths averaged together.
    """
    growth_ratios = [0 for _ in range(str_length)]
    for _ in range(trials):
        rand_string = time_compare.build_random_string(str_length)
        paths = deduce_path_lengths(rand_string, ALPHABET)
        for lvl in range(str_length):
            if paths[lvl] != 0:
                growth_ratios[lvl] += (paths[lvl + 1]
                        / float(paths[lvl]))
    for lvl in range(str_length):
        growth_ratios[lvl] /= float(trials)
    return growth_ratios

def plot_growth_rate(data, string_len, title="Average Path Growth"):
    plt.plot(range(string_len), data, 'bo')
    plt.xlabel('Growth at Level')
    plt.ylabel('Growth')
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_growth_rate(sim_path_growth(10000, 20), 20,
            'Average Path Growth: String Length 20, Trials 10000')
