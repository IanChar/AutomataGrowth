"""Code to test aho_construction against subset construction."""
from __future__ import division
import matplotlib.pyplot as plt
from collections import deque
from random import random
import sys

sys.path.append('./..')
sys.path.append('./../marschall')
sys.path.append('../common')
import aho_construction
import time_compare

ALPHABET = ['A', 'C', 'G', 'T']

def deduce_path_lengths(string, alphabet, return_lvl_size=False,
        as_ratios=False):
    """Finds the number of nodes used in fallback paths

    Args:
        string: The generalized string to compare.
        alphabet: The alphabet of the string.
        return_lvl_size: Whether to return the size of each level as well.
        as_ratios: Whether to return infromation as ratio of nodes used in path
                over the number of total nodes up to that point.
    Returns:
        List of the number of nodes used in fallback (or ratios if flag given).
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
    if as_ratios:
        for lvl in range(len(num_failure_path)):
            total_nodes = 0
            for ancestor_lvl in range(lvl + 1):
                total_nodes += len(level_ptrs[ancestor_lvl])
            if total_nodes > 0:
                num_failure_path[lvl] /= total_nodes
    if return_lvl_size:
        return num_failure_path, [len(level_ptrs[i])
                                  for i in range(len(string) + 1)]
    return num_failure_path

def markov_path_length(level, expanded_nodes, merge_prob=0.75):
    """Uses a Markov model to simulate the nodes used in failure paths.

    Args:
        level: The current level for which we want to simulate the paths length.
        expanded_nodes: How many nodes were expanded at this level.
    Returns:
        Int representing the number of nodes involved with the failure path.
    """
    total_nodes = 0
    for _ in range(expanded_nodes):
        # Run a simulation through the Markov chain for each expanded node.
        curr_lvl = level
        steps = 0
        # Count number of steps until we hit lvl 0 or merged state.
        # Here merged state is represnted by -1.
        while curr_lvl > 0:
            # Subtract either 1,...,curr_lvl + 1 w/ equal prob.
            rand_num = random()
            if rand_num <= merge_prob:
                curr_lvl = -1
            else:
                curr_lvl -= int(rand_num * curr_lvl) + 1
            steps += 1
        total_nodes += steps
    return total_nodes

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
    """Plots the ratios of growth for nodes used in paths.

    Args:
        data: The ratios to plot.
        string_len: The total length of the string.
        title: Title for the plot.
    """
    plt.plot(range(string_len), data, 'bo')
    plt.xlabel('Growth at Level')
    plt.ylabel('Growth')
    plt.title(title)
    plt.grid(True)
    plt.show()

def eval_markov(string_len, trials, tolerance):
    """Compare the markov model to the actual simulation.

    Args:
        string_len: The length of the strings to test on.
        trials: Amount of trials to be performed.
    """
    avg_path = [0 for _ in range(string_len + 1)]
    avg_sim_path = [0 for _ in range(string_len + 1)]
    for _ in range(trials):
        # Simulate a random dfa.
        rand_string = time_compare.build_random_string(string_len)
        paths, level_sizes = deduce_path_lengths(rand_string, ALPHABET,
                return_lvl_size=True)
        sim_path = []
        for lvl, lvl_size in enumerate(level_sizes):
            sim_path.append(markov_path_length(lvl, lvl_size))
        for lvl in range(string_len + 1):
            avg_path[lvl] += paths[lvl]
            avg_sim_path[lvl] += sim_path[lvl]
    for lvl in range(string_len + 1):
        avg_path[lvl] /= float(trials)
        avg_sim_path[lvl] /= float(trials)
    return avg_path, avg_sim_path

def plot_comparison(avg_path, avg_sim_path, string_len, trials):
    """Plot the two path lengths side-by-side.

    Args:
        avg_path: The average path found from the Markov model.
        avg_sim_path: The average path length found from algorithm simulations.
        string_len: The length of the string that was looked at.
        trials: The number of trials performed.
    """
    # Plot the two together on the same plot.
    sim, = plt.plot(range(string_len + 1), avg_path, 'bo', label='Simulation')
    markov, = plt.plot(range(string_len + 1), avg_sim_path, 'ro',
            label='Markov Model')
    plt.xlabel('Level')
    plt.ylabel('Number of Nodes in Paths')
    plt.title('Comparison of Simulation and Markov Model, '
            '%d trials, %d length string' % (trials, string_len))
    plt.grid(True)
    plt.legend([sim, markov])
    plt.show()

"""Main functions"""

def analyze_path_model(string_length, trials, merge_prob):
    """Generate data from algorithm and model and compare.

    Args:
        string_length: Name of the string to look at.
        trials: Number of trials to perform.
        merge_prob: Probability that the failure will be merged in the chain.
    """
    avg_path, avg_sim_path = eval_markov(string_length, trials, merge_prob)
    plot_comparison(avg_path, avg_sim_path, string_length, trials)

def get_ratio_trends(string_length, trials):
    """Get the trend of the ratio of amount of path used to total nodes.

    Args:
        string_length: Length of the string to analyze.
        trials: The number of trials to perform.
    """
    results = [0 for _ in range(string_length + 1)]
    for _ in xrange(trials):
        rand_string = time_compare.build_random_string(string_length)
        ratios = deduce_path_lengths(rand_string, ALPHABET, as_ratios=True)
        for lvl, ratio in enumerate(ratios):
            results[lvl] += ratio
    return [total / trials for total in results]


if __name__ == '__main__':
    print get_ratio_trends(30, 100)
