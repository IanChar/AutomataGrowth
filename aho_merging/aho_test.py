"""Code to test aho_construction against subset construction."""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sys
import random

sys.path.append('./..')
sys.path.append('./../automata')
from automata import Automata
import aho_construction
import single_dfa_construction
import time_compare

def build_random_string(length, alphabet):
    """Build a random string to use.

    Args:
        length: The length of the string to construct.
        alphabet: The alphabet to use in the string construction
    Returns:
        The generalized random string.
    """
    random_string = []
    for _ in range(length):
        rand_code = random.randint(0,
                                   2 ** len(alphabet)
                                   - 2) + 1
        rand_letters = []
        for digit in range(len(alphabet)):
            if rand_code & (1 << digit) > 0:
                rand_letters.append(alphabet[digit])
        random_string.append(rand_letters)
    return random_string


def aho_to_dfa(root_node, string_len):
    """Converts dfa in aho data structure into Automata data structure.

    Args:
        root_node: The root of the aho_construction result.
        string_len: The length of the string being recognized.
    Returns:
        Pointer to the new Automata object.
    """
    # Convert the object-pointer style DS to Automata.
    def get_name(state, name=None):
        """Gets the Automat equivalent state name."""
        if name is None:
            name = []
        name.append(state.level)
        if state.level == 0:
            return tuple(name[::-1])
        else:
            return get_name(state.fallback, name)

    # Traverse in aho structure in BFS structure.
    aho_dfa = Automata()
    queue = deque()
    queue.appendleft(root_node)
    seen = [root_node]
    while len(queue) > 0:
        curr_state = queue.pop()
        curr_name = get_name(curr_state)
        # See if we should add the node to the DFA.
        if curr_name not in aho_dfa.states:
            aho_dfa.add_state(curr_name, is_start=(curr_state is root_node),
                              is_terminal=(string_len in curr_name))
        # Add the transitions for the node.
        for letter, child in curr_state.transitions.iteritems():
            child_name = get_name(child)
            if child_name not in aho_dfa.states:
                aho_dfa.add_state(child_name,
                                  is_start=(child is root_node),
                                  is_terminal=(string_len in child_name))
            aho_dfa.add_transition(curr_name, child_name, letter)
            # If we haven't seen the node yet add it to the queue
            if child not in seen:
                queue.appendleft(child)
                seen.append(child)

    return aho_dfa

def compare_methods(string, alphabet):
    """Compares aho algorithm with subset construction algorithm.

    Args:
        string: The generalized string to compare.
        alphabet: The alphabet of the string.
    Returns:
        Boolean whether the algorithms produce the same output.
    """
    root = aho_construction.construct_dfa(string, alphabet)
    root_dfa = aho_to_dfa(root, len(string))
    sub_dfa = single_dfa_construction.binary_subset_construction(string)
    return root_dfa == sub_dfa

def run_exhaustive_test(max_size, trials, alphabet):
    """Compare techniques for many different sized strings.
    Args:
        max_size: The max string size to consider.
        trials: How many trials to run on each string size.
    """
    for size in range(3, max_size + 1):
        incorrect = 0
        for _ in range(trials):
            string = time_compare.build_random_string(size)
            if not compare_methods(string, alphabet):
                incorrect += 1
        print '%d: %d/%d incorrect.' % (size, incorrect, trials)

def deduce_failures_calculated(string, alphabet):
    """Constructs the aho based on the string and then tries to find the number
       of failures calculated.

    Args:
        string: The generalized string to compare.
        alphabet: The alphabet of the string.
    Returns:
        Integer of how many failure computations were made.
    """
    root = aho_construction.construct_dfa(string, alphabet)
    # Data structure representing the number of states by level.
    num_states = [0 for _ in range(len(string) + 1)]
    # The names of the states we have seen so far.
    seen = []

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
                num_states[linked_node.level] += 1

    # Now multiply out the number of levels for each with the number of
    # possible characters in the next level.
    num_calcs = 0
    for level_index in range(len(string)):
        num_calcs += num_states[level_index] * len(string[level_index])

    return num_calcs

def analyze_data(data, make_histogram=False, title=None, upper_bound=None,
        num_bins=100):
    """Analyszes the data by estimating the mean and variance.

    Args:
        data: The data to analyze.
        make_histogram: Whether to print out a histogram of the data.
        title: Title to put on the graph.
        upper_bound: Upper bound data to plot alongside.
    Returns:
        Tuple containing (mean, variance).
    """
    mean = np.mean(data)
    var = np.var(data)
    if make_histogram:
        if upper_bound:
            maximum = max([max(data), max(upper_bound)])
        else:
            maximum = max(data)
        bins = np.linspace(0, maximum + 10, num_bins)
        _, _, _ = plt.hist(data, bins, facecolor='blue', alpha=0.5)
        if upper_bound is not None:
            _, _, _ = plt.hist(upper_bound, bins, facecolor='green', alpha=0.5)
        plt.xlabel('Failure Calculations')
        if title is None:
            title = 'Histogram of Failure Calculations'
        plt.title(' '.join([title, 'Mean = %d Var = %d' % (mean, var)]))
        plt.grid(True)
        plt.show()
    return (mean, var)

def simulate_failure_dist(trials, string_len):
    """Generates random strings and finds how many failures were calc'd.

    Args:
        trials: The number of samples to pull from the dist.
        string_len: The length of the random strings to be generated.
        alphabet: The alphabet of the strings.
    Returns:
        List of samples generated.
    """
    data = []
    upper_bound = []
    alphabet = ['A', 'C', 'G', 'T']
    for _ in range(trials):
        string = time_compare.build_random_string(string_len)
        data.append(deduce_failures_calculated(string, alphabet))
        bound = 1
        for level in string:
            bound *= len(level)
        upper_bound.append(bound)
    return data, upper_bound

def generate_hists(length_range, trials, upper_bounds=False):
    """Generates histograms for the given length range.

    Args:
        length_range: Tuple representing the start and stop of the length
            of random strings to evaluate.
        trials: The number of samples to pull from the dist.
        upper_bounds: Whether to include the upper_bounds.
    """
    for length in range(length_range[0], length_range[1]):
        sample, sample_upper = simulate_failure_dist(trials, length)
        hist_title = 'Failures Calculated: String Length %d' % length
        if upper_bounds:
            analyze_data(sample, make_histogram=True, upper_bound=sample_upper,
                    title=hist_title)
        else:
            analyze_data(sample, make_histogram=True, title=hist_title)


def plot_trends(length_range, trials):
    """Generates histograms for the given length range.

    Args:
        length_range: Tuple representing the start and stop of the length
            of random strings to evaluate.
        trials: The number of samples to pull from the dist.
        upper_bounds: Whether to include the upper_bounds.
    """
    # Compute averages and std devs for plotting
    avgs = []
    std_devs = []
    for length in range(length_range[0], length_range[1]):
        sample, _ = simulate_failure_dist(trials, length)
        length_avg, length_var = analyze_data(sample)
        avgs.append(length_avg)
        std_devs.append(np.sqrt(length_var))

    plt.errorbar(range(length_range[0], length_range[1]), avgs, yerr=std_devs)
    plt.title('Average Failure Calculations vs. String Length')
    plt.xlabel('String Length')
    plt.ylabel('Failure Calculations')
    plt.xlim([length_range[0] - 1, length_range[1]])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    root = aho_construction.construct_dfa([['A', 'G'], ['T'], ['A', 'T'], ['A', 'C', 'G', 'T'], ['A', 'G']], ['A', 'C', 'G', 'T'])
    aho_construction.print_dfa(root)
    # print aho_to_dfa(root, 5)
    # plot_trends((3, 10), 10000)
