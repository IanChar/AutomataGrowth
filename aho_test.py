"""Code to test aho_construction against subset construction."""
from collections import deque
from automata import Automata
import aho_construction
import single_dfa_construction
import time_compare

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

if __name__ == '__main__':
    run_exhaustive_test(10, 1000, ['A', 'C', 'G', 'T'])
