"""
All of the sampling that doesn't quite fit into sampler.py. Most likely
because we are not looking at trend with depth.
"""

from __future__ import division
import sys
from collections import deque
import pandas as pd

sys.path.append('..')
import merge_alg

sys.path.append('../aho_merging')
import arbitrary_probs_util as string_util

def sample_state_has_new_thread(num_samples, probs, depth,
                                failure_length_conditions=None):
    """Samples whether a state at specified conditions has a new thread.
    Args:
        num_samples: The number of samples to draw.
        probs: The probabilities of seeing each letter in the set.
        depth: The depth that we should look for states at.
        failure_length_conditions: List of conditions for how long the failure
            chain should be for the state we are looking at. If None, there is
            no condition.
    Returns: Pandas DataFrame with 'failure_chain_length' and 'has_new_thread'.
    """
    alphabet = string_util.get_default_alphabet(len(probs))
    chain_lengths = []
    has_new_threads = []
    for _ in xrange(num_samples):
        # Assemble DFA.
        gen_string = string_util.create_random_string(probs, depth + 1)
        dfa = merge_alg.aho_merge(gen_string, alphabet)
        root = dfa.get_root()
        states = _get_states_at_depth(root, depth)
        # Look at DFA and add samples to lists.
        if failure_length_conditions is None:
            selected_state = states[0]
            chain_lengths.append(_get_failure_chain_length(selected_state))
            has_new_threads.append(_has_new_thread(selected_state))
        else:
            curr_chain_lengths = [_get_failure_chain_length(state)
                                  for state in states]
            for length in failure_length_conditions:
                try:
                    state_index = curr_chain_lengths.index(length)
                    chain_lengths.append(length)
                    has_new_threads.append(_has_new_thread(states[state_index]))
                except ValueError:
                    # There was no state with conditions.
                    pass
    # Return lists as a pandas dataframe.
    return pd.DataFrame({'failure_chain_length': chain_lengths,
                         'has_new_thread': has_new_threads})

def _get_states_at_depth(root, depth):
    """Assemble all states at the specified depth."""
    queue = deque()
    queue.appendleft(root)
    seen = set()
    states = []
    while len(queue) > 0:
        curr = queue.pop()
        seen.add(curr.sid)
        if curr.depth == depth:
            states.append(curr)
        else:
            for child in curr.goto.values():
                if child.sid not in seen:
                    queue.appendleft(child)
    return states

def _has_new_thread(state):
    """Returns 1 if state has a new thread, 0 otherwise."""
    for child in state.goto.values():
        if child.failure is not None and child.failure.depth == 1:
            return 1
    return 0

def _get_failure_chain_length(state):
    """Returns the number of failure transitions in the failure chain."""
    count = 0
    curr = state
    while curr.failure is not None:
        curr = curr.failure
        count += 1
    return count

if __name__ == '__main__':
    print sample_state_has_new_thread(10, [0.5 for _ in range(4)], 15)
    print sample_state_has_new_thread(10, [0.5 for _ in range(4)], 15, [2])
