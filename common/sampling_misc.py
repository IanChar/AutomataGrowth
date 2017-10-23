"""
All of the sampling that doesn't quite fit into sampler.py. Most likely
because we are not looking at trend with depth.
"""

from __future__ import division
import sys
from collections import deque
import pandas as pd

from string_comparisons import prefix_substring_match

sys.path.append('..')
import merge_alg

sys.path.append('../aho_merging')
import arbitrary_probs_util as string_util

def sample_stat_failure_chain_dist(num_samples, probs, depth):
    """Sample to estimate the stationary distribution of failure chain length.
    Args:
        num_samples: The number of samples to estimate from.
        probs: The probabilities of seeing each letter in a set.
        depth: The depth to go to in order to start sampling.
    Returns: List representing the probability vector.
    """
    alphabet = string_util.get_default_alphabet(len(probs))
    observed = []
    for _ in xrange(num_samples):
        gen_string = string_util.create_random_string(probs, depth)
        root = merge_alg.aho_merge(gen_string, alphabet).get_root()
        # Iterate through to find first state at appropriate depth.
        state = root
        for child in state.goto.values():
            if child is not state:
                state = child
                break
        while state.depth < depth:
            state = state.goto.values()[0]
        chain_length = _get_failure_chain_length(state)
        while len(observed) < chain_length:
            observed.append(0)
        observed[chain_length - 1] += 1
    return [o / num_samples for o in observed]

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

def sample_failure_depth_size_trend(num_samples, probs, depth1, depth2):
    """Sample whether depth2 has a failure to depth1, how many states depth
       1 has, and if G[1:depth1] == G[depth2-depth1:depth1]
    Args:
        num_samples: The number of samples to look at.
        probs: The probability of seeing each letter in a set.
        depth1: The depth where we count number of states.
        depth2: Want to see if depth2 has failure to depth1.
    Returns: DataFrame with 'has_failure' (0 or 1), 'depth_size', and
        'string_match' (0 or 1).
    """
    has_failure = []
    depth_size = []
    string_match = []
    alphabet = string_util.get_default_alphabet(len(probs))
    for _ in xrange(num_samples):
        gen_string = string_util.create_random_string(probs, depth2)
        # Check the string first.
        string_match.append(prefix_substring_match(gen_string,
                                                   depth2 - depth1, depth1))
        dfa = merge_alg.aho_merge(gen_string, alphabet)
        root = dfa.get_root()
        # Iterate over graph once instead of helper functions.
        curr_depth_size = 0
        curr_has_failure = 0
        queue = deque()
        queue.appendleft(root)
        seen = set()
        seen.add(root.sid)
        while len(queue) > 0:
            curr = queue.pop()
            if curr.depth == depth1:
                curr_depth_size += 1
            elif curr.depth == depth2 and curr.failure.depth == depth1:
                curr_has_failure = 1
                break # Can break here since doing BFS
            for child in curr.goto.values():
                if child.sid not in seen:
                    queue.appendleft(child)
                    seen.add(child.sid)

        has_failure.append(curr_has_failure)
        depth_size.append(curr_depth_size)
    # Return lists as a pandas dataframe.
    return pd.DataFrame({'string_match': string_match,
                         'has_failure': has_failure,
                         'depth_size': depth_size})

def sample_total_states(num_samples, probs, length):
    """Sample the total number of states in the DFA.
    Args:
        num_samples: The number of samples to take.
        probs: The probabilities of seeing each letter.
        length: The length of the generalized string to test.
    Returns: Tuple with first thing being average and second being max.
    """
    max_states = 0
    avg_states = 0
    alphabet = string_util.get_default_alphabet(len(probs))
    for _ in xrange(num_samples):
        gen_string = string_util.create_random_string(probs, length)
        dfa = merge_alg.aho_merge(gen_string, alphabet)
        num_states = dfa.get_num_states()
        avg_states += num_states
        if max_states < num_states:
            max_states = num_states
    return (avg_states / num_samples, max_states)

def sample_signature_size(num_samples, probs, depth):
    """Sample the signature size at the depth.
    Args:
        num_samples: Number of samples to do.
        probs: The probabilities of seeing a letter.
        depth: The depth at which to sample at.
    Returns: List of the samples drawn.
    """
    alph = string_util.get_default_alphabet(len(probs))
    samples = []
    for _ in xrange(num_samples):
        gen_string = string_util.create_random_string(probs, depth)
        dfa = merge_alg.aho_merge(gen_string, alph)
        # Do DFS until we see state of appropriate depth.
        root = dfa.get_root()
        stack = deque()
        stack.append(root)
        seen = set([root.sid])
        while len(stack) > 0:
            curr = stack.pop()
            if curr.depth == depth:
                samples.append(_get_failure_chain_length(curr))
                break
            for child in curr.goto.values():
                if child.sid not in seen:
                    stack.append(child)
                    seen.add(child.sid)
    return samples


def _get_states_at_depth(root, depth):
    """Assemble all states at the specified depth."""
    queue = deque()
    queue.appendleft(root)
    seen = set()
    seen.add(root.sid)
    states = []
    while len(queue) > 0:
        curr = queue.pop()
        if curr.depth == depth:
            states.append(curr)
        else:
            for child in curr.goto.values():
                if child.sid not in seen:
                    queue.appendleft(child)
                    seen.add(child.sid)
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
    # print sample_state_has_new_thread(10, [0.5 for _ in range(4)], 15)
    # print sample_state_has_new_thread(10, [0.5 for _ in range(4)], 15, [2])
    # print sample_stat_failure_chain_dist(10, [0.5 for _ in range(4)], 1)
    print sample_failure_depth_size_trend(10, [0.5 for _ in range(4)], 3, 7)
