"""Helpers/samplers for signature information."""
from __future__ import division
import sys
import pandas as pd
from collections import deque

sys.path.append('..')
import merge_alg
import arbitrary_probs_util as string_util


def get_signature(state):
    """Gets the signature of the state and returns as a tuple."""
    sig = deque()
    ptr = state
    while ptr is not None:
        sig.appendleft(ptr.depth)
        ptr = ptr.failure
    return tuple(sig)

def find_signature_size(root, sizes):
    """Traverse a minimal dfa and return the first state of appropriate size.
    Args:
        root: The root of the DFA.
        sizes: List of valid sizes.
    Returns: Valid state, tuple of signature or None,None if none found.
    """
    stack = deque()
    stack.append(root)
    seen = set()
    seen.add(root.sid)
    while len(stack) > 0:
        curr = stack.pop()
        sig = get_signature(curr)
        if len(sig) in sizes:
            return curr, sig
        for child in curr.goto.values():
            if child.sid not in seen:
                seen.add(child.sid)
                stack.append(child)
    return None, None

class SignatureSampler(object):
    """Class for sampling properties of signatures."""

    def __init__(self, probs, length, sig_sizes):
        """Constructor.
        Args:
            probs: List of probs for seeing each letter.
            legnth: The length of the random generalized string.
            sig_sizes: List of the signature sizes to look for.
        """
        self.probs = probs
        self.length = length
        self.sig_sizes = sig_sizes
        self.alphabet = string_util.get_default_alphabet(len(probs))
        self.SAMPLE_MAP = {
                'offspring': self._get_offspring
        }

    def draw_samples(self, samples, props):
        """Sample the props given as a list and return pandas dataframe."""
        df_columns = list(props)
        df_columns.insert(0, 'signature')
        to_return = pd.DataFrame(columns=df_columns)
        # Draw sample.
        for _ in xrange(samples):
            state, sig = None, None
            while state is None:
                gen_string = string_util.create_random_string(self.probs,
                                                          self.length)
                min_dfa = merge_alg.aho_merge(gen_string, self.alphabet)
                root = min_dfa.get_root()
                # Find a valid state.
                state, sig = find_signature_size(root, self.sig_sizes)
            # Analyze given properties of state.
            sample_props = [self.SAMPLE_MAP[prop](state) for prop in props]
            sample_props.insert(0, [sig])
            df_to_add = pd.DataFrame({col: sample_props[index]
                                      for index, col in enumerate(df_columns)})
            to_return = to_return.append(df_to_add)
        return to_return.apply(pd.to_numeric, errors='ignore')

    def _get_offspring(self, state):
        """Get the amount of offspring for the state and return integer."""
        return len(state.goto)

if __name__ == '__main__':
    SS = SignatureSampler([0.5 for _ in range(4)], 15, [3])
    DF = SS.draw_samples(10, ['offspring'])
    print DF
