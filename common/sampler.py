"""Util for sampling properties of the minimal DFA."""

from __future__ import division
import sys
import pandas as pd
from collections import deque


sys.path.append('..')
import merge_alg

sys.path.append('../aho_merging')
import arbitrary_probs_util as string_util

class DepthSampler(object):
    """Class for sampling properties by depth of the minimal DFA."""

    def __init__(self, probs, length):
        """Constructor for Sampler
        Args:
            probs: List of probs of seeing each letter.
            length: The length of the generalized strings to sample.
        """
        self.probs = probs
        self.length = length
        self.alphabet = string_util.get_default_alphabet(len(probs))
        self.SAMPLE_MAP = {
            'number_of_states': self._get_states_per_depth,
            'failure_dest': None,
            'failures_to': None,
        }


    def draw_samples(self, num_samples, props):
        """Draws samples and finds the requested features.
        Args:
            num_samples: Number of samples to observe.
            props: The properties to be observed, should be key in SAMPLE_MAP.
        Returns: Pandas dataframe with samples and requested properties.
        """
        df_columns = list(props)
        df_columns.insert(0, 'depth')
        to_return = pd.DataFrame(columns=df_columns)
        # Draw sample.
        for _ in xrange(num_samples):
            gen_string = string_util.create_random_string(self.probs,
                                                          self.length)
            min_dfa = merge_alg.aho_merge(gen_string, self.alphabet)
            root = min_dfa.get_root()
            # Analyze given properties.
            sample_props = [self.SAMPLE_MAP[prop](root) for prop in props]
            depths = range(len(sample_props[0]))
            sample_props.insert(0, depths)
            df_to_add = pd.DataFrame({col: sample_props[index]
                                      for index, col in enumerate(df_columns)})
            to_return = to_return.append(df_to_add)
        return to_return.astype(float)

    def _get_states_per_depth(self, root):
        """Finds the number of states per depth.
        Args:
            root: Root of the DFA.
        Returns: List where each entry is how many states are in the depth
            corresponding with the index.
        """
        to_return = []
        queue = deque()
        queue.appendleft((root, 0))
        seen = set()
        seen.add(root.sid)
        while len(queue) > 0:
            curr_node, depth = queue.pop()
            if len(to_return) < depth + 1:
                to_return.append(0)
            to_return[depth] += 1
            for child in curr_node.goto.values():
                if child.sid not in seen:
                    queue.appendleft((child, depth + 1))
                    seen.add(child.sid)
        return to_return
