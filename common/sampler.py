"""Util for sampling properties of the minimal DFA."""

from __future__ import division
import sys
import pandas as pd
from collections import deque


sys.path.append('..')
import merge_alg
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
            'states': self._get_states_per_depth,
            'threads': self._get_threads_per_depth,
            'thread_children': self._get_thread_children,
            'total_children': self._get_total_children,
            'new_thread': self._get_has_new_thread,
            'merge_degree': self._get_merged_degree,
            'failure_chain_lengths': self._get_failure_chain_lengths,
            'failure_chain_path': self._get_single_failure_chain_path,
            'growth_ratio': self._get_growth_ratio,
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
        return to_return.apply(pd.to_numeric, errors='ignore')

    def _get_states_per_depth(self, root):
        """Finds the number of states per depth.
        Args:
            root: Root of the DFA.
        Returns: List where each entry is how many states are in the depth
            corresponding with the index.
        """
        to_return = []
        queue = deque()
        queue.appendleft(root)
        seen = set()
        seen.add(root.sid)
        while len(queue) > 0:
            curr_node = queue.pop()
            if len(to_return) < curr_node.depth + 1:
                to_return.append(0)
            to_return[curr_node.depth] += 1
            for child in curr_node.goto.values():
                if child.sid not in seen:
                    queue.appendleft(child)
                    seen.add(child.sid)
        return to_return

    def _get_threads_per_depth(self, root):
        """Finds the number of threads (failures not going to root) per depth.
        Args:
            root: Root of the DFA.
        Returns: List where each entry is how many states are in the depth
            corresponding with the index.
        """
        to_return = []
        queue = deque()
        queue.appendleft(root)
        seen = set()
        seen.add(root.sid)
        while len(queue) > 0:
            curr_node = queue.pop()
            if len(to_return) < curr_node.depth + 1:
                to_return.append(0)
            if _is_thread(curr_node):
                to_return[curr_node.depth] += 1
            for child in curr_node.goto.values():
                if child.sid not in seen:
                    queue.appendleft(child)
                    seen.add(child.sid)
        return to_return

    def _get_thread_children(self, root):
        """Samples the number of children that each thread has for depths.
        Critical decision made for this is that only count children once ever.
        i.e. if two threads merge the offspring is only counted for one of them.
        Args:
            root: Root of the DFA.
        Returns: List of lists where each inner list has the samples and its
            index is the depth in the DFA.
        """
        to_return = []
        queue = deque()
        queue.appendleft(root)
        seen = set()
        seen.add(root.sid)
        while len(queue) > 0:
            curr_node = queue.pop()
            if len(to_return) < curr_node.depth + 1:
                to_return.append([])
            thread_children = 0 if _is_thread(curr_node) else None
            for child in curr_node.goto.values():
                if child.sid not in seen:
                    queue.appendleft(child)
                    seen.add(child.sid)
                    if (_is_thread(curr_node) and _is_offspring(child)
                            and child.failure.depth > 1):
                        thread_children += 1
            if thread_children is not None:
                to_return[curr_node.depth].append(thread_children)
        return to_return

    def _get_total_children(self, root):
        """Samples the TOTAL number of children that each thread has for depths.
        This will count every child, i.e. double count merged children.
        Args:
            root: Root of the DFA.
        Returns: List of lists where each inner list has the samples and its
            index is the depth in the DFA.
        """
        to_return = []
        queue = deque()
        queue.appendleft(root)
        seen = set()
        seen.add(root.sid)
        while len(queue) > 0:
            curr_node = queue.pop()
            if len(to_return) < curr_node.depth + 1:
                to_return.append([])
            if _is_thread(curr_node):
                thread_children = 0
                for child in set(curr_node.goto.values()):
                    if _is_thread(child):
                        thread_children += 1
                to_return[curr_node.depth].append(thread_children)
            for child in curr_node.goto.values():
                if child.sid not in seen:
                    queue.appendleft(child)
                    seen.add(child.sid)
        return to_return

    def _get_has_new_thread(self, root):
        """Finds whether each depth has a new thread or not.
        Args:
            root: The root of the DFA.
        Returns: List where there is a 0 if the corresponding depth does not
            have a new thread and 1 if it does.
        """
        to_return = []
        queue = deque()
        queue.appendleft(root)
        seen = set()
        seen.add(root.sid)
        while len(queue) > 0:
            curr_node = queue.pop()
            if len(to_return) < curr_node.depth + 1:
                to_return.append(0)
            if curr_node.failure is not None and curr_node.failure.depth == 1:
                to_return[curr_node.depth] = 1
            for child in curr_node.goto.values():
                if child.sid not in seen:
                    queue.appendleft(child)
                    seen.add(child.sid)
        return to_return

    def _get_merged_degree(self, root):
        """If a thread comes from the merging of multiple threads, returns
        number of parents for that node minus one.
        Args:
            root: The root of the DFA.
        Returns: List of lists where each inner list contains degree for each
            merged node in the depth corresponding to the inner list index.
        """
        queue = deque()
        queue.appendleft(root)
        seen = set()
        seen.add(root.sid)
        # List of dictionaries where index i corresponds to depth i + 1.
        degree_mappings = [{}]
        while len(queue) > 0:
            curr_node = queue.pop()
            if len(degree_mappings) < curr_node.depth + 2:
                degree_mappings.append({})
            if _is_thread(curr_node):
                degrees = degree_mappings[curr_node.depth + 1]
                for child in set(curr_node.goto.values()):
                    if _is_thread(child):
                        if child.sid in degrees.keys():
                            degrees[child.sid] += 1
                        else:
                            degrees[child.sid] = 0
            for child in curr_node.goto.values():
                if child.sid not in seen:
                    queue.appendleft(child)
                    seen.add(child.sid)
        # Trim off last entry since the last depth has no children.
        return ([[deg for deg in depth.values() if deg > 0]
                 for depth in degree_mappings][:-1])

    def _get_failure_chain_lengths(self, root):
        """Get the length of the failure chain for each state.
         Args:
            root: The root of the DFA.
        Returns: List of lists where each inner list contains the length of the
            failure chain for each of the states in the depth.
        """
        queue = deque()
        queue.appendleft(root)
        seen = set()
        seen.add(root.sid)
        results = []
        while len(queue) > 0:
            curr_node = queue.pop()
            if len(results) < curr_node.depth + 1:
                results.append([])
            results[curr_node.depth].append(_count_failure_chain(curr_node))
            for child in curr_node.goto.values():
                if child.sid not in seen:
                    queue.appendleft(child)
                    seen.add(child.sid)
        return results

    def _get_single_failure_chain_path(self, root):
        """Get the lengths of failure chains for states across one single path.
        Args:
            root: The root of the DFA.
        Returns: List of integers each representing the size of the failure
            chain at that particular depth.
        """
        path = [0, 1]
        # Iterate through to find first state at appropriate depth.
        state = root
        for child in state.goto.values():
            if child is not state:
                state = child
                break
        while len(state.goto) > 0:
            state = state.goto.values()[0]
            chain_length = _count_failure_chain(state)
            path.append(chain_length)
        return path

    def _get_growth_ratio(self, root):
        """Get the growth ratio i.e. (states in depth n + 1)/(states in depth n)
        Args:
            root: The root of the DFA.
        Returns: list of the ratios, note the last index is 0.
        """
        # This is bad because could possibly call get states twice, doing double
        # the work rather than reusing but oh well.
        states = self._get_states_per_depth(root)
        ratios = [0 for _ in xrange(self.length + 1)]
        for ind in xrange(self.length):
            ratios[ind] = states[ind + 1] / states[ind]
        return ratios

def _count_failure_chain(state):
    count = 0
    curr = state
    while curr.failure is not None:
        curr = curr.failure
        count += 1
    return count

def _is_thread(state):
    return state.failure is not None and state.failure.depth > 0

def _is_offspring(state):
    return state.failure is not None and state.failure.depth > 1

if __name__ == '__main__':
    DS = DepthSampler([0.5 for _ in range(4)], 4)
    DF = DS.draw_samples(1, ['states', 'threads', 'thread_children',
                             'new_thread', 'merge_degree', 'total_children',
                             'failure_chain_lengths', 'failure_chain_path'])
    print DF
