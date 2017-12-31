"""Construct tree out of the Failure functions and analyze its growth."""
from collections import deque

import arbitrary_probs_util as string_util

import sys
sys.path.append('..')
import merge_alg


class TreeNode(object):

    def __init__(self, n_id):
        """Constructor
        Args:
            n_id: The node id.
            signature: The signature represented as a tuple.
        """
        self.n_id = n_id
        self.signature = None
        self.parent = None
        self.children = []
        self.depth = None


class FailureTree(object):

    def __init__(self, root):
        """Constructor.
        Args:
            root: The root for the DFA.
        """
        self.depth_sizes = []
        self.root = self._construct_tree(root)

    def _construct_tree(self, root):
        """Construct the failure tree.
        Args:
            root: The root for the DFA.
        """
        tree_root = TreeNode(root.sid)
        tree_root.signature = (0,)
        node_map = {root.sid: tree_root}

        # BFS to construct our tree.
        queue = deque()
        seen = set()
        seen.add(root.sid)
        self.depth_sizes.append(1)
        for child in root.goto.values():
            if child.sid not in seen:
                queue.appendleft(child)
                seen.add(child.sid)
        while len(queue) > 0:
            curr = queue.pop()
            node = self._add_node(curr, node_map)
            while len(self.depth_sizes) < node.depth + 1:
                self.depth_sizes.append(0)
            self.depth_sizes[node.depth] += 1
            for child in curr.goto.values():
                if child.sid not in seen:
                    queue.appendleft(child)
                    seen.add(child.sid)
        return tree_root

    def _add_node(self, state, node_map):
        """Convert the state to the corresponding node.
        Args:
            state: The state in the DFA.
            node_map: node_id -> node dictionary.
        Returns: The node and adds it to the map.
        """
        node = TreeNode(state.sid)
        parent = node_map[state.failure.sid]
        sig = [s for s in parent.signature]
        sig.append(state.depth)
        node.signature = tuple(sig)
        node.depth = len(sig) - 1
        node.parent = parent
        parent.children.append(node)
        node_map[state.sid] = node
        return node

if __name__ == '__main__':
    PROBS = [0.5 for _ in range(4)]
    LENGTH = 10
    ALPH = string_util.get_default_alphabet(len(PROBS))
    G = string_util.create_random_string(PROBS, LENGTH)
    DFA = merge_alg.aho_merge(G, ALPH)
    FT = FailureTree(DFA.root)
