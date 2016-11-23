"""Perform analysis on the clash number distribution."""

from collections import deque

import aho_construction

def get_clashes(root):
    """Get the clash numbers for all nodes in the DFA.

    Args:
        root: The root of the DFA.
    Returns:
        A nested list of the clash numbers e.g. return_val[lvl][elem] is an int.
    """
    # Keep a list of dictionaries where each index of the list is a level
    # and each dictionary keeps track of how many times we have seen a forward
    # link leading to some node.
    levels = []
    queue = deque()
    seen = []

    # Do a BFS to fill out data structure.
    queue.appendleft(root)
    seen.append(root.unique_id)
    while len(queue) > 0:
        curr_node = queue.pop()

        for _, child in curr_node.transitions.iteritems():
            # Check all forward transtionsd
            if child.level == curr_node.level + 1:
                if len(levels) < child.level:
                    levels.append({})
                if child.unique_id in levels[child.level - 1]:
                    levels[child.level - 1][child.unique_id] += 1
                else:
                    levels[child.level - 1][child.unique_id] = 1
            # Add to the queue if we haven't seen before
            if child.unique_id not in seen:
                queue.appendleft(child)
                seen.append(child.unique_id)

    # Collapse dictionaries to only have frequency.
    levels = [levels[lvl].values() for lvl in range(len(levels))]
    to_return = []
    for lvl in levels:
        full_clash = []
        for elem in lvl:
            full_clash += [elem] * elem
        to_return.append(full_clash)
    return to_return

if __name__ == '__main__':
    print get_clashes(aho_construction.construct_dfa([['A', 'T'], ['A', 'T'],
            ['A', 'G'], ['A', 'C', 'T']], ['A', 'C', 'G', 'T']))
