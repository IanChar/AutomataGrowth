"""Algorithm to create an Aho-Corasick Automata."""
from node import Node
from collections import deque

def create_trie(strings):
    """Creates a trie from the given strings.

    Args:
        strings: A list of strings.
    Returns:
        The root node in the Trie structure.
    """
    root_node = Node(0)
    # Add each string to the trie.
    for string in strings:
        curr_node = root_node
        for level, letter in enumerate(string):
            if letter not in curr_node.transitions.keys():
                curr_node.add_transition(letter, Node(level + 1))
            curr_node = curr_node.transitions[letter]
    return root_node

def add_fall_transitions(root_node):
    """Add the fall transitions for the nodes.

    Args:
        root_node: The root of the trie.
    """
    # Make the root its own fall node.
    root_node.add_fallback(root_node)

    # Do a BFS traversal of the nodes.
    queue = deque()
    for child in root_node.transitions.values():
        queue.appendleft(child)

    while len(queue) > 0:
        curr_node = queue.pop()
        # Add children.
        for child in curr_node.transitions.values():
            queue.appendleft(child)

        # If the parent is the root we have to make the root the fallback.
        if curr_node.parent is root_node:
            curr_node.add_fallback(root_node)
        # Otherwise we have two options...
        # 1) Check the parent's fall and see if we can expand on that.
        elif curr_node.letter in curr_node.parent.fallback.transitions.keys():
            curr_node.add_fallback(curr_node.parent.fallback
                                   .transitions[curr_node.letter])
        # 2) Check if we can expand from the root.
        elif curr_node.letter in root_node.transitions.keys():
            curr_node.add_fallback(root_node.transitions[curr_node.letter])
        # 3) Else we have to fall to the root.
        else:
            curr_node.add_fallback(root_node)

