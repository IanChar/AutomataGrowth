"""Node structure used to create aho-corasick automata"""

class Node(object):
    """Node object for forming the aho-corasick.

    Properties:
        level: Number representing the level in the trie the node is.
        transitions: The transitions to other nodes in the trie. This is a
            dictionary linking a letter in the alphabet to a Node.
        fallback: The node in the tree to fall back to given a failure.
        parent: The parent node that links to this.
        letter: The letter read to get to this node.
    """

    def __init__(self, level):
        self.level = level
        self.transitions = {}
        self.fallback = None
        self.parent = None
        self.letter = None

    def add_transition(self, letter, node):
        """Add a transition to the node."""
        if letter in self.transitions.keys():
            raise KeyError('%s had already been added.' % letter)
        self.transitions[letter] = node
        node.parent = self
        node.letter = letter

    def add_fallback(self, node):
        """Add a fallback to the node which would be taken if no transition."""
        self.fallback = node

    def __str__(self):
        to_print = []
        to_print.append('\nNode: (%s, %d)' % (self.letter,
                                              self.level))
        to_print.append('\nTransitions %s' % str(self.transitions.keys()))
        to_print.append('\nFall: (%s, %d)\n' % (self.fallback.letter,
                                                self.fallback.level))
        for child in self.transitions.values():
            to_print.append(str(child))
        return ''.join(to_print)
