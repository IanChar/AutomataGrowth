"""Object representation of a finite state automata."""

class Automata(object):
    """Object that represents a finite state Automata."""

    def __init__(self):
        self.states = {}
        self.transitions = {}

    def add_state(self, state):
        """Add a state to the automata."""
        pass

    def add_transition(self, from_state, to_state, letter):
        """Add a transition from one state to another based on a letter."""
        pass
