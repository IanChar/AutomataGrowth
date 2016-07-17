"""Object representation of a finite state automata."""

class Automata(object):
    """Object that represents a finite state Automata.

    Properties:
        states: A list of all the states where each state is a letter.
        start_states: A list of all the starting states.
        terminal_states: A list of all the terminal states.
        transitions: A nested dictionary representing the transition function.
            The dictionary maps the name of a state to another dictionary.
            This dictionary is the state's personal transitions where each
            letter in our alphabet maps to a set of possible transition
            states.
    """

    def __init__(self):
        self.states = []
        self.start_states = []
        self.terminal_states = []
        self.transitions = {}

    def add_state(self, state, is_start=False, is_terminal=False):
        """Add a state to the automata.

        Args:
            state: A string of the name of the state.
            is_start: Boolean indicating whether the state is a start state.
            is_terminal: Boolean indicating whether the state is a terminal
                state.
        Raises:
            ValueError: State already exists in automata.
        """
        if state in self.states:
            raise ValueError('State %s already exists in the Automata.'
                             % state)
        else:
            self.states.append(state)

        if is_start:
            self.start_states.append(state)
        if is_terminal:
            self.terminal_states.append(state)

    def add_transition(self, from_state, to_state, letter):
        """Add a transition from one state to another based on a letter.

        Args:
            from_state: The state that the transitions is from.
            to_state: The state that the transition is to.
            letter: The letter of the alphabet that causes the transition to
                occur.
        Raises:
            ValueError: State not in automata.
        """
        if from_state not in self.states or to_state not in self.states:
            raise ValueError('State not in automata.')

        # Check if from_state has transitions yet, if not make new dict.
        if from_state not in self.transitions:
            self.transitions[from_state] = {}

        state_transition = self.transitions[from_state]

        # If the letter has not been used yet make a new set.
        if letter not in state_transition:
            state_transition[letter] = set([to_state])
        else:
            state_transition[letter].add(to_state)

    def is_deterministic(self):
        """Decide whether the automata is deterministic or not.

        Returns:
            Boolean of whether the automata is deterministic or not.
        """
        # Check whether there is only one start state.
        if len(self.start_states) != 1:
            return False

        # Check whether the transitions have no epsilon transitions (in this
        # case '') and each letter has a unique transition.
        for state_transition in self.transitions.values():
            # Check for epsilon.
            if '' in state_transition:
                return False

            # Check for unique transitions.
            for transitions in state_transition.values():
                if len(transitions) > 1:
                    return False

        # Return True if all conditions pass.
        return True
