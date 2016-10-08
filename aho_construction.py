"""Algorithm to create a DFA inspired by Aho-Corasick"""
from collections import deque

class State(object):
    """States that compose the DFA.

    Properties:
        name: Name for the state.
        level: The level in the aho corasick this state represents.
        unique_id: ID to differentiate state.
        transitions: The transitions of the state represented as a dictionary
            mapping a letter to a state.
        fallback: The state that we fallback to when seeing a mismatch.
    """
    def __init__(self, name, level, unique_id):
        self.name = name
        self.level = level
        self.unique_id = unique_id
        self.transitions = {}
        self.fallback = None

    def move_to(self, letter):
        """Move through the automata given some letter.

        Args:
            letter: Letter being used to transition.
        Returns:
            Pointer to the next state.
        """
        if letter not in self.transitions.keys():
            raise KeyError('Invalid transition: %s' % letter)
        return self.transitions[letter]

    def set_fallback(self, state):
        """Add a fallback state.

        Args:
            state: Pointer to the fallback state.
        """
        if self.fallback is not None:
            raise ValueError('State already has fallback.')
        self.fallback = state

    def add_transitions(self, to_add):
        """Update the transtions with to_add.

        Args:
            to_add: A dictionary used to update the transtions.
        """
        self.transitions.update(to_add)

    def copy_fallback_transitions(self):
        """Copies any missing transitions from the fallback."""
        if self.fallback is None:
            raise RuntimeError('Cannot copy transitions, fallback not set')
        if self.fallback is not self:
            for key, value in self.fallback.transitions.iteritems():
                if key not in self.transitions.keys():
                    self.transitions[key] = value

    def __str__(self):
        to_print = []
        to_print.append('\nNode: (%d; %s; %d)' % (self.level, self.name,
                                                  self.unique_id))
        to_print.append('Transitions:')
        for key, value in self.transitions.iteritems():
            to_print.append('%s -> (%d; %s)' % (key, value.level, value.name))
        return '\n'.join(to_print)

    def __eq__(self, other):
        """Check if equal to another state. Does not check transitions."""
        if not isinstance(other, State):
            return False
        if self.name != other.name:
            return False
        if self.level != other.level:
            return False
        if self.fallback is not other.fallback:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

def construct_dfa(string, alphabet):
    """Constructs a dfa using aho-corasick construction.

    Args:
        string: Nested list of character representing the generalized string.
        alphabet: List of the characters in the alphabet
    Returns:
        Pointer to the root state.
    """
    state_id = 0
    # Initialize root state and fill in some known properties.
    root_state = State('Epsilon', 0, state_id)
    state_id += 1
    root_state.set_fallback(root_state)
    # Add transitions for root.
    # Note we know the first level must be one state since all of their
    # fallbacks must be to the root.
    level_1 = State(','.join(string[0]), 1, state_id)
    state_id += 1
    level_1.set_fallback(root_state)
    for letter in alphabet:
        if letter in string[0]:
            root_state.transitions[letter] = level_1
        else:
            root_state.transitions[letter] = root_state

    # Perform a BFS process and "Discover" the DFS as we go.
    queue = deque()
    queue.appendleft(level_1)
    while len(queue) > 0:
        curr_state = queue.pop()

        if curr_state.level < len(string):
            # Find the children of curr_state and add them to queue.
            fall_groupings = _group_falls(curr_state, root_state, string)
            for fall_group in fall_groupings.values():
                new_state = State(','.join(fall_group[1:]),
                                  curr_state.level + 1, state_id)
                new_state.set_fallback(fall_group[0])
                # Check if we can merge this state with something in the same
                # level (i.e. in the queue).
                is_new = True
                for added_state in queue:
                    if added_state == new_state:
                        new_state = added_state
                        is_new = False
                        break
                curr_state.add_transitions({l:new_state
                                           for l in fall_group[1:]})
                if is_new:
                    queue.appendleft(new_state)
                    state_id += 1

        # Fill out the rest of the transitions for curr_state.
        curr_state.copy_fallback_transitions()

    return root_state


def _group_falls(curr_state, root_state, string):
    """Group the falls for states with curr_state as a parent by fallback.

    Args;
        curr_state: The parent of the states we want to look at.
        root_state: The roote state of the current dfa.
        string: The generalized string we are trying to recognize.
    Returns:
        Dictionary mapping (fallback name, fallback key) -> list where the
        first element is a pointer to the actual fallback and the rest is the
        letters that fallback to this.
    """
    fall_mapping = {}
    def add_to_mapping(fallback, letter):
        """Short helper function to correctly add to fall_mapping.

        Args:
            fallback: The state that is being fallen back to.
            letter: The letter that causes us to fall there.
        """
        if fallback.unique_id not in fall_mapping:
            fall_mapping[fallback.unique_id] = [fallback, letter]
        else:
            fall_mapping[fallback.unique_id].append(letter)

    # Compute the fallback state for each possible foward transition
    # based on three options...

    for fwd_letter in string[curr_state.level]:
        # Start with the parent's fallback and see if we can expand, if not
        # check its fallback, then its fallback, etc. back to root.
        prev_fall = None
        curr_fall = curr_state.fallback
        fallback_found = False
        while prev_fall is not curr_fall:
            if curr_fall.level <= curr_fall.move_to(fwd_letter).level:
                # We have found our fallback.
                add_to_mapping(curr_fall.move_to(fwd_letter), fwd_letter)
                fallback_found = True
                break
            else:
                prev_fall = curr_fall
                curr_fall = curr_fall.fallback
        # If no non-epsilon fallback could be found, epsilon is our fallback.
        if not fallback_found:
            add_to_mapping(root_state, fwd_letter)
    return fall_mapping

def print_dfa(root_state):
    """Print the DFA in BFS order.

    Args:
        root_state: The root state in the DFA.
    """
    queue = deque()
    queue.appendleft(root_state)
    seen = []
    while len(queue) > 0:
        curr = queue.pop()
        print curr
        seen.append(curr)
        for child in curr.transitions.values():
            if child not in seen and child not in queue:
                queue.appendleft(child)
