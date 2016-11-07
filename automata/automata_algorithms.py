"""Tool for creating different types of automata."""

import automata

# Constants representing the indices for the state's attributes..
ACTIVE_STRINGS_ATTR = 0
LEVEL_ATTR = 1
NUM_STATE_ATTRS = 2

def construct_simple_nfa(strings):
    """Constructs a simple NFA froma string using algorithm given by Marschall.

    Args:
        strings: A list of generalized strings where each string is represented
            as a list of of lists.
            e.g. "{A, B, C}{E}" -> [["A", "B", "C"], [E]]
    Returns:
        An Automata object representing the simple NFA.
    Raises:
        ValueError: Strings must have the same length.
    """
    # Check if the string lengths are all the same.
    str_length = len(strings[0])
    for string in strings:
        if len(string) != str_length:
            raise ValueError('Strings must have the same length.')

    # Initialize automata and set the last node.
    nfa = automata.Automata()
    # Make the state name ((<Active strings>), <Level>).
    terminal_state_name = _make_state_name(range(len(strings)), str_length)
    nfa.add_state(terminal_state_name, is_terminal=True)

    # Iterate through the levels.
    for lvl in range(str_length, 0, -1):
        # Iterate through the last level's states and add parents.
        prev_lvl = _get_states_at_level(nfa, lvl)
        for prev_state in prev_lvl:
            _add_state_parents(nfa, prev_state, strings)

    return nfa

def subset_construction(nfa):
    """Construct a DFA from an NFA using subset construction.

    Args:
        nfa: Automata of an NFA.
    Returns:
        An automata of the corresponding DFA.
    """
    print nfa

def _add_state_parents(nfa, state_name, strings):
    """Add the parents and their transitions for a given state in an automata.

    Args:
        nfa: The Automata object to add parents to.
        state_name: The to add parents to.
        strings: A list of generalized strings where each string is represented
            as a list of of lists.
            e.g. "{A, B, C}{E}" -> [["A", "B", "C"], [E]]
    Returns:
        The altered nfa Automata object.
    """
    # Get state information.
    active_strings = state_name[ACTIVE_STRINGS_ATTR]
    level = state_name[LEVEL_ATTR]

    # Find all possible transitions to this state.
    transitions = set([])
    for string_index in active_strings:
        curr_string = strings[string_index]
        curr_transitions = curr_string[level - 1]
        for letter in curr_transitions:
            transitions.add(letter)

    # Loop over the transitions and add states/transitions.
    for letter in transitions:
        # Find the active strings that would be in the parent.
        active_in_parent = []
        for string_index in active_strings:
            curr_string = strings[string_index]
            if letter in curr_string[level - 1]:
                active_in_parent.append(string_index)
        # See if the parent still needs to be added to nfa.
        parent_name = _make_state_name(active_in_parent, level - 1)
        if parent_name not in nfa.states:
            nfa.add_state(parent_name, is_start=(level - 1 == 0))
        # Add the transition to the parent.
        nfa.add_transition(parent_name, state_name, letter)


def _make_state_name(active_strings, level):
    """Make a state name with the form ((<ACTIVE_STRINGS>), <LEVEL>).

    Args:
        active_strings: The list of active strings where each element is
            the index of where the string appears in the list of strings.
        level: The level of the state.
    Returns:
        A nested tuple object representing the name of the state.
    """
    state_name = [None for _ in range(NUM_STATE_ATTRS)]
    state_name[ACTIVE_STRINGS_ATTR] = tuple(sorted(active_strings))
    state_name[LEVEL_ATTR] = level
    return tuple(state_name)

def _get_states_at_level(nfa, level):
    """Returns a list of the states in the nfa at the specified level.

    Args:
        nfa: The Automata object to read states from.
        level: The level of the states to get.
    """
    level_states = []
    for state in nfa.states:
        if state[LEVEL_ATTR] == level:
            level_states.append(state)
    return level_states
