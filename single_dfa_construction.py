"""Algorithms used for constructing a DFA of single generalized strings."""

from collections import deque
import automata

# DNA Alphabet strings are made from.
ALPHABET = ['A', 'T', 'C', 'G']

def subset_construction(string):
    """Construct a DFA using subset construction algorithm.

    Args:
        string: The generalized string in the form of a nested list.
            e.g. [['A','C','G','T'],['A','G'],['G']]
        Returns: An Automata object representing the final DFA.
    """
    # Assume linear form of the NFA w/o construction. To do this just assume
    # that each index of the generalized string is the transition function
    # for the ith node.
    string_size = len(string)
    string.append([])

    # Set up automata and add start state.
    dfa = automata.Automata()
    start_state = tuple([0])
    dfa.add_state(start_state, is_start=True)

    # Add and compute transitions for all of the accessible states.
    to_evaluate = deque()
    to_evaluate.append(start_state)

    while len(to_evaluate) > 0:
        curr_state = to_evaluate.pop()
        # Compute the transtions for the current state.
        for letter in ALPHABET:
            to_states = [0] if 0 in curr_state else []
            # Find all states that can be transitioned to with this letter.
            for nfa_state in curr_state:
                if letter in string[nfa_state]:
                    to_states.append(nfa_state + 1)
            # Check if this state is already in the list of states.
            to_state_set = set(to_states)
            in_automata = False
            for existing in dfa.states:
                if to_state_set == set(existing):
                    in_automata = True
                    break
            to_state_key = tuple(to_states)
            if not in_automata:
                dfa.add_state(to_state_key,
                              is_terminal=(string_size in to_states))
                to_evaluate.append(to_state_key)
            # Add transition.
            dfa.add_transition(curr_state, to_state_key, letter)
    return dfa

def intersection_construction(string):
    """Construct a DFA using the intersection construction method.

    Args:
        string: The generalized string in the form of a nested list.
            e.g. [['A','C','G','T'],['A','G'],['G']]
        Returns: An Automata object representing the final DFA.
    """
    string_size = len(string)

    # Create a new DFA and addd the start state.
    dfa = automata.Automata()
    start_state = tuple([0])
    dfa.add_state(start_state, is_start=True)

    # Form the string bit-mapping that will be used for comparison.
    # e.g. {AC} -> 1010
    string_codes = []
    for letters in string:
        curr_code = 0
        for letter in letters:
            curr_code = curr_code | 1 << ALPHABET.index(letter)
        string_codes.append(curr_code)

    # Add transitions.
    to_compute = deque([start_state])
    while len(to_compute) > 0:
        curr_state = to_compute.pop()
        for letter_index, letter in enumerate(ALPHABET):
            # Build a string to compare to, if we are at state we already know
            # we have read in certain characters.
            compare_str = string_codes[:max(curr_state)]
            compare_str.append(0 | 1 << letter_index)
            if len(compare_str) > string_size:
                compare_str.pop(0)
            matched_lengths = _find_possible_transitions(curr_state,
                                                         compare_str,
                                                         string_codes)

            # Form a state using matched_lengths, add to DFA if not present.
            to_state = tuple(matched_lengths[::-1])
            if to_state not in dfa.states:
                dfa.add_state(to_state, is_terminal=(string_size in matched_lengths))
                to_compute.append(to_state)
            dfa.add_transition(curr_state, to_state, letter)
    return dfa

def _find_possible_transitions(curr_state, compare_str, string_codes):
    """Find the possible transitions for a state in iteration method.

    Args:
        curr_state: The current state that we are looking at represented as a
            tuple.
        compare_str: The generated code for the current state plus some letter.
        string_codes: The code for the language we are recognizing.
    Returns:
        A list of lengths of strings that could be transitioned to.
    """
    # Find all of the lengths of this compare string that match the
    # generalized string.
    matched_lengths = []
    for start_index in range(len(compare_str)):
        # Skip over this iteration if it would be impossible to get
        # to this string length.
        if len(compare_str) - start_index - 1 not in curr_state:
            continue
        matches = True
        # Check the current iteration of the string.
        for i in range(0, len(compare_str) - start_index):
            if compare_str[start_index + i] & string_codes[i] == 0:
                matches = False
                break
        if matches:
            matched_lengths.append(len(compare_str) - start_index)
    matched_lengths.append(0) # Since 0 always matches.
    return matched_lengths


if __name__ == '__main__':
    print subset_construction([['C', 'T'], ['A', 'G'], ['G', 'C']])
    print intersection_construction([['C', 'T'], ['A', 'G'], ['G', 'C']])
