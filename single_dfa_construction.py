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

    # We assume that there will be string length + 1 states in the DFA.
    dfa = automata.Automata()
    dfa.add_state(0, is_start=True)
    for i in range(1, string_size):
        dfa.add_state(i)
    dfa.add_state(string_size, is_terminal=True)

    # Form the string bit-mapping that will be used for comparison.
    # e.g. {AC} -> 1010
    string_codes = []
    for letters in string:
        curr_code = 0
        for letter in letters:
            curr_code = curr_code | 1 << ALPHABET.index(letter)
        string_codes.append(curr_code)

    # Add transitions.
    for state in range(string_size + 1):
        for letter_index, letter in enumerate(ALPHABET):
            # Build a string to compare to, if we are at state we already know
            # we have read in certain characters.
            compare_str = string_codes[:state]
            compare_str.append(0 | 1 << letter_index)
            if len(compare_str) > string_size:
                compare_str.pop(0)
            largest_transition_found = False
            while not largest_transition_found:
                # Compare downwards since likely to see failure more quickly.
                matches = True
                for i in range(len(compare_str) - 1, -1, -1):
                    if compare_str[i] & string_codes[i] == 0:
                        matches = False
                        break
                if matches:
                    dfa.add_transition(state, len(compare_str), letter)
                    largest_transition_found = True
                else:
                    compare_str.pop(0)
    return dfa


if __name__ == '__main__':
    print intersection_construction([['A', 'C', 'G', 'T'], ['A', 'G'], ['G']])

