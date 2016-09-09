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

if __name__ == '__main__':
    print subset_construction([['A', 'C', 'G', 'T'], ['A', 'G'], ['G']])

