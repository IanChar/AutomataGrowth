"""
Utility functions for case where letters can have non-uniform probabilities
of appearing.
"""

from random import random, randint

def create_random_string(probs, string_size):
    """Creates a randomly generated string based on the alphabet distribution.
    Args:
        probs: List containing the probabilities of each letter occuring.
               It is not required that all of these probabilities add up to 1
               because a letter being added to a set is independent of another
               letter being added.
        string_size: The size of the string to be produced.
    Returns: List of sets.
    """
    alph = get_default_alphabet(len(probs))
    string = []
    for _ in xrange(string_size):
        curr_set = set()
        while len(curr_set) == 0:
            for letter_index, letter in enumerate(alph):
                if random() <= probs[letter_index]:
                    curr_set.add(letter)
        string.append(curr_set)
    return string

def get_default_alphabet(alph_size):
    """Gets the default alphabet (A, B, C, etc) of the requested size.
    Args:
        alph_size: The size of the alphabet to create.
    Returns: List of characters.
    """
    return [chr(ord('A') + a) for a in xrange(alph_size)]

def get_random_probs(alph_size=None, max_alph_size=5):
    """Gets a random list of probabilities (note not probability vector since
    probabilities need not sum to 1).
    Args:
        alph_size: The number of probabilities to be produced. If this is not
            included, this too will be random.
        max_alph_size: The maximum alphabet size to to have if the alphabet
            size should also be randomized. If alph_size is provided, this
            does not matter.
    Returns: A list of unrelated probabilities.
    """
    if alph_size is None:
        alph_size = randint(1, max_alph_size)
    probs = []
    for _ in xrange(alph_size):
        probs.append(random())
    return probs
