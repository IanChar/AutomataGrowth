"""Common probabilities to be used for analysis."""
from __future__ import division

def get_c2(alph_size):
    """Gets the probability of two independenct, non-empty, uniformly random
    subsets of the alphabet sharing at least one letter in common.
    Args:
        alph_size: The size of the alphabet in question.
    Returns: The desired probability.
    """
    return (4 ** alph_size - 3 ** alph_size) / ((2 ** alph_size - 1) ** 2)

def get_c12(alph_size):
    """Gets the probability that one of the non-empty subsets matches with two
    other non-empty subsets. Each chosen independently and uniformly at random.
    Args:
        alph_size: The size of the alphabet in question.
    Returngs: The desired probability.
    """
    to_return = (8 ** alph_size - 2 * 6 ** alph_size + 5 ** alph_size)
    return to_return / ((2 ** alph_size - 1) ** 3)
