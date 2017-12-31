"""Common probabilities to be used for analysis."""
from __future__ import division
from scipy.special import binom as ncr

def get_c2(alph_size):
    """Gets the probability of two independenct, non-empty, uniformly random
    subsets of the alphabet sharing at least one letter in common.
    Args:
        alph_size: The size of the alphabet in question.
    Returns: The desired probability.
    """
    return (4 ** alph_size - 3 ** alph_size) / ((2 ** alph_size - 1) ** 2)

def get_arbitrary_c2(probs):
    """Get the theoretical value of c_2 for arbitrary probabilities.
    Args:
        probs: A list of the probabilities for each letter.
    Returns: The calculated probability.
    """
    numer = 1
    all_comps = 1
    for prob in probs:
        all_comps *= (1 - prob)
        numer *= 1 - prob ** 2
    return (1 - numer) / ((1 - all_comps) ** 2)

def get_c12(alph_size):
    """Gets the probability that one of the non-empty subsets matches with two
    other non-empty subsets. Each chosen independently and uniformly at random.
    Args:
        alph_size: The size of the alphabet in question.
    Returngs: The desired probability.
    """
    to_return = (8 ** alph_size - 2 * 6 ** alph_size + 5 ** alph_size)
    return to_return / ((2 ** alph_size - 1) ** 3)

def uniform_urn_dist(urn, balls):
    """Gets the probability distribution of the number of non-empty urns after
    throwing balls into the urns.
    Args:
        urn: The number of urns.
        balls: The number of balls to be throwing.
    Returns: List of probabilities in the distribution summing up to 1.
    """
    dists = []
    prob = (1 / urn) ** balls
    for k in xrange(urn + 1):
        dists.append(ncr(urn, k) * stirling_numerator(balls, k) * prob)
    return dists

def stirling_numerator(n, k):
    """Gets the stirling numerator.
    Args:
        Number of way to partition an n element set into k non-empty subsets.
    Returns: Numerator (w/o factorial) of the stirling number of the 2nd kind.
    """
    total = 0
    for j in xrange(k + 1):
        total += (-1) ** (k - j) * ncr(k, j) * j ** n
    return total
