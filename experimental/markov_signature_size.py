"""Markov model for signature size."""

import numpy as np
from scipy.stats import binom

def get_ptm(c_2, n):
    """Get the probably transition matrix.
    Args:
        c_2: The probability of having overlap.
        n: The size of the ptm.
    Returns: np ptm matrix.
    """
    ptm = np.zeros((n, n))
    for row in xrange(n - 1):
        ptm[row, :row + 2] = binom.pmf(range(row + 2), row + 1, c_2)
    last_row = binom.pmf(range(n + 1), n + 1, c_2)
    last_row[-2] += last_row[-1]
    ptm[-1, :] = last_row[:-1]
    return ptm

def lower_bound(c_2, n):
    """Get the lower bound of expected size at each depth of the tree.
    Args:
        c_2: The probability of having overlap.
        n: The depth to go to in the tree.
    Returns: Array of the expected sizes where ith index is number of size
             i + 2.
    """
    ptm = get_ptm(c_2, n)
    curr = np.zeros((1, n))
    curr[0, 0] = 1
    total = np.zeros((1, n))
    total += curr
    for _ in xrange(n - 1):
        curr = np.dot(curr, ptm)
        total += curr
    return total

if __name__ == '__main__':
    P = 0.25
    V = expected_sizes(P, 20)
    print V / np.sum(V)
