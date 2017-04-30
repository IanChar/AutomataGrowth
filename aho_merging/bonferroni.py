"""
Explore trying to find probability there is a substring matching with prefix
probability using Bonferroni inequalities.
"""
from __future__ import division
import operator as op

def get_intersection(a, k, c):
    """Get the intersection probability using total law of probability."""
    to_return = 0
    lambdas = [_get_lambda_const(i, a, c) for i in xrange(1, a + 1)]
    configs = _get_size_configs(a, k)
    for config in configs:
        term = mcr(k, list(config))
        for s, s_size in enumerate(config):
            term *= lambdas[s] ** s_size
        to_return += term
    to_return *= (1 / (2 ** a - 1)) ** k
    return to_return

def _get_lambda_const(size, a, c):
    """Get the size constant for the computation.
    Args:
        size: The size, used to compute the constant.
        a: The alphabet size in question.
        c: The number of c we are looking at.
    Returns: The constant in question.
    """
    to_return = (1 - (2 ** (a - size) - 1) / (2 ** a - 1)) ** c
    to_return *= ncr(a, size)
    return to_return

def _get_size_configs(a, k, curr = None, seen = None, solutions = None):
    """Get the number of configs (w/ DP) for prefix sizes.
    Args:
        a: The size of the alphabet.
        k: The size of the substring to match with. (sum of config is k)
        curr: The current configuration being worked on.
        seen: The partial configurations seen.
        solutions: The set of solutions.
    Returns:
        Set of all possible configurations as an ordered tuple. The first
        element of the tuple is how many sets have size 1, then 2, etc.
    """
    # If this is the first call init seen and solutions.
    if seen is None:
        seen = set()
    if solutions is None:
        solutions = set()
    # If there are no more marbles left to distribute.
    if k == 0:
        if curr is not None:
            solutions.add(curr)
    else:
        # If this is the first call init curr.
        if curr is None:
            curr = tuple(0 for _ in range(a))
        # Add a marble to each bucket and explore the tree if have not yet.
        for to_add in range(a):
            # Make a copy of the tuple with updated index.
            updated = tuple(curr[i] if i != to_add else curr[i] + 1
                            for i in range(a))
            if updated not in seen:
                seen.add(updated)
                _get_size_configs(a, k - 1, updated, seen, solutions)
    return solutions

# Taken from http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
def ncr(n, r):
    r = min(r, n-r)
    if r == 0:
        return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer / denom

def mcr(n, rs):
    """Multinomial coefficient computation.
    Args:
        n: Number of total.
        rs: The list of sizes of groups one divides into.
    Returns: The answer.
    """
    max_denom = max(rs)
    rs.remove(max_denom)
    if max_denom == n:
        return 1
    numer = reduce(op.mul, xrange(n, max_denom, -1))
    denom = 1
    for r in rs:
        if r > 1:
            denom *= reduce(op.mul, xrange(1, r + 1))
    return numer / denom

if __name__ == '__main__':
    print get_intersection(4, 3, 2)
