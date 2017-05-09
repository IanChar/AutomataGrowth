"""Estimate probabilities of matching substrings with overlaps"""
from __future__ import division
import operator as op
import bonferroni
import aho_test

def get_single_match(a, k, overlap):
    """Probability of a single substring match (A_i) with overlap w/ prefix.
    Args:
        a: The size of the alphabet.
        k: The size of the matching substring.
        overlap: The amount of overlap in the substring.
    """
    if overlap > k / 2 or overlap < 1:
        raise NotImplementedError('Not found yet for this case')
    to_return = 0
    num_possibilities = 2 ** a - 1
    for j in range(1, a + 1):
        sum_term = (1 - (2 ** (a - j) - 1) / num_possibilities) ** 2
        sum_term *= ncr(a, j)
        to_return += sum_term
    to_return /= num_possibilities
    to_return = to_return ** overlap
    if overlap < k / 2:
        to_return *= bonferroni.get_single_match(a, k - 2 * overlap)
    return to_return

def sim_single_match(a, k, overlap, trials):
    """Simulate prob of single match with overlap.
    Args:
        a: The alphabet size.
        k: The size of the matching substring.
        overlap: The amount of overlap in the substring with prefix.
        trials: Number of trials to run for estimate.
    """
    alph = [chr(ord('A') + i) for i in range(a)]
    num_match = 0
    for _ in xrange(trials):
        prefix = aho_test.build_random_string(k, alph)
        substring = prefix[-overlap:]
        substring += aho_test.build_random_string(k - overlap, alph)
        has_match = True
        for index in range(k):
            if not len(set(prefix[index]).intersection(set(substring[index]))):
                has_match = False
                break
        if has_match:
            num_match += 1
    return num_match / trials


# Taken from http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
def ncr(n, r):
    r = min(r, n-r)
    if r == 0:
        return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer / denom

if __name__ == '__main__':
    print get_single_match(4, 7, 3)
    print sim_single_match(4, 7, 3, 100)
