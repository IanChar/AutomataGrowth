"""Estimate probabilities of matching substrings with overlaps"""
from __future__ import division
import operator as op
import bonferroni
import aho_test

def get_single_match(alph_size, k, overlap):
    """Probability of a single substring match (A_i) with overlap w/ prefix.
    Args:
        alph_size: The size of the alphabet.
        k: The size of the matching substring.
        overlap: The amount of overlap in the substring.
    """
    if overlap > k / 2 or overlap < 1:
        raise NotImplementedError('Not found yet for this case')
    to_return = 0
    num_possibilities = 2 ** alph_size - 1
    for j in range(1, alph_size + 1):
        sum_term = (1 - (2 ** (alph_size - j) - 1) / num_possibilities) ** 2
        sum_term *= ncr(alph_size, j)
        to_return += sum_term
    to_return /= num_possibilities
    to_return = to_return ** overlap
    if overlap < k / 2:
        to_return *= bonferroni.get_single_match(alph_size, k - 2 * overlap)
    return to_return

def sim_single_match(alph_size, k, overlap, trials):
    """Simulate prob of single match with overlap.
    Args:
        alph_size: The alphabet size.
        k: The size of the matching substring.
        overlap: The amount of overlap in the substring with prefix.
        trials: Number of trials to run for estimate.
    """
    alph = _get_alphabet(alph_size)
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

def get_half_overlap(gamma, alph_size):
    """Probability of two substrings match with prefix when the substrings
    overlap by exactly half.
    Args:
        gamma: The amount of overlap between the substrings (gamma = k/2)
        alph_size: Size of alphabet.
    Returns:
        Theoretical probability see June 5, 2017 notebook.
    """
    alph = _get_alphabet(alph_size)
    power_set = _get_all_sets(alph)

    two_a = 2 ** alph_size
    total = 0
    for set1 in power_set:
        for set2 in power_set:
            set1_size, set2_size = len(set1), len(set2)
            union_size = len(set1.union(set2))
            two_1 = 2 ** (-1 * set1_size)
            two_2 = 2 ** (-1 * set2_size)
            to_add = 1 - two_1
            to_add *= 1 - two_2
            to_add *= (1 - two_1 - two_2 + 2 ** (-1 * union_size))
            total += to_add
    return (total / ((two_a - 1) ** 5) * two_a ** 3) ** gamma

def sim_half_overlap(gamma, alph_size, trials):
    """Simulated robability of two substrings match with prefix when the
    substrings overlap by exactly half.
    Args:
        gamma: The amount of overlap between the substrings (gamma = k/2)
        alph_size: Size of alphabet.
        trials: The number of trials to be performed.
    Returns:
        Simulated probability see June 5, 2017 notebook.
    """
    alph = _get_alphabet(alph_size)
    total = 0
    for _ in xrange(trials):
        has_match = True
        prefix_1 = aho_test.build_random_string(gamma, alph)
        prefix_2 = aho_test.build_random_string(gamma, alph)
        sub_1 = aho_test.build_random_string(gamma, alph)
        sub_2 = aho_test.build_random_string(gamma, alph)
        sub_3 = aho_test.build_random_string(gamma, alph)
        for index in xrange(gamma):
            if len(set(prefix_1[index]).intersection(set(sub_1[index]))) == 0:
                has_match = False
                break
            if len(set(prefix_1[index]).intersection(set(sub_2[index]))) == 0:
                has_match = False
                break
            if len(set(prefix_2[index]).intersection(set(sub_2[index]))) == 0:
                has_match = False
                break
            if len(set(prefix_2[index]).intersection(set(sub_3[index]))) == 0:
                has_match = False
                break
        if has_match:
            total += 1
    return total / trials

# Taken from http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
def ncr(num, choose):
    """Computes bionmial coefficient"""
    choose = min(choose, num - choose)
    if choose == 0:
        return 1
    numer = reduce(op.mul, xrange(num, num - choose, - 1))
    denom = reduce(op.mul, xrange(1, choose + 1))
    return numer / denom

def _get_all_sets(alph, curr=None, solns=None):
    """Gets all possible letter sets.
    Args:

        The alphabet to work with.
    Returns:
        List of sets.
    """
    if curr is None or solns is None:
        curr = []
        solns = []
    if len(alph) == 0:
        if len(curr) > 0:
            solns.append(set(curr))
    else:
        letter = alph.pop(0)
        added = list(curr)
        added.append(letter)
        _get_all_sets(list(alph), added, solns)
        _get_all_sets(alph, curr, solns)
    return solns

def _get_alphabet(alph_size):
    """Returns alphabet with the desired size.
    Args:
        alph_size: The size of the alphabet to return.
    Returns:
        The alphabet as a list of strings.
    """
    return [chr(ord('A') + i) for i in range(alph_size)]

if __name__ == '__main__':
    print get_half_overlap(1, 4)
    print sim_half_overlap(1, 4, 1000)
