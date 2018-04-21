"""Find different probabilities for arbitrary letter probabilities."""

from __future__ import division

import arbitrary_probs_util as util

def sim_c2(probs, trials):
    """Simulate c_2 (probability two random sets "match" each other) for
    arbitrary probabilities.
    Args:
        probs: A list of the probabilities for each letter.
        trials: The number of trials to perform.
    Returns: Simulated probability.
    """
    total = 0
    for _ in xrange(trials):
        set1 = util.create_random_string(probs, 1)[0]
        set2 = util.create_random_string(probs, 1)[0]
        if len(set1.intersection(set2)) > 0:
            total += 1
    return total / trials

def theoretical_c2(probs):
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

def sim_c12(probs, trials):
    """Simulate c_12 (probability one set "mathces" with two others) for
    arbitrary probabilities.
    Args:
        probs: A list of the probabilities for each letter.
        trials: The number of trials to perform.
    Returns: Simulated probability.
    """
    total = 0
    for _ in xrange(trials):
        pref = util.create_random_string(probs, 1)[0]
        suff1 = util.create_random_string(probs, 1)[0]
        suff2 = util.create_random_string(probs, 1)[0]
        if (len(pref.intersection(suff1)) > 0
                and len(pref.intersection(suff2)) > 0):
            total += 1
    return total / trials

def theoretical_c12(probs):
    """Calculate the theoretical c_12 for arbitrary probabilities.
    Args:
        probs: A list of the probabilities for each letter.
    Returns: The calculated probability.
    """
    result = 0
    alph = util.get_default_alphabet(len(probs))
    normalizing = 1
    for prob in probs:
        normalizing *= (1 - prob)
    normalizing = 1 - normalizing
    for pref in _get_all_prefixes(alph):
        p_prob = 1
        c_prob = 1
        for letter_index, letter in enumerate(alph):
            if letter in pref:
                p_prob *= probs[letter_index]
                c_prob *= (1 - probs[letter_index])
            else:
                p_prob *= (1 - probs[letter_index])
        result += ((p_prob / normalizing)
                   * ((1 - c_prob) / normalizing) ** 2)
    return result

def _get_all_prefixes(alph, curr=None, results=None):
    """Compiles a list of all possible prefixes for the alphabet.
    Args:
        alph: The alphabet in question.
    Returns: A list of sets.
    """
    alph = list(alph)
    if curr is None:
        curr = set()
        results = []
    if len(alph) == 0:
        if len(curr) > 0:
            results.append(curr)
        return results
    curr_letter = alph.pop()
    _get_all_prefixes(alph, set(curr), results)
    curr.add(curr_letter)
    _get_all_prefixes(alph, curr, results)
    return results
