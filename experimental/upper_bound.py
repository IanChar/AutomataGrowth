"""Experiment with different upper bounds for size of each depth."""
from __future__ import division
import sys

sys.path.append('../common')
import arbitrary_probs_util as string_util
from string_comparisons import prefix_substring_match

def compute_signature_bound(gen_string, depth):
    """Computes the upper bound for a depth, generalized string pair.
    Args:
        gen_string: The generalized string (list of sets).
        depth: The depth to compute for.
    Returns: The upper bound for this depth.
    """
    if depth == 0 or depth == 1:
        return 1
    matches = [prefix_substring_match(gen_string, depth - j, j)
               for j in xrange(1, depth)]
    bound = 0
    for ind in xrange(len(matches)):
        if matches[ind]:
            partial_bound = 1
            for lower_ind in xrange(ind):
                if matches[lower_ind]:
                    partial_bound *= 2
            bound += partial_bound
    return bound

def compute_all_signature_bounds(gen_string):
    """Computes upper signature upper bound for every depth.
    Args:
        gen_string: The generalized string to analyze.
    Returns: A list of the upper bounds.
    """
    bounds = [1, 1]
    for depth in xrange(2, len(gen_string) + 1):
        bounds.append(compute_signature_bound(gen_string, depth))
    return bounds

def average_signature_bound(probs, num_samples, length):
    """Compute average upper bound.
    Args:
        probs: Probabilities of seeing each letter.
        num_samples: Number of samples to use.
        length: The length of the generalized strings to look at.
    Returns: List of upper bounds for each depth.
    """
    avg_bounds = [0 for _ in xrange(length + 1)]
    for _ in xrange(num_samples):
        gen_string = string_util.create_random_string(probs, length)
        for ind in xrange(len(avg_bounds)):
            avg_bounds[ind] += compute_signature_bound(gen_string, ind)
    return [b / num_samples for b in avg_bounds]
