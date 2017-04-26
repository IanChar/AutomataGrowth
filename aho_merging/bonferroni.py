"""
Explore trying to find probability there is a substring matching with prefix
probability using Bonferroni inequalities.
"""

def get_intersection():
    """Get the intersection probability using total law of probability."""
    pass

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

