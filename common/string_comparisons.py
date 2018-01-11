"""Utility functions to compare generalized strings."""

def do_substrings_match(substring1, substring2):
    """Check if substring1 == substring2
    Args:
        Substrings should be put in as a list of sets.
    Returns: True or False.
    """
    if len(substring1) != len(substring2):
        return False
    for letter1, letter2 in zip(substring1, substring2):
        if len(letter1.intersection(letter2)) == 0:
            return False
    return True

def prefix_substring_match(gen_string, sub_start, sub_len):
    """Checks if G[1:k] == G[start:k]
    Args:
        gen_string: The generalized string.
        sub_start: The start of the substring to check (starts at 0).
        sub_len: The length of the substring to check.
    Returns: True or false.
    """
    if sub_start + sub_len > len(gen_string):
        return False
    return do_substrings_match(gen_string[:sub_len],
                               gen_string[sub_start:sub_start + sub_len])

if __name__ == '__main__':
    G1 = [set(['a']), set(['b']), set(['a'])]
    print prefix_substring_match(G1, 2, 1)
