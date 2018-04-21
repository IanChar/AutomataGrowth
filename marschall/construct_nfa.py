"""Reads in strings stored in a .txt file and constructs and nfa.

The file containing the strings should be formatted like the following:

{B,C} {A,C} {A,B}
{A} {B} {A,B,C}

In particular, each line represents a generalized string and each {} contains
possible letters for that position in the string. Note that each of the sets
of curly braces are separated by one space, but there are no spaces inside
of the curly braces.
"""

import sys

import automata_algorithms

def _read_in_strings(filename):
    """Read in the generalized strings from a file.

    Args:
        filename: The path to the file containing the strings.
    Returns:
        A list of generalized strings formatted as a nested list.
        e.g. "{A, B, C},{E}" -> [["A", "B", "C"], [E]]
    """
    # Read in the strings from the file.
    reader = open(filename, 'r')
    try:
        to_parse = [line for line in reader]
    finally:
        reader.close()
    # Remove new line characters.
    to_parse = [line[:-1] for line in to_parse]

    # Parse into usable data structure.
    strings = []
    for file_line in to_parse:
        letters = file_line.split(' ')
        letters = [l.split(',') for l in letters]
        # Remove { } characters.
        for string_piece in letters:
            if len(string_piece) == 1:
                string_piece[0] = string_piece[0][1:-1]
            else:
                string_piece[0] = string_piece[0][1:]
                string_piece[-1] = string_piece[-1][:-1]
        strings.append(letters)

    return strings

def main():
    """Read in the strings and print out the formed nfa."""
    filename = sys.argv[1]
    strings = _read_in_strings(filename)
    print automata_algorithms.construct_simple_nfa(strings)


if __name__ == '__main__':
    main()
