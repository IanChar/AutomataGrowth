"""Implementation of algorithm found in Marschall's paper."""

class DfaCreator(object):
    """Class to read in/parse generalized strings and make a DFA."""

    def __init__(self, filename):
        # Read in the strings from a file and set the generalized strings.
        file_contents = None
        self.strings = self._parse_strings(file_contents)

    def create_nfa(self):
        """Create an NFA from the generalized strings."""
        pass

    def _parse_strings(self, to_parse):
        pass

    def subset_construction(self):
        """Create a DFA from an NFA using subset construction."""
        pass

def main():
    pass

if __name__ == '__main__':
    pass
