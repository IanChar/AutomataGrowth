"""Example given in Marschall's paper constructing a simple nfa from 3 strings.
"""

import construct_automata

if __name__ == '__main__':
    print construct_automata.construct_simple_nfa([
        [['B', 'C'], ['A', 'C'], ['A', 'B']],
        [['A'], ['B'], ['A', 'B', 'C']],
        [['C'], ['B', 'C'], ['A', 'C']]])
