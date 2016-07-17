"""Module for unittesting."""
import unittest

import automata


class TestAutomata(unittest.TestCase):
    """Class to test the Automata class."""

    def setUp(self):
        self.automata = automata.Automata()

    def test_add_start_state(self):
        """Tests adding a start state to the automata."""
        self.automata.add_state('A', is_start=True)
        self.assertEqual(len(self.automata.states), 1)
        self.assertEqual(len(self.automata.start_states), 1)

    def test_add_terminal_state(self):
        """Tests adding a terminal state to the automata."""
        self.automata.add_state('A', is_terminal=True)
        self.assertEqual(len(self.automata.states), 1)
        self.assertEqual(len(self.automata.terminal_states), 1)

    def test_add_transition(self):
        """Tests the creation of transitions between states."""
        # Create two states and add a transition.
        self.automata.add_state('A', is_start=True)
        self.automata.add_state('B', is_terminal=True)
        self.automata.add_transition('A', 'B', 'l')
        self.assertEqual({'A': {'l': set(['B'])}}, self.automata.transitions)

        # Test creating another state and adding it.
        self.automata.add_state('C')
        self.automata.add_transition('A', 'C', 'l')
        self.assertEqual({'A': {'l': set(['B', 'C'])}},
                         self.automata.transitions)

if __name__ == '__main__':
    unittest.main()
