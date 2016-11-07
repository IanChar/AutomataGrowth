"""Automata library module"""
import automata


def make_pam_machine():
    """Make a pam machine and print what it looks like."""
    pam_machine = automata.Automata()
    pam_machine.add_state(((1, 2), 1))
    pam_machine.add_state('NR')
    pam_machine.add_state('NRT')

    pam_machine.add_transition(((1, 2), 1), ((1, 2), 1), 'C')
    pam_machine.add_transition(((1, 2), 1), ((1, 2), 1), 'T')
    pam_machine.add_transition(((1, 2), 1), 'NR', 'A')
    pam_machine.add_transition(((1, 2), 1), 'NR', 'G')
    pam_machine.add_transition('NR', 'NR', 'A')
    pam_machine.add_transition('NR', ((1, 2), 1), 'T')
    pam_machine.add_transition('NR', ((1, 2), 1), 'C')
    pam_machine.add_transition('NR', 'NRT', 'G')
    pam_machine.add_transition('NRT', 'NR', 'A')
    pam_machine.add_transition('NRT', 'NRT', 'G')
    pam_machine.add_transition('NRT', ((1, 2), 1), 'C')
    pam_machine.add_transition('NRT', ((1, 2), 1), 'T')

    print pam_machine

if __name__ == '__main__':
    make_pam_machine()
