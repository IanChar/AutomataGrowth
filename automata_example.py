"""Automata library module"""
import automata


def make_pam_machine():
    """Make a pam machine and print what it looks like."""
    pam_machine = automata.Automata()
    pam_machine.add_state('N')
    pam_machine.add_state('NR')
    pam_machine.add_state('NRT')

    pam_machine.add_transition('N', 'N', 'C')
    pam_machine.add_transition('N', 'N', 'T')
    pam_machine.add_transition('N', 'NR', 'A')
    pam_machine.add_transition('N', 'NR', 'G')
    pam_machine.add_transition('NR', 'NR', 'A')
    pam_machine.add_transition('NR', 'N', 'T')
    pam_machine.add_transition('NR', 'N', 'C')
    pam_machine.add_transition('NR', 'NRT', 'G')
    pam_machine.add_transition('NRT', 'NR', 'A')
    pam_machine.add_transition('NRT', 'NRT', 'G')
    pam_machine.add_transition('NRT', 'N', 'C')
    pam_machine.add_transition('NRT', 'N', 'T')

    print pam_machine

if __name__ == '__main__':
    make_pam_machine()
