""" Implementation of merging algorithm as presented in the paper. """

from collections import deque

class State(object):

    def __init__(self, sid, name='', depth=None):
        self.sid = sid
        self.name = name
        self.goto = {}
        self.failure = None
        self.depth = depth

    def set_goto(self, letter, state):
        self.goto[letter] = state

    def set_failure(self, failure):
        self.failure = failure

    def transition(self, letter):
        if letter in self.goto.keys():
            return self.goto[letter]
        else:
            return self.failure.transition(letter)

    def __str__(self):
        to_print = []
        to_print.append('-------------------------------------')
        to_print.append('State ID: %d' % self.sid)
        to_print.append('Depth: %d' % self.depth)
        if self.name != '':
            to_print.append('State Name: %s' % self.name)
        to_print.append('Goto Values:')
        for letter, state in self.goto.iteritems():
            to_print.append('\t%s => (%d, %s)' % (letter, state.sid,
                                                  state.name))
        if self.failure is not None:
            to_print.append('Failure: (%d, %s)' % (self.failure.sid,
                                               self.failure.name))
        to_print.append('-------------------------------------\n')
        return '\n'.join(to_print)

class DFA(object):

    def __init__(self, root, num_states):
        self.root = root
        self.num_states = num_states

    def get_root(self):
        return self.root

    def get_num_states(self):
        return self.num_states

    def print_automaton(self):
        queue = deque()
        queue.appendleft(self.root)
        seen = [self.root]
        while len(queue) > 0:
            curr = queue.pop()
            print curr
            for child in curr.goto.values():
                if child not in seen:
                    queue.appendleft(child)
                    seen.append(child)

    def __str__(self):
        to_print = []
        to_print.append('Number of states: %d' % self.num_states)
        to_print.append(str(self.root))
        return '\n'.join(to_print)

def aho_merge(g, alphabet):
    sid_counter = 0
    # Initialize start state and child.
    start_state = State(sid_counter, 'Epsilon', 0)
    sid_counter += 1
    start_child = State(sid_counter, depth=1)
    start_child.set_failure(start_state)
    sid_counter += 1
    for letter in alphabet:
        if letter in g[0]:
            start_state.set_goto(letter, start_child)
        else:
            start_state.set_goto(letter, start_state)

    next_depth = deque()
    next_depth.append(start_child)
    for depth in range(len(g)):
        # Init empty hash table
        failMap = {}
        # For state in this depth.
        for curr_state in next_depth:
            # Mark accepting?a
            if depth == len(g) - 1:
                curr_state.name = 'Accepting'
            else:
                # For letter in this depth
                for letter in g[depth + 1]:
                    # Find failure, add to hash table if needed, link to parent.
                    failure = find_failure(curr_state, letter)
                    if failure not in failMap.keys():
                        child = State(sid_counter, depth=curr_state.depth + 1)
                        sid_counter += 1
                        child.set_failure(failure)
                        failMap[failure] = child
                    curr_state.set_goto(letter, failMap[failure])
        next_depth = deque(failMap.values())
    return DFA(start_state, sid_counter)

def find_failure(parent, letter):
    failure = None
    curr_state = parent
    while curr_state.failure is not None:
        curr_state = curr_state.failure
        if letter in curr_state.goto.keys():
            failure = curr_state.goto[letter]
            break
    if failure is None:
        failure = curr_state
    return failure


if __name__ == '__main__':
    dfa = aho_merge([['A', 'B'], ['C', 'B', 'D'], ['A'], ['C', 'B', 'D']], ['A', 'B', 'C', 'D'])
    dfa.print_automaton()
