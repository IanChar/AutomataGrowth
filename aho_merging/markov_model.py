""" Simulation of the growth of the merged automata using Markov Model """
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

def autoamta_sim(word_length, marble_cap):
    """Simulate the automata growth and return size at each level"""
    curr_sizes = [1, 2]
    marbles = []
    for _ in range(word_length):
        new_marble = 0
        while new_marble == 0:
            new_marble = np.random.binomial(marble_cap, 0.5, 1)[0]
        marbles.append(new_marble)
    for i in range(word_length - 2):
        curr_sizes.append(curr_sizes[-1] + k_sim(curr_sizes[-1],
            marbles[i + 2] * (curr_sizes[-1] - curr_sizes[-2])))
    return curr_sizes, marbles

def k_sim(bins, marbles):
    simulated = np.random.multinomial(marbles, [1 / float(bins)] * bins)
    num_full = 0
    for urn in simulated:
        if urn > 0:
            num_full += 1
    return num_full

def make_hist(data, string_length, alphabet_size):
    """Make a histogram from the data."""
    _, _, _ = plt.hist(data, 50, facecolor='blue', alpha=0.5)
    plt.xlabel('Failure Calculations')
    title = ('Failure Calculations in Model, Trials: %d, Word Length: %d, Alphabet Size: %d'
            % (len(data), string_length, alphabet_size))
    plt.title(title)
    plt.grid(True)
    plt.show()

def failure_trials(trials, string_length, alphabet_size):
    """Count the number of failures calculated for each trial."""
    data = []
    for _ in range(trials):
        level_sizes, marbles = autoamta_sim(string_length, alphabet_size)
        data.append(level_sizes[-2] * marbles[-2])
    return data

if __name__ == '__main__':
    for word_length in range(5, 10):
        make_hist(failure_trials(10000, word_length, 4), word_length, 4)
