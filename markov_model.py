""" Simulation of the growth of the merged automata using Markov Model """
from __future__ import division
import operator as op
import math
import random
import numpy as np
from matplotlib import pyplot as plt

def autoamta_sim(word_length, alphabet_size):
    """Simulate the automata growth and return size at each level"""
    curr_sizes = [1, 2]
    for _ in range(word_length - 2):
        curr_sizes.append(curr_sizes[-1] + k_sim(curr_sizes[-1],
            alphabet_size * (curr_sizes[-1] - curr_sizes[-2])))
    return curr_sizes

def k_sim(bins, marbles):
    simulated = np.random.multinomial(marbles, [1/float(bins)]*bins)
    num_full = 0
    for urn in simulated:
        if urn > 0:
            num_full += 1
    return num_full

def make_hist(data):
    _, _, _ = plt.hist(data, 75, facecolor='blue', alpha=0.5)
    plt.xlabel('Failure Calculations')
    title = 'Histogram of Failure Calculations for Model'
    plt.title(title)
    plt.grid(True)
    plt.show()

def failure_trials(trials):
    data = []
    for _ in range(trials):
        data.append(autoamta_sim(8, 4)[-2] * 4)
    return data

if __name__ == '__main__':
    make_hist(failure_trials(1000))
