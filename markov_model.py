""" Simulation of the growth of the merged automata using Markov Model """
from __future__ import division
import operator as op
import math
from random import random
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
    """Simulate a value of k given the specifications. """
    probs = [k_pdf(bins, marbles, k) for k in range(1, min(bins, marbles) + 1)]
    curr, curr_prob = 1, probs[0]
    rand_num = random()
    while rand_num > curr_prob:
        curr += 1
        try:
            curr_prob += probs[curr - 1]
        except:
            print "PROBLEM", curr, probs, rand_num
            curr = len(probs)
            rand_num = -1
    return curr


def k_pdf(bins, marbles, selected):
    """Calculate the probability K takes thes selected value."""
    stir_num = float(0)
    for j in range(selected + 1):
        stir_num += (-1) ** (selected - j) * ncr(selected, j) * j ** marbles
    stir_num /= math.factorial(selected)

    return (float(ncr(bins, selected) * stir_num * math.factorial(selected))
            / (bins ** marbles))

def ncr(n, r):
    """ Compute choose (from stack overflow) """
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return float(numer//denom)

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
    make_hist(failure_trials(10000))
