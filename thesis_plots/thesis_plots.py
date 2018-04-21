# %% Imports
from __future__ import division
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append('./aho_merging')
sys.path.append('./common')
import arbitrary_probs_util as util
from merge_alg import aho_merge
from sampler import DepthSampler
from sampling_misc import sample_total_states

# %% Make plots comparing 2^n bound with reality.

def compare_2n_bound(alph_size, max_depth, sims):
    alph = util.get_default_alphabet(alph_size)
    probs = [0.5 for _ in range(alph_size)]
    depths = range(max_depth + 1)

    # Make upper bound.
    upper = [1]
    for i in depths[1:]:
        upper.append(2 ** i)

    # Get average trend.
    avgs = [1, 1]
    highest = [1, 1]
    stds = [0, 0]
    for d in depths[2:]:
        a, h, s = sample_total_states(sims, probs, d)
        avgs.append(a)
        highest.append(h)
        stds.append(s)

    plt.plot(depths, upper, ':', alpha=0.75, label='Upper Bound')
    plt.plot(depths, highest, '--', alpha=0.75, label='Max Found')
    plt.errorbar(depths, avgs, alpha=0.75, yerr=stds, capsize=3, label='Average')
    plt.xlabel('Generalized String Length')
    plt.ylabel('Number of States')
    plt.legend(bbox_to_anchor=[0.35, 1])
    # plt.show()

# plt.figure(1, figsize=(16, 6))
# plt.subplot(121)
compare_2n_bound(4, 15, 1000)
# plt.subplot(122)
# compare_2n_bound(8, 13, 10000)
plt.savefig('figs/deterministic_upper.png')
plt.show()

# %% Plots to show c_2 and c_3
plt.rc('text', usetex=True)
c_2 = lambda a: (4 ** a - 3 ** a) / (2 ** a - 1) ** 2
c_3 = lambda a: (8 ** a - 2 * 6 ** a + 5 ** a) / (2 ** a - 1) ** 3

alphs = range(2, 20)
plt.plot(alphs, [c_2(a) for a in alphs], 'o', label='$c_2$')
plt.plot(alphs, [c_3(a) for a in alphs], 'o', label='$c_3$')
plt.xticks(range(0, 21, 5))
plt.legend(bbox_to_anchor=[1,0.2])
# plt.title('Probability vs. Alphabet Size')
plt.xlabel('Alphabet Size')
plt.ylabel('Probability')
plt.savefig('figs/prob_constants.png')
plt.show()

# %% log_{1/c_2}(2) vs alphabet

log2 = np.log(2)
c_2 = lambda a: (4 ** a - 3 ** a) / (2 ** a - 1) ** 2
exponent = lambda a: 1 + log2 / np.log(1 / c_2(a))
alphs = range(2, 20)

plt.plot(alphs, [exponent(a) for a in alphs])
# plt.title('Exponent vs. Alphabet Size')
plt.xlabel('Alphabet Size')
plt.ylabel('Exponent Value')
plt.xticks(range(0, 21, 5))
plt.savefig('figs/exponent_size.png')
plt.show()

# %% Make plots demonstrating how memory consumption becomes linear at some point.

def states_per_depth(alph_size, max_depth, num_samples):
    alph = util.get_default_alphabet(alph_size)
    probs = [0.5 for _ in range(alph_size)]
    ds = DepthSampler(probs, max_depth)
    df = ds.draw_samples(num_samples, ['states'])
    avgs = df.groupby('depth')['states'].mean().tolist()

    one_real = ds.draw_samples(1, ['states']).groupby('depth')['states'].mean().tolist()

    c_2 = (4 ** alph_size - 3 ** alph_size) / (2 ** alph_size - 1) ** 2
    k_n = np.ceil(np.log(max_depth) / np.log(1 / c_2))

    plt.plot(range(max_depth + 1), avgs, label='Average')
    plt.plot(range(max_depth + 1), one_real, '--', label='Single Realization')
    plt.axvline(k_n, ls=':', label='k(n)')
    plt.legend(bbox_to_anchor=[0.33, 1])
    plt.title('Alphabet Size %d' % alph_size)
    plt.xlabel('Depth')
    plt.ylabel('Number of States in Depth')

plt.figure(2, figsize=(16, 6))
plt.subplot(121)
states_per_depth(4, 30, 10000)
plt.subplot(122)
states_per_depth(8, 50, 10000)
plt.savefig('figs/lingrowth2_alph4_8_sims1000.png')
plt.show()

# %% Generate data for the high prob bound.
alph_size = 3
probs = [0.5 for _ in range(alph_size)]
depths = range(20 + 1)

avgs = [1, 1]
highest = [1, 1]
stds = [0, 0]
for d in depths[2:]:
    a, h, s = sample_total_states(10000, probs, d)
    avgs.append(a)
    highest.append(h)
    stds.append(s)

# %% Total state high probability bounds.

c_2 = lambda a: (4 ** a - 3 ** a) / (2 ** a - 1) ** 2

# Make upper bound.
prob = c_2(alph_size)
upper = [1]
for i in depths[1:]:
    upper.append(i ** (1 + np.log(2) / np.log(1 / prob)))
# make exp upper bound
exp_upper = [2 ** i for i in depths]
print (1 + np.log(2) / np.log(1 / prob))

plt.plot(depths, exp_upper, ':', label='Deterministic Bound')
plt.plot(depths, upper, ':', alpha=0.75, label='Polynomial Bound')
plt.plot(depths, highest, '--', alpha=0.75, label='Max Found')
plt.errorbar(depths, avgs, alpha=0.75, yerr=stds, capsize=3, label='Average')
plt.xlabel('Generalized String Length')
plt.ylabel('Number of States')
plt.ylim([-1000, 35000])
# plt.title('Number of States vs Generalized String Length')
plt.legend(bbox_to_anchor=[0.45, 1])
# plt.show()

plt.savefig('figs/high_prob_bound4.png')
plt.show()

# %% Show that there is linear growth
c_2 = lambda a: (4 ** a - 3 ** a) / (2 ** a - 1) ** 2
def realization_growth(alph_size, max_depth, num_samples):
    kn = lambda a: np.log(max_depth) / np.log(1 / c_2(a))
    alph = util.get_default_alphabet(alph_size)
    probs = [0.5 for _ in range(alph_size)]
    ds = DepthSampler(probs, max_depth)
    depths = range(max_depth + 1)
    for _ in xrange(num_samples):
        vals = ds.draw_samples(1, ['states']).groupby('depth')['states'].mean().tolist()
        acum = [vals[0]]
        for i in range(1, len(vals)):
            acum.append(vals[i] + acum[i - 1])
        plt.plot(depths, acum, alpha=0.75)

    plt.axvline(kn(alph_size), linestyle='--', label='$k(n)$')
    plt.legend(bbox_to_anchor=[0.2, 1])
    plt.title('Alphabet Size %d' % alph_size)
    plt.xlabel('Depth')
    plt.ylabel('Total Number of States Added')

plt.figure(2, figsize=(16, 6))
plt.subplot(121)
realization_growth(4, 100, 3)
plt.subplot(122)
realization_growth(8, 100, 3)
plt.savefig('figs/fixed_total_states.png')
plt.show()
