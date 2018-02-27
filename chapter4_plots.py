"""
Plots for chapter four of the thesis.
"""
# %% Imports
from __future__ import division
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.append('./aho_merging')
sys.path.append('./common')
import arbitrary_probs_util as util
from merge_alg import aho_merge
from sampler import DepthSampler
from sampling_misc import sample_total_states

# %% Constants

c_2 = lambda a: (4 ** a - 3 ** a) / (2 ** a - 1) ** 2

"""
Mu_x and mu_y convergence plots.

"""


# %% Get data for mu_x, mu_y plots
SAMPLES = 10000
DEPTH_MAX = 50
ds = DepthSampler([0.5 for _ in range(4)], DEPTH_MAX)
df = ds.draw_samples(SAMPLES, ['threads', 'new_thread', 'thread_children'])
df['thread_children'] = (df['thread_children']
                         .map(lambda l: 0 if len(l) == 0 else sum(l) / len(l)))
depth_group = df.groupby('depth')
muxs = depth_group['new_thread'].mean().tolist()
stdxs = depth_group['new_thread'].std().tolist()
muys = depth_group['thread_children'].mean().tolist()
stdys = depth_group['thread_children'].std().tolist()

# %% Plot averages of new particle and offspring
depths = range(DEPTH_MAX + 1)

plt.figure(1, figsize=(18, 6))

plt.subplot(121)
plt.plot(depths, muxs)
# plt.errorbar(depths, muxs, yerr=stdxs, capsize=3)
plt.xlabel('Depth')
plt.ylabel('Average New Particles')

plt.subplot(122)
# plt.errorbar(depths, muys, yerr=stdys, capsize=3)
plt.plot(depths, muys)
plt.xlabel('Depth')
plt.ylabel('Average Offspring')
plt.savefig('figs/avg_new_n_offspring.png')
plt.show()

# %% Plot errors in mu_x & E[X_n] and mu_y & E[Y_n]
plt.rc('text', usetex=False)
depths = range(DEPTH_MAX + 1)
fin_mux, fin_muy = muxs[-1], muys[-2]
xdiffs = [abs(x - fin_mux) for x in muxs]
ydiffs = [abs(y - fin_muy) for y in muys]

prob = c_2(4)
ref_func = lambda t: prob ** t
ref_vals = [ref_func(d) for d in depths]

plt.figure(1, figsize=(18, 6))

plt.subplot(121)
plt.plot(depths, xdiffs, label='Absolute Difference')
plt.plot(depths, ref_vals, '--', label='$c_2^n$')
plt.legend(bbox_to_anchor=[1, 1])
plt.xlabel('Depth')
plt.ylabel('Absolute Difference')
plt.title('Average New Particles')

plt.subplot(122)
plt.plot(depths[:-1], ydiffs[:-1], label='Absolute Difference')
plt.plot(depths[:-1], ref_vals[:-1], '--', label='$c_2^n$')
plt.legend(bbox_to_anchor=[1, 1])
plt.xlabel('Depth')
plt.ylabel('Absolute Difference')
plt.title('Average Offspring')

# plt.savefig('figs/convergence_errors.png')
plt.show()

# %% Plot upper bound and average number of particles.
depths = range(DEPTH_MAX + 1)
avg_particles = depth_group['threads'].mean().tolist()
std_particles = depth_group['threads'].mean().tolist()
upper = fin_mux / ( 1 - fin_muy)

# plt.errorbar(depths, avg_particles, yerr=std_particles, capsize=3)
plt.plot(depths, avg_particles)
plt.axhline(upper, linestyle='--')
plt.xlabel('Depth')
plt.ylabel('Average Number of States in Depth')
plt.savefig('figs/upperbound_particles.png')
plt.show()

"""
Negative correlation plots
"""

# %% Get synthetic data for negative correlation Plots
SAMPLES = 100
DEPTH_MAX = 50
ds = DepthSampler([0.5 for _ in range(4)], DEPTH_MAX)
df = ds.draw_samples(SAMPLES, ['threads', 'thread_children'])
df['thread_children'] = (df['thread_children']
                         .map(lambda l: 0 if len(l) == 0 else sum(l) / len(l)))
# df['merge_degree'] = (df['merge_degree']
#                       .map(lambda l: 0 if len(l) == 0 else sum(l) / len(l)))

# %% Get overall spearman r coeff
d = 20
ys = df[df['depth'] >= d]['thread_children'].tolist()
zs = df[df['depth'] >= d]['threads'].tolist()
print spearmanr(zs, ys)

# %% Plot Spearman r coeff estimates by depth.
rhos = []
p_vals = []
depths = range(1, DEPTH_MAX + 1)
for d in depths:
    ys = df[df['depth'] == d]['thread_children'].tolist()
    zs = df[df['depth'] == d]['threads'].tolist()
    r, p = spearmanr(zs, ys)
    rhos.append(r)
    p_vals.append(p)

plt.plot(depths, rhos)
plt.plot(depths, p_vals, '--')
plt.axhline(0, linestyle=':')
plt.show()

# %% Negative correlation between number of particles and offspring.
depth = 20
cleaned = df[df['depth'] >= depth]
# cleaned = df
cleaned = cleaned[np.abs(cleaned.threads-cleaned.threads.mean())<=(3*cleaned.threads.std())]
cleaned = cleaned[np.abs(cleaned.thread_children-cleaned.thread_children.mean())<=(3*cleaned.thread_children.std())]
cleaned.plot('threads', 'thread_children', kind='scatter')
plt.xlabel('Number of Particles')
plt.ylabel('Average Offspring')
plt.savefig('figs/offspring_correlation.png')
plt.show()

# %% Correlation between number of particles and merge degree.
cleaned = df
cleaned = cleaned[np.abs(cleaned.threads-cleaned.threads.mean())<=(3*cleaned.threads.std())]
cleaned = cleaned[np.abs(cleaned.merge_degree-cleaned.merge_degree.mean())<=(3*cleaned.merge_degree.std())]
cleaned.plot('threads', 'merge_degree', kind='scatter')
plt.xlabel('Number of Particles')
plt.ylabel('Merged')
plt.show()

"""
Average Number of Total States vs String Length
"""
#%%
MAX_DEPTH = 50
SAMPLES = 1000
PROBS = [0.5 for _ in range(4)]
depths = range(1, MAX_DEPTH + 1)

avgs = []
stds = []
for depth in depths:
    a, _, s = sample_total_states(SAMPLES, PROBS, depth)
    avgs.append(a)
    stds.append(s)
plt.errorbar(depths, avgs, yerr=stds, capsize=3)
plt.xlabel('Generalized String Length')
plt.ylabel('Average Total Number of States')
plt.savefig('figs/average_growth.png')
plt.show()

"""
Assumption comparing values and rates for different gen string sizes
"""

# %% Get data for length 10.
SAMPLES = 10000
ds10 = DepthSampler([0.5 for _ in range(4)], 10)
df10 = ds10.draw_samples(SAMPLES, ['threads', 'new_thread', 'thread_children'])
df10['thread_children'] = (df10['thread_children']
                         .map(lambda l: 0 if len(l) == 0 else sum(l) / len(l)))
depth_group10 = df10.groupby('depth')
muxs10 = depth_group10['new_thread'].mean().tolist()
stdxs10 = depth_group10['new_thread'].std().tolist()
muys10 = depth_group10['thread_children'].mean().tolist()
stdys10 = depth_group10['thread_children'].std().tolist()
df10.to_csv('10k_10length.csv')

# %% Get data for length 25
ds25 = DepthSampler([0.5 for _ in range(4)], 25)
df25 = ds25.draw_samples(SAMPLES, ['threads', 'new_thread', 'thread_children'])
df25['thread_children'] = (df25['thread_children']
                         .map(lambda l: 0 if len(l) == 0 else sum(l) / len(l)))
depth_group25 = df25.groupby('depth')
muxs25 = depth_group25['new_thread'].mean().tolist()
stdxs25 = depth_group25['new_thread'].std().tolist()
muys25 = depth_group25['thread_children'].mean().tolist()
stdys25 = depth_group25['thread_children'].std().tolist()
df25.to_csv('10k_25length.csv')

# %% Plot means of all three.


plt.figure(1, figsize=(18, 6))

plt.subplot(121)
plt.plot(range(11), muxs10, alpha=0.75, label='$|G| = 10$')
plt.plot(range(26), muxs25, alpha=0.75, label='$|G| = 25$')
plt.plot(range(51), muxs, alpha=0.75, label='$|G| = 50$')
# plt.errorbar(depths, muxs, yerr=stdxs, capsize=3)
plt.xlabel('Depth')
plt.ylabel('Average New Particles')
plt.legend(bbox_to_anchor=[1, 0.25])

plt.subplot(122)
# plt.errorbar(depths, muys, yerr=stdys, capsize=3)
plt.plot(range(10), muys10[:-1], alpha=0.75, label='$|G| = 10$')
plt.plot(range(25), muys25[:-1], alpha=0.75, label='$|G| = 25$')
plt.plot(range(50), muys[:-1], alpha=0.75, label='$|G| = 50$')
plt.xlabel('Depth')
plt.ylabel('Average Offspring')
plt.legend(bbox_to_anchor=[1, 0.25])
plt.savefig('figs/three_avg_new_n_offspring.png')
plt.show()

# %% Plot error of all three.
fin_mux, fin_muy = muxs[-1], muys[-2]
xdiffs = [abs(x - fin_mux) for x in muxs]
ydiffs = [abs(y - fin_muy) for y in muys]

fin_mux10, fin_muy10 = muxs10[-1], muys10[-2]
xdiffs10 = [abs(x - fin_mux10) for x in muxs10]
ydiffs10 = [abs(y - fin_muy10) for y in muys10]

fin_mux25, fin_muy25 = muxs25[-1], muys25[-2]
xdiffs25 = [abs(x - fin_mux25) for x in muxs25]
ydiffs25 = [abs(y - fin_muy25) for y in muys25]

prob = c_2(4)
ref_func = lambda t: prob ** t
ref_vals = [ref_func(d) for d in depths]

plt.figure(1, figsize=(18, 6))

plt.subplot(121)
plt.plot(range(11), xdiffs10, alpha=0.75, label='$|G|= 10$')
plt.plot(range(26), xdiffs25, alpha=0.75, label='$|G|= 25$')
plt.plot(range(51), xdiffs, alpha=0.75, label='$|G|= 50$')
plt.plot(range(51), ref_vals, '--', label='$c_2^n$')
plt.legend(bbox_to_anchor=[1, 1])
plt.xlabel('Depth')
plt.ylabel('Absolute Difference')
plt.title('Absolute Difference for Average New Particles')

plt.subplot(122)
plt.plot(range(10), ydiffs10[:-1], alpha=0.75, label='$|G| = 10$')
plt.plot(range(25), ydiffs25[:-1], alpha=0.75, label='$|G| = 25$')
plt.plot(range(50), ydiffs[:-1], alpha=0.75, label='$|G| = 50$')
plt.plot(range(50), ref_vals[:-1], '--', label='$c_2^n$')
plt.legend(bbox_to_anchor=[1, 1])
plt.xlabel('Depth')
plt.ylabel('Absolute Difference')
plt.title('Absolute Difference for Average Offspring')
plt.savefig('figs/three_convergence_errors.png')
plt.show()

# %% all three for expected value upper bound.
avgs10 = depth_group10['threads'].mean().tolist()
avgs25 = depth_group25['threads'].mean().tolist()
avgs50 = depth_group['threads'].mean().tolist()
upper = fin_mux / ( 1 - fin_muy)

# plt.errorbar(depths, avg_particles, yerr=std_particles, capsize=3)
plt.plot(range(11), avgs10, alpha=0.75, label='$|G| = 10$')
plt.plot(range(26), avgs25, alpha=0.75, label='$|G| = 25$')
plt.plot(range(51), avgs50, alpha=0.75, label='$|G| = 50$')
plt.axhline(upper, linestyle='--')
plt.xlabel('Depth')
plt.ylabel('Average Number of States in Depth')
plt.legend(bbox_to_anchor=[1, 0.35])
plt.savefig('figs/three_upperbound_particles.png')
plt.show()

# %% all three total states trend.
avgs10 = depth_group10['threads'].mean().tolist()
avgs25 = depth_group25['threads'].mean().tolist()
avgs50 = depth_group['threads'].mean().tolist()

acum10 = [avgs10[0]]
for a in avgs10[1:]:
    acum10.append(acum10[-1] + a)
acum25 = [avgs25[0]]
for a in avgs25[1:]:
    acum25.append(acum25[-1] + a)
acum50 = [avgs50[0]]
for a in avgs50[1:]:
    acum50.append(acum50[-1] + a)

plt.plot(range(11), acum10, alpha=0.75, label='$|G| = 10$')
plt.plot(range(26), acum25, alpha=0.75, label='$|G| = 25$')
plt.plot(range(51), acum50, alpha=0.75, label='$|G| = 50$')
plt.xlabel('Depth')
plt.ylabel('Cumulative Number of States in Depth')
plt.legend(bbox_to_anchor=[1, 0.35])
plt.savefig('figs/three_expected_lin_trend.png')
plt.show()
