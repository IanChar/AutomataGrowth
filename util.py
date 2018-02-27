"""List of useful, commonly appearing functions."""

import operator as op
import matplotlib.pyplot as plt

# Taken from http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
def ncr(n, r):
    """Choose function.
    Args:
        n: Number to choose from.
        r: Amount to be chosen.
    Returns: Evaluation of binomial coefficient.
    """
    r = min(r, n-r)
    if r == 0:
        return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer / denom

def plot_comparison(val_range, func1, func2, func1_label='Function 1',
                    func2_label='Function 2', legend_loc='upper right'):
    """Plots comparison of two different functions.
    (Should actually probably just use plot_many from now on.
    Args:
        val_range: The range for the functions to be evaluated on.
        func1: The first function to evaluate.
        func2: The second function to evaluate.
        func1_label: The label to put for the first function.
        func2_label: The label to put for the second function.
        legend_loc: The location for the label.
    """
    func1_vals = [func1(a) for a in val_range]
    func2_vals = [func2(a) for a in val_range]
    plt.plot(val_range, func1_vals, label=func1_label)
    plt.plot(val_range, func2_vals, label=func2_label)
    plt.legend(loc=legend_loc)
    plt.show()

def plot_many(val_range, func_map, legend_loc='upper right'):
    """Plots an arbitrary amount of functions on the values range.
    Args:
        val_range: The range over which to evaluate the functions.
        func_map: Dictionary mapping function to label.
        legend_loc: The location for the plot's legend.
    """
    to_plot = [([func(a) for a in val_range], label)
               for func, label in func_map.iteritems()]
    for values, plot_label in to_plot:
        plt.plot(val_range, values, label=plot_label)
    plt.legend(loc=legend_loc)
    plt.show()
