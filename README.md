# Char/Lladser Research
## Optimal Construction of Minimal DFA for Generalized Strings
## Overview

Here you will find the code for research on minimal deterministic finite automata for generalized strings. For more information about this research see https://link.springer.com/content/pdf/10.1007/s11009-019-09706-8.pdf

## Downloading this Repository
You will also need to download this repository onto your local machine. To do
this, open a terminal and navigate to a directory in which you wish to store
the code. Then run the following command:

```
git clone https://github.com/IanChar/AutomataResearch.git
```

If this is your first time using git, you may have to also enter in you GitHub
credentials.

## Dependencies

To run all of the code in this repository you will need to have...

* [python 2.7](https://www.python.org/download/releases/2.7/)
* [matplotlib](https://matplotlib.org/)
* [numpy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [jupyter](http://jupyter.org/) (Notebooks only)

## Code

The main algorithm presented in the BS/MS thesis and paper can be found in [merge_alg.py](./merge_alg.py). This file also contains the deterministic finite automaton object that is returned by the algorithm. The implementation of this data structure is done via objects and pointers. Currently this file is set up so that if one runs

```
python merge_alg.py
```

a detailed description of the minimal DFA corresponding to {A, B}{C, B, D}{A}{C, B, D} is printed out. This can be changed by adjusting the last two lines of the file.

#### Sampling from Synthetic Data

I have also set up a framework to sample from synthetic data relatively easily. If you want to observe how a certain value changes over the depth of the automaton, you might be able to do this with [sampler.py](./common/sampler.py). This file contains the DepthSampler object which, once initialized, can draw samples and find the values of a list of different properties. The synthetic data is returned as a pandas' DataFrame where each draw in the DataFrame has a corresponding "depth." The properties that one can observe are as follows:

* states - The number of states within each depth.
* threads - The number of "threads" (states with a failure not going to start state) within each depth.
* thread_children - Gets the number of children each thread has (does not double count children between threads).
* total_children - Get the number of children for each thread within a depth (may double count children between threads).
* new_thread - Gets the number of new threads per depth (i.e. failure goes to depth 1).
* merge_degree - Gets the number of times each thread was merged for each depth.
* failure_chain_lengths - Gets the length of each failure chain with each depth.
* failure_chain_path - For each depth, pick a random failure chain and return the depths visited by that failure chain.
* growth_ratio - Gets (states in depth n + 1) / (states in depth n + 1) for each depth n.

A detailed usage of this framework as well as some of the ways to extract statistics from the DataFrame can be found in the notebook [July24_2017_SizeModel.ipynb](./notebooks/July24_2017_SizeModel.ipynb) (as well in other notebooks).

If you don't care about observing a trend in the depth, you might be able to find what you are looking for in [sampling_misc.py](./common/sampling_misc.py). This contains methods to do things such as sample the total number of states in the automaton.

#### Other notes about the code

* The directory [common](./common) contains code that is useful for implementing new features. Along with the sampling mentioned above, it contains utility functions for generating random generalized strings and finidng probabilities that two generalized characters share a letter given some distribution.

* The directory [experimental](./experimental) contains code that I wrote to answer specific research questions I had. Most of these will have a short description at the top. Many times there will also be a corresponding notebook that calls functions from the file in question.

* The directory [marschall](./marschall) contains code implementing the algorithm described in Tobias Marschall's paper [Construction of minimal deterministic finite automata from biological motifs](https://www.sciencedirect.com/science/article/pii/S0304397510006948).

## Notebooks

At some point, I started making weekly research reports in the form of ipython notebooks
found in the [notebooks](./notebooks) directory. From GitHub, you should be
able to click on each notebook and see a pdf rendering of the report. From now
on, I will be naming the reports with the format MonthDay\_Year\_Subject.ipynb

### Running/Altering the Notebooks

#### Installing ipython
If you wish to run the code and the notebook on your local machine, you will
need to install ipython. Assuming you already have pip installed (which should
be the case if you have python installed), run the following command:

```
sudo pip install ipython
```

#### Running the Notebooks
Once ipython has been installed and the code has been downloaded, navigate to
the notebooks directory in your terminal and type the following command:

```
ipython notebook
```

This should open your internet browser where you can see each of the available
notebooks you can select.

#### Running Code in the ipython Notebook
ipython notebooks are divided into different cells that could have code or
LaTeX code in them. In order to run a cell use CTRL-ENTER. The first thing that
should be done in any of the notebooks is to run the cell at the top of the
notebook containing code. This will import necessary libraries to run the code
found further down in the notebook.

Starting with my May 29th notebook, I have tried to structure my code in such
a way that it is divided into two cells. The first cell contains the logic of
whatever I am doing while the second cell contains the inputs and runs the
functions in the first cell. Therefore if you want to run the functions with
different inputs you would run the first cell in order to load in the functions,
alter the inputs in the second cell, and then run the second cell.

If the logic that needs to be performed is too complex it is written in a
different file (see the next section for more information about that).
