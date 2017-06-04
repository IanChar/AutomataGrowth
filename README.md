# Char/Lladser Research

## Research Reports

I am hoping to make weekly research reports in the form of ipython notebooks
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

#### Downloading this Repository
You will also need to download this repository onto your local machine. To do
this, open a terminal and navigate to a directory in which you wish to store
the code. Then run the following command:

```
git clone https://github.com/IanChar/AutomataResearch.git
```

If this is your first time using git, you may have to also enter in you GitHub
credentials.

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

## Code

Almost all of the code I have written for the particular algorithm using
an AC automaton with merging can be found in [aho\_merging](./aho_merging).
In here, [aho\_construction](./aho_merging/aho_construction.py) contains the
algorithm for constructing the minimal DFA. The rest of the files in this
directory contain helper functions for analyzing the produced minimal DFA.
