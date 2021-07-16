# M3C Coursework 1: Tribal Competition Game

Project 1 ‘Tribal Competition Game’ completed for ‘High Performance Computing’ module.

The code simulates a game consisting of an N x N grid where each cell represents a village. Within each iteration a village changes to either a ‘Mercenaries’ or ‘Collaborators’ based on randomly generated probabilities. The affiliation of a cell influences the affiliation of neighbouring cell. This code was used the simulate the game over multiple iterations and investigate into the convergence behaviours of the villages based on several input parameters. All code was written in Python.

Below is a description of the files included in this repository:

- hw1_dev.py - File containing functions named `simulate1` and `simulate2` used to simulate the game. Functions named `analyse` was used to generate four plots used to analyse the behaviours emerging from the game. The docstring of this function contains a detailed analysis describing each plot, the main trends observed and a conclusion. The function `plot_S` is used to create an animation showing the progression of the game. 

- hw11.png - Figure 1: The two plots in this figure show the number of iterations (years) it takes for villages in N x N grid (matrix S) to converge to either all Collaborators or all Mercenaries for 100 values in the range 1.0<b<=1.5 over 500 iterations.

- hw12.png - Figure 2: This figure shows the fraction of villages which are Collaborator's over 200 iterations for 6 values of b 1.05<=b<=1.1 (as b is greater than 1).

- hw13.png - Figure 3: This figure shows the fraction of villages which are Collaborator's over 200 iterations for 6 values of b 1.01<=b<=1.11 (as b is greater than 1).

- hw14.png - Figure 4: This figure shows the fraction of villages which are Collaborator's over 200 iterations for 6 values of b 1.4<=b<=1.5.
