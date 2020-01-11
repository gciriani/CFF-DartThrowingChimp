# -*- coding: utf-8 -*-
"""
Calculates the probability that a Dart-Throwing Chimp has to provide a
Counter-Factual Forecast that is useful

Created on Thu Jan  9 17:00:17 2020
@author: Giovanni
Will need ternary library  for plot of 3 bin example: 
    https://github.com/marcharper/python-ternary
    installed through Anaconda dashboard (for instructions
    https://stackoverflow.com/questions/39299726/cant-find-package-on-anaconda-navigator-what-to-do-next)
"""

import numpy as np
#import pandas as pd
#import os.path
import ternary
import matplotlib.pyplot as plt


EPSILON = 0.05


def FS( f, p, m):
    """    Calculates Fair Skill score with log base 2 score
    f = frequency vector of actual outcomes
    p = probability vector of mutually exclusive events
    m = number of mutually exclusive events
    check if it works even when y has both 0 and 1 values?
    """
    #timing: %timeit np.multiply(x, x)
    #the following implementation of multiply, needs to be a pandas DataFrame
    #FS = np.log2(x).multiply(y,axis='index')      
    return (f * np.log2(p)).sum(axis=1) + np.log2(m) 


def FS2(f, x, y):
    """Calculates Fair Skill score with log base 2 score

    Args:
        f (ndarray) : frequency vector of actual outcomes
        x, y (ndarray) : linspace (0.01, 1) with 100 points

    Returns:
        Fair score calculation
    """
    f1, f2, f3 = f
    return f1 * np.log2(x) + f2 * np.log2(y) + f3 * np.log2(1 - x - y) + np.log2(3)


#3-bin example
m = 3 #mutually excluding events
freq = [0.5, 0.3, 0.2] #frequency of each event, must sum to 1 = 100% prob.
prob = [[0.6, 0.25, 0.15] , [0.7, 0.2, 0.1]] #forecast probabilities
print (FS(freq, prob, m) )

#1st step: 
#obtain a linear space of all 100 percentage probabilities combos
#for a problem with 3 bins and calculate FS, and Brier
#p_grid = np.linspace (0, 1, 3)# (0, 1, 3) for testing, eventually (0, 1, 101)
#also consider logspace for development in tune with log concept of information
#idea to refine to obtain the space
"""
https://stackoverflow.com/questions/28825219/how-can-i-create-an-n-dimensional-grid-in-numpy-to-evaluate-a-function-for-arbit
#You could use a list comprehension to generate all of the linspaces, and then pass that list to meshgrid with a * (to convert the list to a tuple of arguments).

XXX = np.meshgrid(*[np.linspace(i,j,numPoints)[:-1] for i,j in zip(mins,maxs)])
XXX is now a list of n arrays (each n dimensional).

I'm using straight forward Python list and argument operations.

np.lib.index_tricks has other index and grid generation functions and classes that might be of use.
"""
#X, Y = np.meshgrid(p_grid, p_grid)
#P_vect = [X,Y]
#eliminate all probability vectors that sum up to > 1 
#X and Y correspond respectively to bin 1 and 2
#calculate bin 3 by difference (1- X - Y)
#calculate probability of being above score obtained with uniform probability
#this is the probability that a Dart-Throwing Chimp has to be better than naive
#print (X,Y,P_vect)
#because log(0) = -inf we need to replace 0 with a number in its proximity
z = 0.005 #or other choices
#replace all 0 values with z, and normalize to obtain a total of 1
#c
#2nd step: 
#plot on a ternary graph the contour of FS and of BS identical to
#that obtained for uniform distribution: FS=0, BS=(1-1/m)^2
#3rd step:
#scale up the calculation so that it is possible to do calculate it for an
#arbitrary number of bins.
#4th step:
#apply 3rd step to round 3, and see where teams and individuals are situated 
#compared to the probability of being above uniform probability score.

def linear_space():
    p = np.linspace(0.01, 1, 100)
    x1, x2 = np.meshgrid(p, p, sparse=True)
    return x1, x2


if __name__ == "__main__":
    f = [0.5, 0.3, 0.2] #frequency of each event, must sum to 1 = 100% prob.
    xx, yy = linear_space()
    z = FS2(f, xx, yy)
    p = xx.squeeze()
    h = plt.contourf(p, p, z)
    plt.show()
