# -*- coding: utf-8 -*-
"""Calculates the probability that a Dart-Throwing Chimp has to provide a
Counter-Factual Forecast that is better than a naive uniform prob. distr.

Created on Thu Jan  9 17:00:17 2020
@authors: Giovanni Ciriani & Zarak Mahmud
"""
# Will need ternary library  for plot of 3 bin example:
#   https://github.com/marcharper/python-ternary
#   installed through Anaconda dashboard (for instructions
#   https://stackoverflow.com/questions/39299726/cant-find-package-on-anaconda-navigator-what-to-do-next)

import math
import numpy as np
import scipy.special
#from scipy import special
#import matplotlib.pyplot as plt
import pandas as pd
#import os.path
#import fire
#import ternary

Z_LOG = 0.005 # constant replacing zero values that would cause log = -inf

F_TEST = [0.5, 0.3, 0.2] # frequency values to test the code


def prob_space(m=2, g=2, type= 'lin', MIN_LOG = 0.005):
    """
    Provide the probability space for: 
    m = number of mutually exclusive events;
    g = size of grid of probabilities.
    Returns a np.array of m rows, and n columns, where each column is a vector 
    of probabilities, i.e. a vector with sum = 1 = 100% probability; 
    each component of the vector is the probability for that event;
    g is the number of values on the grid, i.e with g = 3, the probabilities 
    used for the components are 0, 50% or 100%;
    type of grid is "lin" for linearly spaced grid points, or "log" spaced;
    min_log is the minimum log space value, default 0.5%.
    Exact number of points in space = scipy.special.comb(g+m-2, m-1). Examples:
    prob.space(2, 3).shape -> (2, 6), i.e. 6 two-dimensional vectors;
    prob.space(4, 11).shape -> (4, 286), i.e. 286 four-dim. vectors;
    prob.space(8, 11).shape ->(8, 19448), i.e. 19,448 eight-dim. vectors;
    %timeit -n 1 -r 1 prob_space(8,11) takes 4.8 s;
    use carefully, because m=8, g=101 would produce 26*10^9 vectors, 74 days.
    Could be improved by using:
    numpy.ma.masked_where(condition, a, copy=True)[source]
    """
    if type == 'lin':
        p = np.linspace(0, 1, g)
    elif type == 'log':
        p = np.logspace(math.log10(MIN_LOG), 0, g)
    else: 
        print("Custom probability space not implemented yet;", 
              "using [0., 1.] instead.")
        p = np.array([0.,1.])
    # Generate grid of only m-1 event
    P = [p for _ in range(m - 1)]
    prob = np.meshgrid(*P ) #, sparse=True) #list of m-1 elements
    # add last dimension as complement to 1 of sum of prob of previous m-1
    prob.append(1 - np.sum(prob, axis=0))
    prob = np.array(prob)
    # find vectors columns to keep, by finding points whose total prob <=1
    z = np.isclose(prob, 0) # index of elements close to zero
    prob[z] = 0 # set to exact zero all elements close enough
    # only vectors whose last row is not negative are to be kept
    keep = prob[-1] >= 0. 
    return prob[:, keep].T


def FS(F, P, C=0.005):
    """    
    Calculates Fair Skill score with scaling log base 2 
    F = frequency, row vector of actual results.
    P = 2-dim. array of forecasts, each forecast is a row vector of 
        probabilities of mutually exclusive events summing to 1.
    Returns an np.array row of n scores, one for each row of P passed. 
    C = small constant to handle log(0) exceptions.
    Examples: F = np.array([0.01, 0.04, 0.11, 0.22, 0.62])
    P = np.array([0.29,0.45,0.19,0.07,0.])
    FS(F, P) -> -3.59153205
    FS(np.array([0,.1,.9]),prob_space(3,3)) ->  array([ 0.80752727, 
    -0.08593153, -6.05889369,  0.57773093, -5.39523123, -5.29595803])
    """
    # replace 0 values with C then normalize
    # start by counting 0 occurences
    zz = np.isclose(P, 0)
    # number of zero exceptions for each vector
    if P.ndim == 1:
        # handle case when P is a single row
        n_zeros = np.count_nonzero(zz) # number of zeros in row
        # scale down non-zero components, so that with C the sum is 1 
        P = (1- (n_zeros*C)) * P 
    else:
        # handle case when P is an array of several rows
        n_zeros = np.count_nonzero(zz, axis=1) 
        # normalize by scaling each row for its own number of zeros * C
        for i, nz in enumerate(n_zeros):
            P[i,:] = (1- (nz*C)) * P[i,:]
    # replaces zero elements with C to handle log(0) exceptions
    P[zz] = C
    # calculate Fair Skill
    return np.dot(F, np.log2(P.T)) + np.log2(len(F))


def DTC(actual: np.ndarray, max_size = 10000, grid = 101) -> float:
    """    
    Calculates the chance that a Dart Throwing Chimp has to score better than
    clueless, i.e. ignorance prior, or uniform forecast, using the Fair Skill.
    actual = np.array vector of actual frequencies 
    max_size
    grid
    Returns the probability that a dart-throwing chimp 
    would have to obtain a score better than ignorance prior.
    Examples: 
        actual1 = np.array([0.5,0.3,0.2]), 
        DTC(actual1) -> 0.13434285

        actual2 = np.array([0.6,0.4])]
        DTC(actual2) -> 0.18811881
    """
    # actual = actual.dropna().values
    actual = actual[~np.isnan(actual)]

    m = actual.shape[0] # mutually exclusive events
    # calculate gridsize g that produces prob_space of size at most max_size
    g = 5 # minimum estimate to start searching from
    # combinatorial formula to calculate prob.space
    while scipy.special.comb(g+m-2, m-1) <= max_size:
       g = g + 1
    g = min( g-1, 101 ) # keep 101, should the calculation exceed 101
    # set up dart table of all possible probabilities combinations
    dart_table = prob_space(len(actual), g) 
    # calculate Fair Skill for all probability combinations
    chimp_FS = FS(actual, dart_table)
    # calculate the chance of doing better than clueless and add it 
    return (chimp_FS > 0).sum() / chimp_FS.shape[0] 


def apply_DTC(Actuals: list) -> list:
    """Performs DTC on a collection of numpy arrays. """
    # [f(x) for x in X] is a common pattern for applying some function to
    # a collection
    Chimp_chances = [DTC(actual) for actual in Actuals]
    print ("Chimp Test \n chimp has chances ", Chimp_chances, 
           " \n of doing better than clueless in the respective CFFs")
    return Chimp_chances


def read_data(file_path: str) -> tuple:
    """Returns a tuple of a list of questions and a list of np arrays"""
    Actuals = pd.read_csv(file_path)
    CFFs = Actuals.CFFs
    Actuals = list(Actuals.drop(columns='CFFs').values)
    return CFFs, Actuals


def write_data(labels: list, data: list, filename: str) -> None:
    assert len(labels) == len(data), "Length of labels does not match data"
    df = pd.DataFrame({'CFFs': labels, 'DTC': data})
    df.to_csv(filename, index=False)


def main(test=False):
    folder_GJ= "C:/Users/Giovanni/Documents/Documents/Good Judgment/FOCUS project/"
    file_actuals = "Analysis Cycle 3 GJ2.0 - Actuals.csv"
    file_chimp = "Chimp Chance.csv"

    if test:
        Actuals = [np.array([0.5,0.3,0.2]), np.array([0.6,0.4])] # example
        CFFs = ['Test1', 'Test2']
    else:
        # read actuals from file
        CFFs, Actuals = read_data(file_actuals)

    Chimp_chances = apply_DTC(Actuals)

    #write results to file_chimp          
    write_data(CFFs, Chimp_chances, file_chimp)


if __name__ == "__main__":
    main(test=True)
