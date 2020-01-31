# -*- coding: utf-8 -*-
"""Calculates the probability that a Dart-Throwing Chimp has to provide a
Counter-Factual Forecast that is better than a naive uniform prob. distr.
Created on Thu Jan  9 17:00:17 2020
@authors: Giovanni Ciriani & Zarak Mahmud
"""


import math
import numpy as np
import scipy.special
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


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
    #generate grid for one axis
    if type == 'lin':
        p = np.linspace(0, 1, g)
    elif type == 'log':
        p = np.logspace(math.log10(MIN_LOG), 0, g)
    else: 
        print("Custom probability space not implemented yet;", 
              "using [0., 1.] instead.")
        p = np.array([0.,1.])
    
    # Propagate grid along m-1 axes, the m-th generaated by complement to 1
    P = [p for _ in range(m - 1)] # input for meshgrid
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


def DTC(actual, forecast, max_size = 10000, max_grid = 101):
    """    
    Calculates the chance that a Dart Throwing Chimp has to score better than
    a forecast, using the Fair Skill.
    actuals = np.array row vector of actual frequencies 
    forecast = np.array row vector of forecasted frequencies
    max_size = maximum number of points to fit the probability grids
    max_grid = maximum number of points on grid (0, 0.01, ... 1.)
    Returns a scalar, probabilty that a dart-throwing chimp 
    would have to obtain scores better than ignorance prior.
    Examples: 
    DTC(np.array([0.5,0.3,0.2]), np.array([1/3,1/3,1/3])) -> 0.13434285
    DTC(np.array([0.6,0.4]), np.array([1/3,1/3,1/3])) -> 0.18811881
    in these example forecast is ignorance prior, or uniform forecast.
    
    Precision of result can be calculated: max_size * DTC 
    gives the number of points n_res; apply n_res and (m-1) to scipy formula 
    to obtain g_res, which is the nominal grid size; error is proportional to 
    (m-1)/g_res.
    """
    
    m = actual.shape[0] # number of mutually exclusive events forecasted
    # calculate gridsize g that produces prob_space of size at most max_size
    g = 5 # minimum estimate to start searching from
    # combinatorial formula to calculate prob.space fitting in max_size
    while scipy.special.comb(g+m-2, m-1) <= max_size:
       g = g + 1
       
    g = min( g-1, max_grid ) # ex. keep 101, should the calculation exceed 101
    
    # set up dart table of all possible probabilities combinations
    dart_table = prob_space(m, g) 
    # calculate Fair Skill for all probability combinations
    chimp_FS = FS(actual, dart_table)
    # calculate Fair Skill for the forecaster considered
    forecast_FS = FS(actual, forecast)
    # calculate chance of forecast scoring better than chimp
    return (chimp_FS >= forecast_FS).sum() / chimp_FS.shape[0] 


def DTC_demo(p1=0.5, p2=0.3):
    """3-Bin, 2-D representation to illustrate the concept"""
    Example = np.array([p1, p2, 1-p1-p2])
    P = prob_space(3,101)
    m = Example.shape[0]
    Z = FS(Example, P)
    chimp = DTC(Example, np.array([1/3,1/3,1/3]))
    Better = Z > 0
    Xbetter, Ybetter = P[Better,0], P[Better,1]
    X , Y = P[:,0] , P[:,1]
    #plt.plot(X,Y, 'b.')
    plt.plot(Xbetter, Ybetter, 'g.', 
             label='Forecast space with Fair Skill > 0')
    plt.plot(X, Y, '.b', markersize=0.5, 
             label='Probability space of possible forecasts')
    plt.plot(Example[0], Example[1], 'ro', 
             label='Perfect Counter-Factual Forecast')
    plt.plot([1/m], [1/m], '+b', markersize=10.,
             label='Ignorance prior')
    plt.xlabel('Prob. Bin 1')
    plt.ylabel('Prob. Bin 2')
    plt.title("Dart-Throwing Chimp - 3-Bin Forecast\n" + 
              "{0:.0%}".format(chimp) +
              " Chances of Forecasting Better Than Ignorance Prior")
    plt.legend(loc="upper right")
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 8)
    return


# 1 commented folder makes it easier to move between computers
#folder_GJ= "C:/Users/DebraLMFT/Downloads/"
folder_GJ= "C:/Users/Giovanni/Documents/Documents/Good Judgment/FOCUS project/"
file_actuals = "Analysis Cycle 3 GJ2.0 - Actuals.csv"
file_team = "Analysis Cycle 3 GJ2.0 - TeamArray.csv"
file_chimp = "Chimp Chance.csv"

# read actuals from file
Actuals = pd.read_csv(folder_GJ + file_actuals, index_col=0)
CFFs = Actuals.index.values # keep labels for each CFF
# read team's corresponding file
Tvalues = pd.read_csv(folder_GJ + file_team, index_col=0)

# calculate DTC chances for all CFFs
# %timeit [ DTC(actual.dropna()) for _, actual in Actuals.iterrows() ]
#        3.87 s ± 326 ms per loop
# %timeit [ DTC( Actuals.iloc[i].dropna() ) for i in range(Actuals.shape[0]) ]
#        3.57 s ± 42 ms per loop
Chimp_chances = [ DTC( actual.dropna(), forecast.dropna(), max_size = 1000 )
                       for (_, actual), (_, forecast) 
                       in zip(Actuals.iterrows(), Tvalues.iterrows()) ]

print ("Chimp Test \n chimp has chances ", Chimp_chances, 
       " \n of doing better than forecast in the respective CFFs")

#write results to file_chimp          
df = pd.DataFrame({'CFFs': CFFs, 'DTC': Chimp_chances})
df.to_csv(folder_GJ + file_chimp, index=False)
