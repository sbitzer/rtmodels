# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:51:27 2016

@author: Sebastian Bitzer (sebastian.bitzer@tu-dresden.de)
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize

class rtmodel(metaclass=ABCMeta):

    # number of trials currently stored in the model
    _L = 0

    @property
    def L(self):
        return self._L
    
    # maximum RT considered by the model; anything above will be timed-out
    maxrt = np.inf
    
    # choice and rt used for timed out trials
    toresponse = np.array([0, np.inf])
    
    # choices
    choices = np.array([1, 2])
    
    @property
    def C(self):
        return self.choices.size
    
    def __init__(self, choices=None, maxrt=None, toresponse=None):
        if choices is not None:
            self.choices = np.array(choices)
        if maxrt is not None:
            self.maxrt = maxrt
        if toresponse is not None:
            self.toresponse = toresponse
    
    @abstractmethod
    def gen_response(self, trind):
        pass
    
    @abstractmethod
    def gen_response_with_params(self, trind, params, parnames, user_code):
        pass
    
    
    def gen_distances_with_params(self, choice_data, rt_data, trind, params, 
                                  parnames):
        choices, rts = self.gen_response_with_params(trind, params, parnames)
        
        return rtmodel.response_distance(choice_data, choices, rt_data, rts)
    
    
    def plot_response_distribution(self, choice, rt):
        rtlist = []
        for c in self.choices:
            rtlist.append(rt[choice == c])
            
        ax = plt.axes()
        
        ax.hist(rtlist, bins=20, normed=True, range=(0, self.maxrt), stacked=True)
        
        plt.show()
        
        return ax
        
    
    def response_distance(choice1, choice2, rt1, rt2, useRT=True):
        if ((np.isscalar(choice1) and type(choice1) is str) or 
            choice1.dtype.type is np.str_):
            raise ValueError('The distance function is currently not ' + 
                             'implemented for choices coded with strings.')
        
        if useRT:
            return response_dist_ufunc(choice1, choice2, rt1, rt2)
        else:
            return response_dist_ufunc(choice1, choice2, np.nan, np.nan)
            
            
    def estimate_abc_loglik(choice_data, rt_data, choice_sample, rt_sample, epsilon):
        return ( np.log((rtmodel.response_distance(choice_data, choice_sample, 
            rt_data, rt_sample) < epsilon).sum(axis=0)) - 
            np.log(choice_sample.shape[0]) )
    

@vectorize(nopython=True)
def response_dist_ufunc(choice1, choice2, rt1, rt2):
    """
    Simple distance between two responses.
    
    Is infinite, if choices don't match, else it's the absolute difference 
    in RTs. If any of the RTs is NaN, RTs are ignored (dist=0, if choices match).
    """
    useRT = True
    if np.isnan(rt1) or np.isnan(rt2):
        useRT = False
    
    # this does not work with strings in numba
    match = choice1 == choice2
    
    if match:
        dist = 0.0
        if useRT:
            dist += np.abs(rt1 - rt2)
    else:
        dist = np.inf
    
    return dist