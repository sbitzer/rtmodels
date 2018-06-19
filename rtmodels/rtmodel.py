# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:51:27 2016

@author: Sebastian Bitzer (sebastian.bitzer@tu-dresden.de)
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from numba import vectorize
import seaborn as sns

class rtmodel(object):
    __metaclass__ = ABCMeta
    
    # name of the model
    name = 'General RT model'
    
    # memory limit: the gen_response methods should not use more than that
    # at any point in time
    memlim = 6000.0
    
    # number of parameters in model
    P = None

    @property
    def L(self):
        return self._L
    
    @property
    def C(self):
        return self.choices.size

    def __init__(self, choices=[1, 2], maxrt=np.inf, toresponse=None):
        # number of trials currently stored in the model
        self._L = 0
        
        # maximum RT considered by the model; anything above will be timed-out
        self.maxrt = maxrt
        
        # choice and rt used for timed out trials
        if toresponse is None:
            self.toresponse = np.array([0, maxrt+1])
        else:
            self.toresponse = toresponse
        
        # choices (should be an iterable, will be converted to numpy array)
        self.choices = np.array(choices)
    
    
    @abstractmethod
    def gen_response(self, trind):
        pass
    
    @abstractmethod
    def gen_response_with_params(self, trind, params, parnames, user_code):
        pass
    
    @abstractmethod
    def estimate_memory_for_gen_response(self, N):
        pass
    
    def gen_response_from_Gauss_posterior(self, trind, parnames, mean, cov, S,
                                          transformfun=None, return_samples=False):
        samples = np.random.multivariate_normal(mean, cov, S)
        if transformfun is not None:
            samples = transformfun(samples)
            
        N = trind.size
        trind = np.tile(trind, (S, 1)).reshape((N*S, 1), order='F').squeeze()
        samples_long = np.tile(samples, (N, 1))
        
        choices, rts = self.gen_response_with_params(trind, samples_long, parnames)
        
        choices = np.reshape(choices, (N, S), order='C')
        rts = np.reshape(rts, (N, S), order='C')
        
        if return_samples:
            return choices, rts, samples
        else:
            return choices, rts
    
    
    def gen_distances_with_params(self, choice_data, rt_data, trind, params, 
                                  parnames):
        choices, rts = self.gen_response_with_params(trind, params, parnames)
        
        return response_distance(choice_data, choices, rt_data, rts)
    
    
    def plot_response_distribution(self, choice, rt, alpha=1, ax=None):
        rtlist = []
        for c in self.choices:
            rtlist.append(rt[choice == c])
        
        if ax is None:
            ax = plt.axes()
        
        _, _, patches = ax.hist(rtlist, bins=20, normed=True, 
                                range=(0, self.maxrt), stacked=True)
        
        if alpha < 1:
            for container in patches:
                for patch in container.patches:
                    patch.set_alpha(alpha)
        
        plt.show()
        
        return ax
        
    def plot_parameter_distribution(self, samples, names, q_lower=0, q_upper=1):
        pg = sns.PairGrid(samples, hue='distribution', diag_sharey=False)
        pg.map_diag(sns.distplot, kde=False)
        pg.map_offdiag(plt.scatter, alpha=0.3)
        
        # adjust axis limits distorted by kdeplot
        for p, name in enumerate(names):
            low = samples[name].quantile(q_lower)
            up = samples[name].quantile(q_upper)
            lims = [low - 0.1 * up, 1.1 * up]
            pg.axes[0, p].set_xlim(lims)
            
            if pg.axes.shape[0] > 1:
                if p == 0:
                    pg.axes[p, 1].set_ylim(lims)
                else:
                    pg.axes[p, 0].set_ylim(lims)
                
        pg.add_legend(frameon=True)
        
        return pg
        
        
    def estimate_entropy(self, choice, rt, B=50):
        """Estimates entropy of sampled response distribution based on histogram.
        
            See II.2 (ii) The resubstitution estimate of 
            "Nonparametric entropy estimation: an overview" by Beirlant et al.
            http://jimbeck.caltech.edu/summerlectures/references/Entropy%20estimation.pdf
            
            Current implementation is even simpler: it just makes a fixed 
            discrete approximation of the response distribution based on the 
            given number of bins B and the maxrt of the model. Then it computes
            the entropy of that discrete distribution.
            
            NOTE: you cannot easily compare entropies based on different 
            discrete approximations, i.e., entropies based on different B or
            maxrt.
        """
        # get distribution of choices
        ch_prob = np.array([np.sum(choice==ch) for ch in self.choices])
        ch_prob = ch_prob / choice.size
        
        # contribution from timed out trials
        to_prob = 1-ch_prob.sum()
        if to_prob > 0:
            entropy = -(to_prob) * np.log(to_prob)
        else:
            entropy = 0
        
        # for each choice compute entropy from histogram of RTs
        for ch, prob in zip(self.choices, ch_prob):
            if prob > 0:
                cnts, edges = np.histogram(rt[choice==ch], bins=B, range=(0, 
                                           self.maxrt))
                N = cnts.sum()
                dens = cnts / N * prob
                
                # ignore empty bins
                dens = dens[dens>0]
                
                entropy -= np.sum(dens * np.log(dens))
            
        return entropy
            
        
    def __str__(self):
        # underlined model name
        info =  self.name + '\n' + '-' * len(self.name) + '\n'  
        
        # parameters
        info += 'choices  : ' + ', '.join(map(lambda s: '{: 1d}'.format(s), 
                                              self.choices)) + '\n'
        info += 'maxrt    : %4.2f' % self.maxrt + '\n'
        info += 'time outs: %2d, %4.2f' % (self.toresponse[0], 
                                           self.toresponse[1]) + '\n'
    
        return info
        

def estimate_abc_loglik(choice_data, rt_data, choice_sample, rt_sample, epsilon):
    N = choice_data.size
    if choice_sample.ndim == 2:
        NP, P = choice_sample.shape
    elif N == 1:
        P = choice_sample.size
        NP = 1
        choice_sample = choice_sample[None, :]
        rt_sample = rt_sample[None, :]
    if N != NP:
        raise ValueError('The number of data points (N=%d) ' % N + 
                         'does not fit together with the dimensions of ' + 
                         'the samples from the model (%d x %d)!' % (NP, P))
                         
    # expand choice_data and rt_data
    choice_data = np.tile(choice_data, (1, P)).squeeze()
    rt_data = np.tile(rt_data, (1, P)).squeeze()
    # reshape the samples
    choice_sample = choice_sample.reshape((N*P, 1), order='F').squeeze()
    rt_sample = rt_sample.reshape((N*P, 1), order='F').squeeze()
    
    # then compute distances, reshape
    accepted = response_distance(choice_data, choice_sample, 
                                 rt_data, rt_sample) < epsilon
    # reshape into sample shape and sum
    naccepted = np.sum(accepted.reshape((N, P), order='F'), axis=1)
    
    return np.log(naccepted) - np.log(P)


def response_distance(choice1, choice2, rt1, rt2, useRT=True):
    if ((np.isscalar(choice1) and type(choice1) is str) or 
        choice1.dtype.type is np.str_):
        raise ValueError('The distance function is currently not ' + 
                         'implemented for choices coded with strings.')
    
    if useRT:
        return response_dist_ufunc(choice1, choice2, rt1, rt2)
    else:
        return response_dist_ufunc(choice1, choice2, np.nan, np.nan)


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