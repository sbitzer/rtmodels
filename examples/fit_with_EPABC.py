# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:48:16 2016

@author: Sebastian Bitzer (sebastian.bitzer@tu-dresden.de)
"""

import os.path
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from pyEPABC import run_EPABC
import rtmodels


def plot_param_dist(samples, parnames, axlim_q=1):
    pg = sns.PairGrid(samples, hue='distribution', diag_sharey=False)
    pg.map_diag(sns.kdeplot, shade=True)
    pg.map_offdiag(plt.scatter, alpha=0.3)
    
    # adjust axis limits distorted by kdeplot
    for p, name in enumerate(parnames):
        m = samples[name].quantile(axlim_q)
        lims = [samples[name].min() - 0.1 * m, 1.1 * m]
        pg.axes[0, p].set_xlim(lims)
        if p == 0:
            pg.axes[p, 1].set_ylim(lims)
        else:
            pg.axes[p, 0].set_ylim(lims)
            
    pg.add_legend(frameon=True)
    
    return pg


if __name__ == "__main__":
    # load data
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'responsedata.csv'))
    
    parnames = ['ndtmean', 'ndtspread', 'noisestd', 'bound', 'prior', 
                'lapseprob', 'lapsetoprob']

    # number of samples to plot later
    N = 500
    
    # initialise samples
    samples = pd.DataFrame([], columns=(parnames+['distribution']))
    
    # implement constraints through transformations
    # ndtmean is unconstrained, 
    # ndtspread and noisestd must be positive, 
    # bound must be in [0.5, 1]
    # prior, lapseprob and lapsetoprob must be in [0, 1]
    paramtransform = lambda params: np.c_[params[:, 0], 
                                          np.exp(params[:, 1:3]), 
                                          norm.cdf(params[:, 3] / 2 + 0.5), 
                                          norm.cdf(params[:, 4:])]
    
    # these define the priors
    prior_mean = np.array([-1, -1.5, 4, 0, 0, -1, 0])
    prior_cov = np.diag(np.array([1, 1, 1.5, 1, 1, 1, 1]) ** 2)
    
    # sample from EPABC prior
    samples_pr = np.random.multivariate_normal(prior_mean, prior_cov, N)
    samples_pr = pd.DataFrame(paramtransform(samples_pr), columns=parnames)
    samples_pr['distribution'] = 'epabc_prior'
    samples = samples.append(samples_pr, ignore_index=True)
    
    # plot the prior(s)
    pg_prior = plot_param_dist(samples[samples['distribution'].map(
        lambda x: x.endswith('prior'))], parnames, axlim_q=0.98)
    
    # make model
    model = rtmodels.discrete_static_gauss(maxrt=2.5, choices=[1, 2], 
        Trials=data.stimulus.values, means=np.c_[[-25, 0], [25, 0]], 
        intstd=70)
    
    # wrapper for sampling from model and directly computing distances
    simfun = lambda data, dind, parsamples: model.gen_distances_with_params(
        data[0], data[1], dind, paramtransform(parsamples), parnames)
    distfun = None
    
    # maximum distance allowed for accepting samples, note that for 
    # response_dist this is in units of RT
    epsilon = 0.05
    
    # normalising constant of the uniform distribution defined by distfun and 
    # epsilon; for response_dist this is: (2*epsilon for the Euclidean distance 
    # between true and sampled RT and another factor of 2 for the two possible 
    # responses - timed out responses don't occur here, because they are only a
    # point mass, i.e., one particular value of response and RT)
    veps = 2 * 2 * epsilon
    
    # run EPABC
    ep_mean, ep_cov, ep_logml, nacc, ntotal, runtime = run_EPABC(
        data.values[:, 1:], simfun, distfun, prior_mean, prior_cov, 
        epsilon=epsilon, minacc=500, samplestep=10000, samplemax=2000000, 
        npass=3, alpha=0.5, veps=veps)
    
    # sample from EPABC posterior
    samples_pos = np.random.multivariate_normal(ep_mean, ep_cov, N)
    samples_pos = pd.DataFrame(paramtransform(samples_pos), columns=parnames)
    samples_pos['distribution'] = 'epabc_pos'
    samples = samples.append(samples_pos, ignore_index=True)
        
    # see what happens when you ignore RTs, epco = ep choice only
    simfun = lambda data, dind, parsamples: model.gen_distances_with_params(
        data[0], np.nan, dind, paramtransform(parsamples), parnames)
    epco_mean, epco_cov, epco_logml, epco_nacc, epco_ntotal, epco_runtime = run_EPABC(
        data.values[:, 1:], simfun, distfun, prior_mean, prior_cov, 
        epsilon=epsilon, minacc=10000, samplestep=10000, samplemax=2000000, 
        npass=3, alpha=0.5, veps=0.5)
    
    # sample from EPABC choice only posterior
    samples_posco = np.random.multivariate_normal(epco_mean, epco_cov, N)
    samples_posco = pd.DataFrame(paramtransform(samples_posco), columns=parnames)
    samples_posco['distribution'] = 'epabcco_pos'
    samples = samples.append(samples_posco, ignore_index=True)
    
    # plot the posterior(s)
    pg_pos = plot_param_dist(samples[samples['distribution'].map(
        lambda x: x.endswith('pos'))], parnames)
        
    plt.show()