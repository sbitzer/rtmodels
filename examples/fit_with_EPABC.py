# -*- coding: utf-8 -*-
"""
This script demonstrates basic properties of a discrete static Gauss model 
together with inference using pyEPABC. The example neglects the most useful
component of the model: the ability to include within-trial variations of 
stimulus features into model predictions. Consequently, the discrete static 
Gauss model here only implements a discretised drift-diffusion model (cf. 
Bitzer et al., Frontiers in Human Neuroscience, 2014, 8, 
https://doi.org/10.3389/fnhum.2014.00102).

Created on Tue Mar 29 15:48:16 2016

@author: Sebastian Bitzer (sebastian.bitzer@tu-dresden.de)
"""

import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyEPABC
from pyEPABC.parameters import exponential, gaussprob
import rtmodels


#%% load data
data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'responsedata.csv'))

N = data.shape[0]


#%% create response model with some standard parameter settings
model = rtmodels.discrete_static_gauss(dt=0.05, maxrt=2.5, choices=[1, 2], 
    Trials=data.stimulus.values, means=np.c_[[-25, 0], [25, 0]], 
    intstd=70, noisestd=60)

print(model)

ch_sim, rt_sim = model.gen_response(np.arange(N), 1000)

fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
model.plot_response_distribution(data.choice, data.RT, ax=axes[0])
model.plot_response_distribution(ch_sim, rt_sim, ax=axes[1])

axes[0].set_ylabel('response data')
axes[1].set_ylabel('model simulations')
axes[1].set_xlabel('RT (s)')


#%% define parameter prior
pars = pyEPABC.parameters.parameter_container()

pars.add_param('noisestd', np.log(50), 1.2, exponential())
pars.add_param('bound', 0.1, 1, gaussprob(0.5, 0.5))
pars.add_param('prior', 0, .5, gaussprob())
pars.add_param('ndtmean', -1, 1)
pars.add_param('ndtspread', np.log(.2), 1, exponential())
pars.add_param('lapseprob', -1.65, 1, gaussprob()) # median approx at 0.05
pars.add_param('lapsetoprob', 0, 1, gaussprob())

pg = pars.plot_param_dist();
pg.fig.tight_layout()


#%% fit with EP-ABC
simfun = lambda data, dind, parsamples: model.gen_distances_with_params(
        data[0], data[1], dind, pars.transform(parsamples), pars.names)

# maximum allowed difference in RT for accepting samples
epsilon = 0.05

ep_mean, ep_cov, ep_logml, nacc, ntotal, runtime = pyEPABC.run_EPABC(
        np.c_[data.choice, data.RT], simfun, None, pars.mu, pars.cov, 
        epsilon=epsilon, minacc=2000, samplestep=10000, samplemax=6000000, 
        npass=2, alpha=0.5, veps=2 * epsilon)


#%% investigate posterior
modes = pars.get_transformed_mode(ep_mean, ep_cov)

fig, axes = pars.compare_pdfs(ep_mean, ep_cov)

ch_post, rt_post = model.gen_response_from_Gauss_posterior(
        np.arange(N), pars.names, ep_mean, ep_cov, 500, pars.transform)

fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
model.plot_response_distribution(data.choice, data.RT, ax=axes[0])
model.plot_response_distribution(ch_post, rt_post, ax=axes[1])

axes[0].set_ylabel('response data')
axes[1].set_ylabel('posterior predictive')
axes[1].set_xlabel('RT (s)')


#%% show single trial accumulated evidence trajectories

# set model parameter values to mode of posterior
for mode, name in zip(modes, pars.names):
    setattr(model, name, mode)
    
trials = np.r_[0, 4, 175]
R = 5
logpost, logliks = model.compute_logpost_from_features(trials, R)

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)

times = (np.arange(logpost.shape[0]) + 1) * model.dt

for ax, tr in zip(axes, range(trials.size)):
    ax.set_title('trial %d' % trials[tr])
    for rep in range(R):
        # find first time point at which bound is crossed
        tind, cind = np.nonzero(logpost[:, :, tr, rep] > np.log(model.bound))
        if tind.size > 0:
            tind = tind[0]
        else:
            tind = logpost.shape[0]
        
        for alt in range(2):
            ax.plot(times[:tind+1], np.exp(logpost[:tind+1, alt, tr, rep]), 
                    color='C%d' % alt)
            ax.plot(times[tind:], np.exp(logpost[tind:, alt, tr, rep]), ':',
                    color='C%d' % alt)
    
xl = ax.get_xlim()
for ax in axes:
    ax.plot(xl, model.bound * np.ones(2), '--k', label='bound')
    ax.set_ylabel('prob. correct')
ax.set_xlim(xl)
ax.set_xlabel('decision time (s)');