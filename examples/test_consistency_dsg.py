# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:27:15 2016

@author: bitzer
"""

import rtmodels
import numpy as np
from scipy.stats import multivariate_normal


#%% make model
dotstd = 70.0

stimulus = np.random.randint(1, 3, 200)
means = np.c_[[-25., 0], [25, 0]]
features = means[:, stimulus-1]
features = np.tile(features, (25, 1, 1))
features += np.random.normal(scale=70, size=features.shape)

model = rtmodels.discrete_static_gauss(Trials=features, dt=0.1, choices=[1,2],
                                       means=means, maxrt=2.5, toresponse=[0, 5.])
                                       
model.noisestd = 1e-10
model.intstd = dotstd
model.bound = 0.99
model.bstretch = 0.2
model.bshape = 0.4
model.lapseprob = 0.0
model.ndtmean = -20
model.prior = 0.34


#%% sample responses from model with numba
ch, rt = model.gen_response(np.arange(200))
model.plot_response_distribution(ch, rt)


#%% sample responses through computation of logpost
log_post, log_lik = model.compute_logpost_from_features(np.arange(200))
ch2, rt2 = model.gen_response_from_logpost(log_post)
model.plot_response_distribution(ch2, rt2)

print('sum of absolute differences between responses: %e' % 
    np.sum(np.abs(rt-rt2.squeeze())))


#%% test surprise
trind = 0
surprise = model.compute_surprise(np.c_[np.full(model.S, -np.inf), 
    np.zeros(model.S)], log_lik[:, :, trind].squeeze())
                                        
mvnlogpdf = multivariate_normal.logpdf(features[:, :, trind], mean=means[:, 1], 
                                       cov=model.intstd**2)

# model.compute_surprise will use model.prior as log_post for first time point
# hence, you'd expect a discrepancy as long as the prior differs from [0, 1]
print('absolute difference of surprise for first time point: %e' % 
    (surprise[0] + mvnlogpdf[0],))
print('sum of absolute differences for remaining time points: %e' % 
    np.sum(np.abs(surprise[1:] + mvnlogpdf[1:])))