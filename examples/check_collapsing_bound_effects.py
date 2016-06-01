# -*- coding: utf-8 -*-
"""
This script implements sanity checks for the collapsing bounds in the discrete
static Gauss model.

Created on Wed Jun  1 16:02:43 2016

@author: Sebastian Bitzer (sebastian.bitzer@tu-dresden.de)
"""

import numpy as np
import matplotlib.pyplot as plt
import rtmodels

dt = 0.1

N = 2000

# make model
model = rtmodels.discrete_static_gauss(maxrt=2.5, choices=[-1, 1], 
    Trials=(np.random.randint(2, size=N) - 0.5) * 2, 
    means=np.c_[[-25, 0], [25, 0]], intstd=70.)
    
# ignore lapses
model.lapseprob = 0    
    
# set nondecision time distribution with small, but realistic values
model.ndtmean = -0.9
model.ndtspread = 0.41

# noise relatively small
model.noisestd = 70

# simulate from model
choice, rt = model.gen_response(np.arange(N))

plt.figure()
ax = model.plot_response_distribution(choice, rt)


#%% now use collapsing bound to cut off some of the tail
model.bstretch = 0.6

choice, rt = model.gen_response(np.arange(N))

plt.figure()

model.plot_response_distribution(choice, rt)


#%% check the boundfun
T = 20
tratios = np.logspace(np.log10(0.01), 0, T)

def compute_bound(tratios, bound, bstretch, bshape):
    boundvals = np.zeros(tratios.size)
    for (ind, t) in enumerate(tratios):
        boundvals[ind] = rtmodels.discrete_gauss.boundfun(t, bound, bstretch,
            bshape)
            
    return boundvals

bound = 0.8
bstretch = 0.7
    
plt.figure()
plt.plot(tratios, np.c_[compute_bound(tratios, bound, bstretch, 0.7), 
                        compute_bound(tratios, bound, bstretch, 1.4), 
                        compute_bound(tratios, bound, bstretch, 2.8)])
plt.ylim([0.5, 1])