# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:06:05 2016

@author: Sebastian Bitzer (sebastian.bitzer@tu-dresden.de)
"""

import re
import math
import random
import numpy as np
from numba import jit
from warnings import warn
from .rtmodel import rtmodel

class discrete_static_gauss(rtmodel):

    "Whether the model uses time-varying features for decision making."
    use_features = True
    
    _D = None
    @property
    def D(self):
        """The dimensionality of the features space assumed by the model."""
        return self._D
    
    _dt = 0.1
    @property
    def dt(self):
        """Time resolution of model simulations."""
        return self._dt
        
    @dt.setter
    def dt(self, dt):
        self._dt = dt
    
    _Time = None
    
    _Trials = None
    @property
    def Trials(self):
        """Trial information used by the model.
        
        Either a 1D (use_features=False), or 3D (use_features=True) numpy array.
        When 1D, Trials contains the code of the correct choice in that trial.
        When 3D, Trials contains the stream of feature values that the subject
        may have seen in all the trials of the experiment. Then,
            S, D, L = Tials.shape
        where S is the length of the sequence in each trial
        """
        if self.use_features:
            return self._Trials
        else:
            return self.choices[self._Trials]
        
    @Trials.setter
    def Trials(self, Trials):
        if self.use_features:
            self._Trials = Trials
            S, D, self._L = Trials.shape
            
            if self.D != D:
                warn('The dimensions of the input features in "Trials" ' + 
                     '(D=%d) do not match those stored in the model (D=%d)' % 
                     (D, self.D), RuntimeWarning)
        else:
            # check that Trials only contains valid choice codes
            if np.all(np.in1d(np.unique(Trials), self.choices)):
                # transform to indices into self.choices
                self._Trials = np.array([np.flatnonzero(self.choices == i)[0] 
                                         for i in Trials])
                self._L = len(Trials)
            else:
                raise ValueError('Trials may only contain valid choice codes' + 
                                 ' when features are not used.')
    
    @property
    def S(self):
        """Number of time steps maximally simulated by the model."""
        if len(self.Trials.shape) == 3:
            return self.Trials.shape[0]
        else:
            # the + 1 ensures that time outs can be generated
            return math.ceil(self.maxrt / self.dt) + 1
    
    _means = None
    @property
    def means(self):
        """Mean features values assumed in model."""
        return self._means
    
    @means.setter
    def means(self, means):
        if means.ndim == 1:
            means = means[None, :]
            
        D, C = means.shape
        
        if C != self.C:
            raise ValueError('The number of given means (%d) ' % C + 
                             'does not match the number of choices (%d) ' % self.C +
                             'processed by the model')
        else:
            self._means = means
            if self._D != D and self.use_features:
                warn('The given means changed the dimensionality of the ' + 
                     'model. Update Trials!')
            self._D = D

    parnames = ['bound', 'bstretch', 'bshape', 'noisestd', 'intstd', 'prior', 
                'ndtmean', 'ndtspread', 'lapseprob', 'lapsetoprob']

    """Bound that needs to be reached before decision is made.
       If collapsing, it's the initial value."""
    bound = 0.8
    
    "Extent of collapse for bound, see boundfun."
    bstretch = 0
    
    "Shape parameter of the collapsing bound, see boundfun"
    bshape = 1.4
    
    "Standard deviation of noise added to feature values."
    noisestd = 1
    
    "Standard deviation of internal uncertainty."
    intstd = 1

    "Prior probabilities over choices."
    prior = None
    
    "Mean of nondecision time."
    ndtmean = 0
    
    "Spread of nondecision time."
    ndtspread = 0
    
    "Probability of a lapse."
    lapseprob = 0.05
    
    "Probability that a lapse will be timed out."
    lapsetoprob = 0.1
    
    prior_re = re.compile('(?:prior)(?:_(\d))?$')
    
    def __init__(self, use_features=None, Trials=None, dt=None, means=None, 
                 prior=None, noisestd=None, intstd=None, bound=None, 
                 bstretch=None, bshape=None, ndtmean=None, 
                 ndtspread=None, lapseprob=None, lapsetoprob=None,
                 choices=None, maxrt=None, toresponse=None):
        super(discrete_static_gauss, self).__init__(choices, maxrt, 
            toresponse)
            
        if dt is not None:
            self.dt = dt
            
        if use_features is not None:
            if Trials is not None:
                warn('use_features and Trials are given to construct model. ' +
                     'The value of use_features will be determined from ' +
                     'Trials and the given value will be discarded!', 
                     RuntimeWarning)
            else:
                self.use_features = use_features
            
        if Trials is not None:
            if type(Trials) is list:
                Trials = np.array(Trials)
                
            dim = len(Trials.shape)
            if dim == 1:
                self.use_features = False
                self._D = 1
            elif dim == 3:
                self.use_features = True
                self._D = Trials.shape[1]
                
            self.Trials = Trials
        
        if means is not None:
            self.means = means
        elif self.D == 1:
            self._means = np.arange(self.C)
            self._means = self._means - np.mean(self._means)
            self._means = self._means[None, :]
        elif self.D == 2:
            phi = math.pi/4 + np.linspace(0, 2*math.pi * (1-1/self.C), self.C)
            self._means = np.r_[np.cos(phi), np.sin(phi)];
        else:
            warn('Cannot set default means. Please provide means yourself!',
                 RuntimeWarning)
                 
        if prior is not None:
            self.prior = prior
        else:
            self.prior = np.ones(self.C-1) / self.C
            
        if noisestd is not None:
            self.noisestd = noisestd
            
        if intstd is not None:
            self.intstd = intstd
            
        if bound is not None:
            self.bound = bound
            
        if bstretch is not None:
            self.bstretch = bstretch
        
        if bshape is not None:
            self.bshape = bshape
            
        if ndtmean is not None:
            self.ndtmean = ndtmean
            
        if ndtspread is not None:
            self.ndtspread = ndtspread
            
        if lapseprob is not None:
            self.lapseprob = lapseprob
            
        if lapsetoprob is not None:
            self.lapsetoprob = lapsetoprob
            
    
    def gen_response(self, trind, rep=1):
        N = trind.size
        if rep > 1:
            trind = np.tile(trind, rep)
        
        choices, rts = self.gen_response_with_params(trind)
        
        if rep > 1:
            choices = choices.reshape((rep, N))
            rts = rts.reshape((rep, N))
        
        return choices, rts
        
        
    def gen_response_with_params(self, trind, params={}, parnames=None, 
                                 user_code=True):
        if parnames is None:
            assert( type(params) is dict )
            pardict = params
        else:
            assert( type(params) is np.ndarray )
            pardict = {}
            new_prior = True
            for ind, name in enumerate(parnames):
                match = self.prior_re.match(name)
                if match is None:
                    pardict[name] = params[:, ind]
                else:
                    ind_prior = match.groups()[0]
                    if ind_prior is None:
                        pardict['prior'] = params[:, ind]
                        pardict['prior'] = pardict['prior'][:, None]
                    else:
                        if new_prior:
                            pardict['prior'] = np.full((params.shape[0], 
                                                        self.C-1), np.nan)
                            new_prior = False
                        pardict['prior'][:, int(ind_prior)] = params[:, ind]
        parnames = pardict.keys()
        
        # check whether any of the bound parameters are given
        if any(x in ['bound', 'bstretch', 'bshape'] for x in parnames):
            changing_bound = True
        else:
            changing_bound = False
        
        # get the number of different parameter sets, P, and check whether the 
        # given parameter counts are consistent (all have the same P)
        P = None
        for name in parnames:
            if not np.isscalar(pardict[name]):
                if P is None:
                    P = pardict[name].shape[0]
                else:
                    if P != pardict[name].shape[0]:
                        raise ValueError('The given parameter dictionary ' +
                            'contains inconsistent parameter counts')
        if P is None:
            P = 1
        
        # get the number of trials, N, and check whether it is consistent with 
        # the number of parameters P
        if np.isscalar(trind):
            trind = np.full(P, trind, dtype=int)
        N = trind.shape[0]
        if P > 1 and N > 1 and N != P:
            raise ValueError('The number of trials in trind and the ' +
                             'number of parameters in params does not ' + 
                             'fit together')
        
        # make a complete parameter dictionary with all parameters
        # this is quite a bit of waste of memory and should probably be recoded
        # more sensibly in the future, but for now it makes the jitted function
        # simple
        allpars = {}
        for name in self.parnames:
            if name in parnames:
                allpars[name] = pardict[name]
            else:
                allpars[name] = getattr(self, name)
                
            if name == 'prior':
                if allpars[name].ndim == 1:
                    allpars[name] = np.tile(allpars[name], (N,1))
                elif allpars[name].shape[0] == 1 and N > 1:
                    allpars[name] = np.tile(allpars[name], (N,1))
            elif np.isscalar(allpars[name]) and N > 1:
                allpars[name] = np.full(N, allpars[name], dtype=float)
            elif allpars[name].shape[0] == 1 and N > 1:
                allpars[name] = np.full(N, allpars[name], dtype=float)
        
        # select input features
        if self.use_features:
            features = self.Trials[:, :, trind]
        else:
            features = self.means[:, self._Trials[trind]]
            features = np.tile(features, (self.S, 1, 1))
            
        # call the compiled function
        choices, rts = self.gen_response_jitted(features, allpars, 
                                                changing_bound)
            
        # transform choices to those expected by user, if necessary
        if user_code:
            toresponse_intern = np.r_[-1, self.toresponse[1]]
            timed_out = choices == toresponse_intern[0]
            choices[timed_out] = self.toresponse[0]
            in_time = np.logical_not(timed_out)
            choices[in_time] = self.choices[choices[in_time]]
            
        return choices, rts
        
        
    def gen_response_jitted(self, features, allpars, changing_bound):
        toresponse_intern = np.r_[-1, self.toresponse[1]]
            
        # call the compiled function
        choices, rts = gen_response_jitted_dsg(features, self.maxrt, toresponse_intern, 
            self.choices, self.dt, self.means, allpars['prior'], allpars['noisestd'], 
            allpars['intstd'], allpars['bound'], allpars['bstretch'], 
            allpars['bshape'], allpars['ndtmean'], allpars['ndtspread'], 
            allpars['lapseprob'], allpars['lapsetoprob'], changing_bound)
            
        return choices, rts
        
    
    def plot_parameter_distribution(self, samples, names, q_lower=0, q_upper=1):
        if 'ndtmean' in samples.columns and 'ndtspread' in samples.columns:
            samples = samples.copy()
            samples['ndtmode'] = np.exp(samples['ndtmean'] - samples['ndtspread']**2)
            samples['ndtstd'] = np.sqrt( (np.exp(samples['ndtspread']**2) - 1) * 
                np.exp(2 * samples['ndtmean'] + samples['ndtspread']**2) )
            samples.drop(['ndtmean', 'ndtspread'], axis=1, inplace=True)
            
            names = names.copy()
            names.remove('ndtmean')
            names.remove('ndtspread')
            names.append('ndtmode')
            names.append('ndtstd')
            
        super(discrete_static_gauss, self).plot_parameter_distribution(
            samples, names, q_lower, q_upper)


class sensory_discrete_static_gauss(discrete_static_gauss):
    "Drift of sensory accumulator"
    sensdrift = 1.0
    
    parnames = ['bound', 'bstretch', 'bshape', 'noisestd', 'intstd', 
                'sensdrift', 'prior', 'ndtmean', 'ndtspread', 'lapseprob', 
                'lapsetoprob']

    def __init__(self, use_features=None, Trials=None, dt=None, means=None, 
                 prior=None, noisestd=None, sensdrift=None, intstd=None, 
                 bound=None, bstretch=None, bshape=None, ndtmean=None, 
                 ndtspread=None, lapseprob=None, lapsetoprob=None,
                 choices=None, maxrt=None, toresponse=None):
        super(sensory_discrete_static_gauss, self).__init__(use_features, 
            Trials, dt, means, prior, noisestd, intstd, bound, bstretch, 
            bshape, ndtmean, ndtspread, lapseprob, lapsetoprob, choices, maxrt, 
            toresponse)
        
        if self.C != 2:
            raise ValueError("The sensory discrete static gauss model is " + 
                             "currently only implemented for 2 alternatives.")
        
        if sensdrift is not None:
            self.sensdrift = sensdrift

            
    def gen_response_jitted(self, features, allpars, changing_bound):
        toresponse_intern = np.r_[-1, self.toresponse[1]]
            
        # call the compiled function
        choices, rts = gen_response_jitted_sdsg(features, self.maxrt, toresponse_intern, 
            self.choices, self.dt, self.means, allpars['prior'], allpars['noisestd'], 
            allpars['intstd'], allpars['sensdrift'], allpars['bound'], 
            allpars['bstretch'], allpars['bshape'], allpars['ndtmean'], 
            allpars['ndtspread'], allpars['lapseprob'], allpars['lapsetoprob'], 
            changing_bound)
        
        return choices, rts
        

@jit(nopython=True, cache=True)
def gen_response_jitted_dsg(features, maxrt, toresponse, choices, dt, means,
    prior, noisestd, intstd, bound, bstretch, bshape, ndtmean, ndtspread, 
    lapseprob, lapsetoprob, changing_bound):
    
    C = len(choices)
    S, D, N = features.shape
    
    choices_out = np.full(N, toresponse[0], dtype=np.int8)
    rts = np.full(N, toresponse[1])
    
    # pre-compute collapsing bound
    boundvals = np.full(S, math.log(bound[0]))
    if bstretch[0] > 0:
        for t in range(S):
            boundvals[t] = math.log( boundfun((t+1.0) / maxrt, bound[0], 
                bstretch[0], bshape[0]) )
    
    for tr in range(N):
        # is it a lapse trial?
        if random.random() < lapseprob[tr]:
            # is it a timed-out lapse trial?
            if random.random() < lapsetoprob[tr]:
                choices_out[tr] = toresponse[0]
                rts[tr] = toresponse[1]
            else:
                choices_out[tr] = random.randint(0, C-1)
                rts[tr] = random.random() * maxrt
        else:
            logev = np.zeros(C)
            logev[:C-1] = np.log(prior[tr, :])
            logev[-1] = math.log(1 - prior[tr, :].sum())
            
            # for all presented features
            exitflag = False
            for t in range(S):
                # get current bound value
                if changing_bound:
                    # need to compute boundval from parameters in this trial
                    if bstretch[tr] == 0:
                        boundval = math.log(bound[tr])
                    else:
                        boundval = math.log( boundfun((t+1.0) / maxrt, 
                            bound[tr], bstretch[tr], bshape[tr]) )
                else:
                    # can use pre-computed bound value
                    boundval = boundvals[t]
                
                # add noise to feature
                noisy_feature = np.zeros(D)
                for d in range(D):
                    noisy_feature[d] = features[t, d, tr] + random.gauss(
                        0, noisestd[tr])
                
                # compute log-likelihoods of internal generative models
                for c in range(C):
                    # I used numpy sum and power functions here before, but
                    # this didn't compile with numba
                    sum_sq = 0
                    for d in range(D):
                        sum_sq += (noisy_feature[d] - means[d, c]) ** 2
                    logev[c] += -1 / (2 * intstd[tr]**2) * sum_sq
                        
                logpost = normaliselogprob(logev)
                
                for c in range(C):
                    if logpost[c] >= boundval:
                        choices_out[tr] = c
                        # add 1 to t because t starts from 0
                        rts[tr] = (t+1) * dt + random.lognormvariate(
                            ndtmean[tr], ndtspread[tr])
                        exitflag = True
                        break
                    
                if exitflag:
                    break
                
            if rts[tr] > maxrt:
                choices_out[tr] = toresponse[0]
                rts[tr] = toresponse[1]
    
    return choices_out, rts
    
    
@jit(nopython=True, cache=True)
def gen_response_jitted_sdsg(features, maxrt, toresponse, choices, dt, means,
    prior, noisestd, intstd, sensdrift, bound, bstretch, bshape, ndtmean, 
    ndtspread, lapseprob, lapsetoprob, changing_bound):
    
    C = len(choices)
    S, D, N = features.shape
    
    choices_out = np.full(N, toresponse[0], dtype=np.int8)
    rts = np.full(N, toresponse[1])
    
    sqrtdt = math.sqrt(dt)
    
    # pre-compute collapsing bound (in DDM, i.e., loglik space)
    boundvals = np.full(S, math.log(bound[0] / (1-bound[0])))
    if bstretch[0] > 0:
        for t in range(S):
            bval = boundfun((t+1.0) / maxrt, bound[0], bstretch[0], bshape[0])
            boundvals[t] = math.log( bval / (1 - bval) )
    
    for tr in range(N):
        # is it a lapse trial?
        if random.random() < lapseprob[tr]:
            # is it a timed-out lapse trial?
            if random.random() < lapsetoprob[tr]:
                choices_out[tr] = toresponse[0]
                rts[tr] = toresponse[1]
            else:
                choices_out[tr] = random.randint(0, C-1)
                rts[tr] = random.random() * maxrt
        else:
            # get DDM starting point from prior (bias for log-likelihood ratio)
            y = math.log(prior[tr, 0] / (1 - prior[tr, 0]))
            
            # for all presented features
            for t in range(S):
                # compute sum of squares between features and means
                sum_sq = np.zeros(C)
                for c in range(C):
                    for d in range(D):
                        sum_sq[c] += (features[t, d, tr] - means[d, c]) ** 2
                        
                # compute the log-likelihood-ratio for this feature
                llr = (sum_sq[1] - sum_sq[0]) / (2 * intstd[tr] ** 2)
                
                # determine the time in which sensdrift applies
                ts = min(dt, math.fabs(llr) / sensdrift[tr])
                
                # increment the accumulating llr by the end state of the 
                # sensory Gaussian accumulator
                if llr > 0:
                    y += random.gauss(sensdrift[tr] * ts, noisestd[tr] * sqrtdt)
                else:
                    y += random.gauss(-sensdrift[tr] * ts, noisestd[tr] * sqrtdt)
            
                # get current bound value
                if changing_bound:
                    # need to compute boundval from parameters in this trial
                    if bstretch[tr] == 0:
                        boundval = math.log(bound[tr] / (1 - bound[tr]))
                    else:
                        bval = boundfun((t+1.0) / maxrt, bound[tr], 
                                        bstretch[tr], bshape[tr])
                        boundval = math.log( bval / (1 - bval) )
                else:
                    # can use pre-computed bound value
                    boundval = boundvals[t]
            
                if math.fabs(y) >= boundval:
                    if y > 0:
                        choices_out[tr] = 0
                    else:
                        choices_out[tr] = 1
                    rts[tr] = (t+1) * dt + random.lognormvariate(
                        ndtmean[tr], ndtspread[tr])
                    
                    break
                
            if rts[tr] > maxrt:
                choices_out[tr] = toresponse[0]
                rts[tr] = toresponse[1]
                    
    return choices_out, rts
    
    
@jit(nopython=True, cache=True)
def normaliselogprob(logvals):
    mlogv = logvals.max()
    
    # bsxfun( @plus, mlogp, log( sum( exp( bsxfun(@minus, logp, mlogp) ) ) ) );
    logsum = mlogv + np.log( np.sum( np.exp(logvals - mlogv) ) )
    
    return logvals - logsum
    

@jit(nopython=True, cache=True)
def boundfun(tfrac, bound, bstretch, bshape):
    tshape = tfrac ** -bshape
    
    return 0.5 + (1 - bstretch) * (bound - 0.5) - ( bstretch * (bound - 0.5) *
        (1 - tshape) / (1 + tshape) );