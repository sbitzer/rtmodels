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
from warnings import warn, filterwarnings
from .rtmodel import rtmodel

filterwarnings("always", message='This call to compute the log posterior', 
               category=RuntimeWarning)

class discrete_static_gauss(rtmodel):

    @property
    def D(self):
        """The dimensionality of the features space assumed by the model (read-only)."""
        return self._D
    
    @property
    def use_features(self):
        """Whether the model uses features as input, or just means."""
        if self._Trials.ndim == 1:
            return False
        else:
            return True
    
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
        Trials = np.array(Trials)
        
        if Trials.ndim == 3:
            self._Trials = Trials
            S, D, self._L = Trials.shape
            
            if self.D != D:
                warn('The dimensions of the input features in "Trials" ' + 
                     '(D=%d) do not match those stored in the model (D=%d)' % 
                     (D, self.D), RuntimeWarning)
        elif Trials.ndim == 1:
            # check that Trials only contains valid choice codes
            if np.all(np.in1d(np.unique(Trials), self.choices)):
                # transform to indices into self.choices
                self._Trials = np.array([np.flatnonzero(self.choices == i)[0] 
                                         for i in Trials])
                self._L = len(Trials)
            else:
                raise ValueError('Trials may only contain valid choice codes' +
                                 ' when features are not used.')
        else:
            raise ValueError('Trials has unknown format, please check!')
    
    @property
    def S(self):
        """Number of time steps maximally simulated by the model."""
        if self.Trials.ndim == 3:
            return self.Trials.shape[0]
        else:
            # the + 1 ensures that time outs can be generated
            return int(math.ceil(self.maxrt / self.dt)) + 1
    
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

    @property
    def prior(self):
        "Prior probabilities over choices."
        return self._prior
        
    @prior.setter
    def prior(self, prior):
        if np.isscalar(prior) and self.C == 2:
            self._prior = np.array([prior])
        elif type(prior) is np.ndarray and prior.size == self.C-1:
            self._prior = prior
        else:
            raise TypeError("The prior should be a numpy array with C-1 "
                            "elements! For two choices only you may also "
                            "provide a scalar.")

    prior_re = re.compile('(?:prior)(?:_(\d))?$')
    
    @property
    def P(self):
        "number of parameters in the model"
        
        # the prior adds C-1 parameters, one of which is counted by its name
        return len(self.parnames) + self.C - 2
    
    
    def __init__(self, Trials, dt=1, means=None, prior=None, noisestd=1, 
                 intstd=1, bound=0.8, bstretch=0, bshape=1.4, ndtmean=-12, 
                 ndtspread=0, lapseprob=0.05, lapsetoprob=0.1, **rtmodel_args):
        super(discrete_static_gauss, self).__init__(**rtmodel_args)
            
        self.name = 'Discrete static Gauss model'
        
        # Time resolution of model simulations.
        self.dt = dt
        
        # Trial information used by the model.
        Trials = np.array(Trials)
        
        # try to figure out the dimensionality of feature space from means
        D_mean = None
        if means is not None:
            means = np.array(means)
            if means.ndim == 1:
                D_mean = 1
            else:
                D_mean = means.shape[0]
        
        # check with dimensionality of feature space from Trials
        if Trials.ndim == 1:
            if D_mean is None:
                # no information provided by user
                self._D = 1
            else:
                self._D = D_mean
        elif Trials.ndim == 3:
            if D_mean is None: 
                self._D = Trials.shape[1]
            else:
                if Trials.shape[1] == D_mean:
                    self._D = Trials.shape[1]
                else:
                    raise ValueError("The dimensionality of the provided "
                                     "means and features in Trials is not "
                                     "consistent!")
        
        # now set Trials internally (because dimensionality of feature space is
        # set before, there should be no warnings)
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
            
        # Prior probabilities over choices.
        if prior is None:
            self.prior = np.ones(self.C-1) / self.C
        else:
            self.prior = prior
            
        # Standard deviation of noise added to feature values.
        self.noisestd = noisestd
            
        # Standard deviation of internal uncertainty.
        self.intstd = intstd
            
        # Bound that needs to be reached before decision is made.
        # If collapsing, it's the initial value.
        self.bound = bound
            
        # Extent of collapse for bound, see boundfun.
        self.bstretch = bstretch
        
        # Shape parameter of the collapsing bound, see boundfun
        self.bshape = bshape
            
        # Mean of nondecision time.
        self.ndtmean = ndtmean
            
        # Spread of nondecision time.
        self.ndtspread = ndtspread
            
        # Probability of a lapse.
        self.lapseprob = lapseprob
            
        # Probability that a lapse will be timed out.
        self.lapsetoprob = lapsetoprob
            
    
    def estimate_memory_for_gen_response(self, N):
        """Estimate how much memory you would need to produce the desired responses."""
        
        mbpernum = 8 / 1024 / 1024
        
        # (for input features + for input params + for output responses)
        return mbpernum * N * (self.D * self.S + self.P + 2)
    
    
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
            if 'prior' in pardict:
                if np.isscalar(pardict['prior']):
                    assert self.C == 2, 'prior can only be scalar, if there are only 2 options'
                    pardict['prior'] = np.array([pardict['prior']])
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
        
        NP = max(N, P)
        
        # if continuing would exceed the memory limit
        if self.estimate_memory_for_gen_response(NP) > self.memlim:
            # divide the job in smaller batches and run those
        
            # determine batch size for given memory limit
            NB = int(math.floor(
                    NP / self.estimate_memory_for_gen_response(NP) * self.memlim))
            
            choices = np.zeros(NP, dtype=np.int8)
            rts = np.zeros(NP)
            
            remaining = NP
            firstind = 0
            while remaining > 0:
                index = np.arange(firstind, firstind + min(remaining, NB))
                if P > 1 and N > 1:
                    trind_batch = trind[index]
                    params_batch = extract_param_batch(pardict, index)
                elif N == 1:
                    trind_batch = trind
                    params_batch = extract_param_batch(pardict, index)
                elif P == 1:
                    trind_batch = trind[index]
                    params_batch = pardict
                else:
                    raise RuntimeError("N and P are not consistent.")
                    
                choices[index], rts[index] = self.gen_response_with_params(
                    trind_batch, params_batch, user_code=user_code)
                
                remaining -= NB
                firstind += NB
        else:
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
                elif np.isscalar(allpars[name]) and N >= 1:
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
        
        
    def estimate_memory_for_logpost(self, N):
        """Estimate how much memory you would need to produce the desired responses."""
        
        mbpernum = 8 / 1024 / 1024
        
        # (for input features + for input params + for output responses)
        return mbpernum * N * self.C * self.S * 2
        
        
    def compute_logpost_from_features(self, trind, R=1):
        if self.estimate_memory_for_logpost(trind.size * R) > self.memlim:
            warn("This call to compute the log posterior is likely to exceed "
                 "the current memory limited.", 
                 RuntimeWarning)
        
        # create / select features
        if self.use_features:
            features = self.Trials[:, :, trind]
        else:
            features = self.means[:, self._Trials[trind]]
            features = np.tile(features, (self.S, 1, 1))
        
        log_post = np.zeros((self.S, self.C, trind.size, R))
        log_liks = np.zeros((self.S, self.C, trind.size, R))
        for rep in range(R):
            # add noise to features
            noisy_features = features + np.random.normal(scale=self.noisestd,
                                                         size=features.shape)
        
            # compute log posterior
            log_post[:, :, :, rep], log_liks[:, :, :, rep] = \
                self.accumulate_evidence(noisy_features)
    
        return log_post, log_liks
        
    
    def accumulate_evidence(self, noisy_features):
        # compute log-likelihoods of generative models
        log_liks = np.full((self.S, self.C, noisy_features.shape[2]), np.nan)
        
        for c in range(self.C):
            log_liks[:, c, :] = -0.5 / self.intstd**2 * np.sum(
                (noisy_features - self.means[:, c][None, :, None]) ** 2, axis=1)
        
        # add log-prior to first time point
        log_post = log_liks.copy()
        log_post[0, :-1, :] += np.log(self.prior)[:, None]
        log_post[0, -1, :] += math.log(1 - self.prior.sum())
        
        # accumulate
        log_post = log_post.cumsum(axis=0)
        
        # normalise
        log_post -= logsumexp_3d(log_post, axis=1)
        
        return log_post, log_liks
        
        
    def gen_response_from_logpost(self, log_post, lapses=True, ndt=True):
        """Generates model responses from pre-computed log-posterior over options.
        
            Parameters
            ----------
            log_post : ndarray
                pre-computed log-posterior values over decision alternatives
                (#time steps, #alternatives, #trials, #repetitions) = shape
                the last dimension (repetitions) may be omitted
            lapses : bool, default True
                whether to also produce lapses according to the model's lapse
                probabilities
            ndt : bool, default True
                whether to add non-decision time to computed decision times
                
            Returns
            -------
            choices : ndarray
                computed choice
                (#trials, #repetitions) = shape
            rts : ndarray
                computed response times
                (#trials, #repetitions) = shape
        """
        # pre-compute collapsing bound
        boundvals = np.full(self.S, math.log(self.bound))
        if self.bstretch > 0:
            for t in range(self.S):
                boundvals[t] = math.log( boundfun((t+1.0) / self.maxrt, 
                    self.bound, self.bstretch, self.bshape) )
        
        # add 4th dim when it's missing
        if log_post.ndim == 3:
            log_post = log_post[:, :, :, None]
            
        N, R = log_post.shape[2:]

        rts = np.full((N, R), self.toresponse[1])
        choices = np.full((N, R), self.toresponse[0], dtype=np.int8)

        # loop over all trials and repetitions
        for tr in range(N):
            for rep in range(R):
                # account for lapses
                if lapses and np.random.rand(1) < self.lapseprob:
                    # if response is NOT a timed-out lapse
                    if np.random.rand(1) > self.lapsetoprob:
                        # generate uniformly distributed lapse response
                        rts[tr, rep] = np.random.rand(1) * self.maxrt
                        choices[tr, rep] = self.choices[np.random.randint(
                                self.C)]
                else:
                    # find log_posteriors which cross the bound
                    tind, cind = np.nonzero(log_post[:, :, tr, rep] > 
                                            boundvals[:, None])
                    
                    # if bound was crossed at some point, record time point 
                    # and choice
                    if tind.size > 0:
                        # numpy always goes through the array in C-style order,
                        #  i.e., nonzero elements will be identified within a 
                        # row before moving to the next row. As rows in 
                        # log_post are time points, this ensures that the first
                        # nonzero element will be also the first time point the
                        # bound is crossed
                        rt = (tind[0] + 1) * self.dt
                        
                        # add non-decision time
                        if ndt:
                            rt += random.lognormvariate(self.ndtmean, 
                                                        self.ndtspread)
                            
                        # record response, if time is within maxrt
                        if rt <= self.maxrt:
                            rts[tr, rep] = rt
                            choices[tr, rep] = self.choices[cind[0]]
        
        return choices, rts
        
    
    def compute_surprise(self, log_post, log_liks):
        """ computes -log(p(x_t|X_0:t-1)) = 
            -log( sum_j p(x_t|M_j)p(M_j|X_0:t-1) )
            everything is done in log-space for numerical stability
            
            log_post[t,j] = log(p(M_j|X_0:t))
            log_liks[t,j] = log(p(x_t|M_j)) + log(gauss_Z)
            with log(gauss_Z) = D/2 * log(2pi) + D * log(intstd)
            
            -log(p(x_t|X_0:t-1)) = -log(sum_j p(x_t|M_j)p(M_j|X_0:t-1))
                = -log(sum_j exp( log(p(x_t|M_j)) + log(p(M_j|X_0:t-1)) ))
                = -log(sum_j exp( log_liks[t,j] - log(gauss_Z) + log_post[t-1,j] ))
                = log(gauss_Z) -log(sum_j exp( log_liks[t,j] + log_post[t-1,j] ))
        """
        if log_post.shape != log_liks.shape:
            raise ValueError("log_post and log_liks must have the same shape.")
        
        # add dimensions when they're missing
        input_shape = log_post.shape
        if log_post.ndim == 3:
            log_post = log_post[:, :, :, None]
            log_liks = log_liks[:, :, :, None]
        if log_post.ndim == 2:
            log_post = log_post[:, :, None, None]
            log_liks = log_liks[:, :, None, None]
            
        N, R = log_post.shape[2:]
        
        # get log-prior
        log_prior = np.r_[np.log(self.prior), math.log(1 - self.prior.sum())]
        
        # log of normalisation constant of Gaussian
        log_gauss_Z = ( self.D/2.0 * math.log(2*math.pi) + 
                        self.D * math.log(self.intstd) )
        
        surprise = np.full((self.S, N, R), log_gauss_Z)
        
        # loop over all trials and repetitions
        for tr in range(N):
            for rep in range(R):
                # new array: log posterior for previous data point
                # i.e.: add log-prior to the front and drop last time point
                log_post_previous = np.r_[log_prior[None, :], 
                                          log_post[:-1, :, tr, rep]]
                surprise[:, tr, rep] -= logsumexp_3d((log_liks[:, :, tr, rep] + 
                    log_post_previous)[:, :, None], axis=1).squeeze(axis=(1,2))
        
        # if log_post had no 4th (reps) dimension, remove the corresponding 
        # dimension from surprise, too
        if len(input_shape) == 3:
            surprise = surprise.squeeze(axis=(2,))
        if len(input_shape) == 2:
            surprise = surprise.squeeze(axis=(1,2))
                    
        return surprise
        
    
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
            
    
    def __str__(self):
        info = super(discrete_static_gauss, self).__str__()
        
        # empty line
        info += '\n'
        
        # model-specific parameters
        info += 'means:\n'
        info += self.means.__str__() + '\n'
        info += 'uses features: %4d' % self.use_features + '\n'
        info += 'dt           : %8.3f' % self.dt + '\n'
        info += 'bound        : %8.3f' % self.bound + '\n'
        info += 'bstretch     : %7.2f' % self.bstretch + '\n'
        info += 'bshape       : %7.2f' % self.bshape + '\n'
        info += 'noisestd     : %6.1f' % self.noisestd + '\n'
        info += 'intstd       : %6.1f' % self.intstd + '\n'
        info += 'ndtmean      : %7.2f' % self.ndtmean + '\n'
        info += 'ndtspread    : %7.2f' % self.ndtspread + '\n'
        info += 'lapseprob    : %7.2f' % self.lapseprob + '\n'
        info += 'lapsetoprob  : %7.2f' % self.lapsetoprob + '\n'
        info += 'prior        : ' + ', '.join(map(lambda s: '{:8.3f}'.format(s), 
                                                  self.prior)) + '\n'
        
        return info
        

class sensory_discrete_static_gauss(discrete_static_gauss):
    parnames = ['bound', 'bstretch', 'bshape', 'noisestd', 'intstd', 
                'sensdrift', 'prior', 'ndtmean', 'ndtspread', 'lapseprob', 
                'lapsetoprob']

    def __init__(self, Trials, sensdrift=1.0, **dsg_args):
        super(sensory_discrete_static_gauss, self).__init__(Trials, **dsg_args)
            
        self.name = 'Sensory discrete static Gauss model'
        
        if self.C != 2:
            raise ValueError("The sensory discrete static gauss model is " + 
                             "currently only implemented for 2 alternatives.")
        
        "Drift of sensory accumulator"
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
        
    
    def compute_logpost_from_features(self, trind, R=1):
        raise NotImplementedError()
        
    def accumulate_evidence(self, noisy_features):
        raise NotImplementedError()
        
    def gen_response_from_logpost(self, log_post):
        raise NotImplementedError()
        
    def compute_surprise(self, log_post, log_liks):
        raise NotImplementedError()
        
        
    def __str__(self):
        info = super(sensory_discrete_static_gauss, self).__str__()
        
        info += 'sensdrift    : %7.2f' % self.sensdrift + '\n'
        
        return info
        

class extended_discrete_static_gauss(discrete_static_gauss):
    """Extended static Gauss model"""
    
    parnames = ['bound', 'noisestd', 'etaN', 'coupling', 'intstd', 'kappa', 
                'prior', 'sP', 'ndtmean', 'ndtspread', 'lapseprob', 
                'lapsetoprob']

    def __init__(self, Trials, ndtmean=0.0, sP=0.0, etaN=0.0, coupling=True, 
                 kappa=0.0, **dsg_args):
        super(extended_discrete_static_gauss, self).__init__(Trials, 
            ndtmean=ndtmean, **dsg_args)
                
        self.name = 'Extended static Gauss model'
        
        self.coupling = coupling
        
        self.etaN=etaN
        
        self.kappa=kappa
        
        self.sP = sP
         
    
    # overwrite method of discrete_static_gauss to skip directly to basic 
    # implementation of rtmodel
    def plot_parameter_distribution(self, samples, names, q_lower=0, q_upper=1):
        super(discrete_static_gauss, self).plot_parameter_distribution(
            samples, names, q_lower, q_upper)
    
    
    def gen_response_jitted(self, features, allpars, changing_bound=False):
        toresponse_intern = np.r_[-1, self.toresponse[1]]
            
        # call the compiled function
        choices, rts = gen_response_jitted_edsg(features, self.maxrt, toresponse_intern, 
            self.choices, self.dt, self.means, allpars['prior'], allpars['sP'],
            allpars['noisestd'], allpars['etaN'], allpars['coupling'], 
            allpars['intstd'], allpars['kappa'], allpars['bound'], 
            allpars['ndtmean'], allpars['ndtspread'], allpars['lapseprob'], 
            allpars['lapsetoprob'])
        
        return choices, rts
        
        
    def compute_logpost_from_features(self, trind, R=1):
        raise NotImplementedError()
        
    def accumulate_evidence(self, noisy_features):
        raise NotImplementedError()
        
    def gen_response_from_logpost(self, log_post):
        raise NotImplementedError()
        
    def compute_surprise(self, log_post, log_liks):
        raise NotImplementedError()
        

    def __str__(self):
        info = super(extended_discrete_static_gauss, self).__str__()
        
        info += 'coupling     : %4d' % self.coupling + '\n'
        info += 'etaN         : %9.4f' % self.etaN + '\n'
        info += 'kappa        : %9.4f' % self.kappa + '\n'
        info += 'sP           : %7.2f' % self.sP + '\n'
        
        return info
        

@jit(nopython=True, cache=False)
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
                        
                # normalise
                logpost = logev - logsumexp(logev)
                
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
def gen_response_jitted_edsg(features, maxrt, toresponse, choices, dt, means,
    prior, sP, noisestd, etaN, coupling, intstd, kappa, bound, ndtmean, 
    ndtspread, lapseprob, lapsetoprob):
    
    C = len(choices)
    assert C == 2, "extended discrete static gauss model only allows 2 choices"
    
    S, D, N = features.shape
    
    meandist = 0.0
    for d in range(D):
        meandist += (means[d, 0] - means[d, 1]) ** 2
    meandist = math.sqrt(meandist)
    
    choices_out = np.full(N, toresponse[0], dtype=np.int8)
    rts = np.full(N, toresponse[1])
    
    for tr in range(N):
       
        if(etaN[tr]!=0):
                
                zetaN=1/etaN[tr]
                
                #sample the noisestd values from the inverse Gaussian distribtuion
                noisestdTrVal=np.random.wald(noisestd[tr],zetaN)
    
                negProp=(1+math.erf(-math.sqrt(zetaN/(2*noisestd[tr]))))/2
        
                #calculate the proportion of flipped trial and generate the respective vector
                #indicating the features from which trial should be flipped
                negNoiseTrial=np.random.binomial(1,1-negProp)

        else:
                noisestdTrVal=noisestd[tr]
                negNoiseTrial=np.sign(noisestdTrVal)
                
        if negNoiseTrial==0:
            negNoiseTrial=-1
            
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
            boundtr = math.log(bound[tr])
            
            #sample the prior values from the uniform distribution
            priorTrVal=np.random.uniform(prior[tr, 0] - sP[tr] / 2,
                                         prior[tr, 0] + sP[tr] / 2)
            
            logev = np.zeros(C)
            logev[0] = np.log(priorTrVal)
            logev[1] = np.log(1 - priorTrVal)

            if coupling[tr]:
                intvartr = noisestdTrVal / (dt * 2.8) * meandist
            elif kappa[tr] == 0:
                intvartr = intstd[tr] ** 2
            else:
                intvartr = np.random.wald(intstd[tr], 1/kappa[tr]) ** 2

            # for all presented features
            exitflag = False
            for t in range(S):
                # add noise to feature
                noisy_feature = np.zeros(D)
                # Perform the flipping for the trials indicated by negprop and binomial sampled negNoiseTrials
                
                for d in range(D):
                    noisy_feature[d] = negNoiseTrial*features[t,d,tr] + random.gauss(
                        0, noisestdTrVal*math.sqrt(dt))
                
                # compute log-likelihoods of internal generative models
                for c in range(C):
                    # I used numpy sum and power functions here before, but
                    # this didn't compile with numba
                    sum_sq = 0

                    for d in range(D):
                        sum_sq += (noisy_feature[d] - means[d, c]) ** 2
                    
                    #calculate the intstd based on the value for noisestdTrVals
#                    intvarTrVal = (2 * noisestdTrVal) / (dt * 0.1)
                    logev[c] += -1 / (2 * dt*intvartr) * sum_sq
                        
                # normalise
                logpost = logev - logsumexp(logev)
                
                for c in range(C):
                    if logpost[c] >= boundtr:
                        choices_out[tr] = c
                        # add 1 to t because t starts from 0
                        NdtTr = np.random.uniform(ndtmean[tr] - ndtspread[tr] / 2, 
                                                  ndtmean[tr] + ndtspread[tr] / 2)
                        rts[tr] = (t+1) * dt + NdtTr
                        exitflag = True
                        break
                    
                if exitflag:
                    break
           
            if rts[tr] > maxrt:
                choices_out[tr] = toresponse[0]
                rts[tr] = toresponse[1]
    
    return choices_out, rts


@jit(nopython=True, cache=True)
def logsumexp_3d(logvals, axis=0):
    shape = logvals.shape
    
    assert len(shape) == 3, 'logvals in logsumexp_3d has to be 3d!'
    
    if axis == 0:
        logsum = np.zeros((1, shape[1], shape[2]))
        for i1 in range(shape[1]):
            for i2 in range(shape[2]):
                logsum[0, i1, i2] = logsumexp(
                    logvals[:, i1, i2])
    elif axis == 1:
        logsum = np.zeros((shape[0], 1, shape[2]))
        for i0 in range(shape[0]):
            for i2 in range(shape[2]):
                logsum[i0, 0, i2] = logsumexp(
                    logvals[i0, :, i2])
    elif axis == 2:
        logsum = np.zeros((shape[0], shape[1], 1))
        for i0 in range(shape[0]):
            for i1 in range(shape[1]):
                logsum[i0, i1, 0] = logsumexp(
                    logvals[i0, i1, :])
    else:
        raise ValueError("Argument 'axis' has illegal value in "
                         "logsumexp_3d!")
    
    return logsum


@jit(nopython=True, cache=True)
def logsumexp(logvals):
    mlogv = logvals.max()
    
    # bsxfun( @plus, mlogp, log( sum( exp( bsxfun(@minus, logp, mlogp) ) ) ) );
    logsum = mlogv + np.log( np.sum( np.exp(logvals - mlogv) ) )
    
    return logsum
    

@jit(nopython=True, cache=True)
def boundfun(tfrac, bound, bstretch, bshape):
    tshape = tfrac ** -bshape
    
    return 0.5 + (1 - bstretch) * (bound - 0.5) - ( bstretch * (bound - 0.5) *
        (1 - tshape) / (1 + tshape) );
        

def extract_param_batch(pardict, index):
    newdict = {}
    for parname, values in pardict.items():
        if values.ndim == 2:
            newdict[parname] = values[index, :]
        else:
            newdict[parname] = values[index]
            
    return newdict