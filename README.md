# rtmodels
A collection of generative models of response time distributions.

Focus is on providing efficient methods for sampling single responses given a collection 
of model parameter values so that probabilistic inference for the model can be conducted
using simulation via [pyEPABC](https://github.com/sbitzer/pyEPABC). To achieve this the 
core functionality of a working model is implemented with JIT-compilation using 
[numba](https://numba.pydata.org/).

Currently only some variants of the model reported in
- Bitzer, S.; Park, H.; Blankenburg, F. & Kiebel, S. J. Perceptual decision making: 
  Drift-diffusion model is equivalent to a Bayesian model Frontiers in Human Neuroscience, 
  2014, 8, https://doi.org/10.3389/fnhum.2014.00102
- Park, H.; Lueckmann, J.-M.; von Kriegstein, K.; Bitzer, S. & Kiebel, S. J. Spatiotemporal 
  dynamics of random stimuli account for trial-to-trial variability in perceptual decision 
  making. Sci Rep, 2016, 6, 18832, https://doi.org/10.1038/srep18832

are implemented.

For a usage example see [examples/fit_with_EPABC.py](https://github.com/sbitzer/rtmodels/blob/master/examples/fit_with_EPABC.py).
