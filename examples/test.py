# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import rtmodels

plt.style.use('ggplot')

choices = np.array([-1, 1])
Trials = choices[np.random.randint(2, size=100)]

model = rtmodels.discrete_static_gauss(maxrt=2.5, choices=choices, Trials=Trials)

trind = np.random.randint(100, size=10000)
choices, rts = model.gen_response_with_params(trind, params={'ndtspread': 0.3})

model.plot_response_distribution(choices, rts)

print('fraction correct = %6.4f' % np.mean(choices == Trials[trind]))