#!usr/bin/env python
"""Main function calls"""

import numpy as np
from scipy import stats
import imp

import summary_stats as sum_stat
import simulate_data as sim
import importance_sampler as isam
import estimators as estim
import selection as select
import assess_sv_estimation as a

__author__ = "yasc"
__date_created__ = "05 July 2016"


obs_x = np.matrix(np.random.randn(30, 2))
obs_y = sim.sim_lin_reg(np.matrix([1, 1, 1]), 0.5, obs_x, 1)

# Integrated expected Bayesian loss (EBL)
#expected_Bayes_loss = select.select_lin_reg(obs_y, obs_x, 1000, 100, 1000,
                                            #'abs_dev_knn')

# Local expected Bayes loss (EBL)
#expected_Bayes_loss = select.select_lin_reg(obs_y, obs_x, 1000, 100, 1000,
                                            #'local_abs_dev')

# Integrated Kullback-Leibler divergence (KLD)
#expected_Bayes_loss = select.select_lin_reg(obs_y, obs_x, 1000, 100, 1000,
                                            #'integrated_kl')

# Local Kullback-Leibler divergence (KLD)
expected_Bayes_loss = select.select_lin_reg(obs_y, obs_x, 3000, 3000, 3000,
                                            'local_kl')
