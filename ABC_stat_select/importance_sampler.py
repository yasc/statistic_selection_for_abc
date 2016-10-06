#!/usr/bin/env python
"""Generate importance sampling distribution"""

import numpy as np
import time
from scipy import stats
from multiprocessing import pool
from os import getpid

import simulate_data as sim
import summary_stats as sum_stat
import selection as select

__author__ = "yasc"
__date_created__ = "12 July 2016"


#@profile  
def lin_reg_importance_sampler(delta, obs_stats, s):
    n = select.sample_dim[1]
    """Generate an importance sampling distribution based on the observed data
    and return the distribution's sample mean and sample standard deviation"""

    theta = np.array(select.theta_initial.T)
    sim_stats = np.array([select.stats_initial[i][delta > 0]
                          for i in range(len(select.stats_initial))])

    perturb_sample_size = round(0.2*s)
    j = 0
    while len(theta) > .005*s:
        j += 1
        params = np.zeros([1, 2+n])
        stats = np.zeros([1, len(sim_stats[0])]) #Potentially problematic to ascertain second dimension in this manner.
        norms = [np.linalg.norm(obs_stats-sim_stats[i])
                 for i in range(len(sim_stats))]
        top_20 = np.argsort(norms)[0:int(round(0.2*len(sim_stats)))]
        params = np.vstack((params, theta[top_20]))
        stats = np.vstack((stats, [sim_stats[i] for i in top_20]))
        samp = np.around(np.random.rand(
            1, len(top_20))*(len(top_20)-1))
        pre_perturb = params[1:len(params)][samp.astype(int)]
        post_perturb = pre_perturb+np.random.multivariate_normal(
            np.zeros(n+2), np.eye(n+2), len(top_20))
        sim_x = np.matrix(np.random.randn(select.sample_dim[0],
                                          (len(top_20))*n))
        sim_y = [sim.sim_lin_reg(np.matrix(post_perturb).T[0:n+1, i],
                                 np.matrix(post_perturb).T[n+1, i],
                                 sim_x[:, i*n:(i+1)*n], 1)
                 for i in range(len(top_20))]
        stats_perturbed = [sum_stat.lin_reg_stats(delta, sim_y[i],
                                                  sim_x[:, i*n:(i+1)*n])
                           for i in range(len(top_20))]
        theta = np.delete(np.vstack((params, post_perturb.squeeze())), 0, 0)
        sim_stats = np.delete(np.vstack((stats, stats_perturbed)), 0, 0)
        mean = sum(theta)/(len(theta))
        sigma = np.sqrt(1./len(theta))
    return [mean, sigma]

#####TEST#####
##/home/yasc/Programs/anaconda3/envs/dissertation/lib/python3.5/site-packages/kernprof.py

#obs_x_30 = np.matrix(np.random.randn(30,2))
#obs_y_30 = sim.sim_lin_reg(np.matrix([1,1,1]).T,1,obs_x_30,1)
#delta = np.array([0]*18)
#delta[[0,1,2,3]] = 1
#x = lin_reg_importance_sampler(delta, obs_y_30, obs_x_30, 10**4)


def isam_w(theta, g_mean, g_sigma, p_dist="uniform",
           g_dist="normal", b_high=2, b_low=-2, s_high=5, s_low=0):

    b_range = b_high - b_low
    s_range = s_high - s_low
    b_density = 1./(b_range**(len(theta)-1))
    s_density = 1./s_range
    numerator = b_density*s_density

    denominator = stats.multivariate_normal.pdf(
        theta, g_mean, np.eye(
            len(theta))*g_sigma)

    return numerator/denominator
