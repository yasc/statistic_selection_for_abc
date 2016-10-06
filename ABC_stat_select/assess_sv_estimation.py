#!/usr/bin/env python
"""Procedures to assess the performance of stochastic volatility estimators"""

import numpy as np

import selection as select
import simulate_data as sim
import summary_stats as sum_stat
import estimators as estim
from scipy import stats
from sklearn import preprocessing as p

__author__ = "yasc"
__date_created__ = "05 July 2016"

theta_sigma = []


def estimation_performance_assessment(theta, delta, obs_stats, t, ar, s, r):
    """Print the RMSE and bias of a given subset of statistics"""

    sbil_kernel_estimates = [sbil_kernel(delta, obs_stats, t, ar, s)
                             for i in range(r)]

    sbil_kernel_estimates = np.array(sbil_kernel_estimates)

    rmse = np.sqrt(sum([np.square(np.matrix(theta) -
                                  sbil_kernel_estimates[i])
                        for i in range(r)])/r)

    bias = np.mean(sbil_kernel_estimates, axis=0) - theta
    print(np.mean(sbil_kernel_estimates, axis=0))

    print("RMSE:", rmse)
    print("Bias:", bias)
    print("Sum of RMSE:", np.sum(rmse))

    return 1


def sbil_kernel(delta, obs_stats, t, ar, s, kernel='Gaussian'):
    """Standardize parameters as well as statistics and return
    a k-nn estimate"""
    #np.random.shuffle(delta)
    print(delta)
    sbil_kernel_estimate = []
    obs_stats = obs_stats[delta > 0]

    sim_theta = [select.generate_theta_sv(ar) for i in range(s)]
    sim_theta = np.matrix(sim_theta).T

    # Generate out sample of time series.
    sim_y = [sim.sim_sv(t, sim_theta[0, i], sim_theta[1, i], sim_theta[2, i],
                        sim_theta[3, i], 1) for i in range(s)]
    
    # Generate out sample statistics.
    sim_stats = [sum_stat.sv_stats(delta, sim_y[i]) for i
                 in range(s)]

    sim_theta_mean = sum(sim_theta.T)/s

    # Compute sample variance.
    u = sum([np.square(sim_theta[:, i] - sim_theta_mean.T)
             for i in range(s)])/s

    # Standardize parameter vectors.
    sim_theta = np.hstack([(sim_theta[:, i] - sim_theta_mean.T)/np.sqrt(u)
                           for i in range(s)])

    global theta_sigma
    global theta_mean
    theta_sigma = np.sqrt(u)
    theta_mean = sim_theta_mean

    # Standardize observed statistics.
    obs_stats = (obs_stats - np.mean(sim_stats, 0))/np.std(sim_stats, 0)

    # Compute sample mean.
    sim_stats_mean = sum(sim_stats)/s

    # Compute sample variance.
    u = sum([np.square(sim_stats[i]-sim_stats_mean) for i in range(s)])/s

    # Standardize simulated statistics.
    sim_stats = [(sim_stats[i] - sim_stats_mean)/np.sqrt(u) for i in range(s)]

    # Identify k nearest neighbors.
    norms = [np.linalg.norm(obs_stats-sim_stats[i]) for i in range(s)]
    closest_index = np.argsort(norms)
    closest_thetas = [sim_theta[:, i] for i in closest_index[0:round(s*0.03)]]

    # Compute k-nn estimate.
    estimate_standard = (sum(closest_thetas)/len(closest_thetas))

    estimate = np.array(estimate_standard.T)*np.array(
        theta_sigma.T) + np.array(theta_mean)

    return estimate
