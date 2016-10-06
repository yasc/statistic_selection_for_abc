#!/usr/bin/env python
"""Approximate Bayesian computation based estimators"""

import numpy as np
import time
import math
from scipy import stats
from multiprocessing import pool
from os import getpid

import importance_sampler as isam
import simulate_data as sim
import summary_stats as sum_stat
import selection as select


def sbil_pd_accept_reject(delta, theta_r, stats_r):
    """Return estimates of posterior probability density."""
    
    theta = np.array(select.theta_initial.T)

    # Select relevant statistics as dictated by delta vector.
    sim_stats = [select.stats_initial[i][delta > 0]
                 for i in range(len(select.stats_initial))]

    # Compute Euclidean distances of simulated statistics
    # to observed statistic.
    norms = [np.linalg.norm(sim_stats[i]-stats_r)
             for i in range(len(sim_stats))]

    # Select k nearest statistic vectors based on Euclidean distance.
    k_nearest_stats = np.argsort(norms)[0:round(len(sim_stats)**0.25)]

    denominator = len(k_nearest_stats)

    # Set bandwidth matrix.
    h = np.eye(np.size(theta_r))

    u = [np.linalg.norm(np.linalg.inv(h)*(np.matrix(theta[i]).T - theta_r))
         for i in k_nearest_stats]

    det_h_inv = np.linalg.det(h)**-1

    numerator = det_h_inv*(1/math.sqrt(2*math.pi))*sum(
        [np.exp(-u[i]/2) for i in range(len(u))])

    return (numerator*denominator**-1)*100


def sbil_posterior_mean_rejection(delta, obs_stats, e):
    """Return estimates of posterior mean based on accept-reject method."""
    
    theta = np.array(select.theta_initial.T)
    sim_stats = [select.stats_initial[i][delta > 0]
                 for i in range(len(select.stats_initial))]

    numerator = sum([theta[i]*int(e >= np.linalg.norm(sim_stats[i]-obs_stats))
                     for i in range(len(sim_stats))])

    denominator = sum([int(e >= np.linalg.norm(sim_stats[i]-obs_stats))
                       for i in range(len(sim_stats))])
    return numerator*denominator**-1


def sbil_kernel(delta, obs_stats):
    """Return estimates of posterior mean based on kernel smoother method."""
    
    n = select.sample_dim[1]  # Number of regressors.

    theta = np.array(select.theta_initial.T)
    sim_stats = [select.stats_initial[i][delta > 0]
                 for i in range(len(select.stats_initial))]

    denominator = sum([stats.norm.pdf(np.linalg.norm(sim_stats[i]-obs_stats))
                       for i in range(len(sim_stats))])
    numerator = sum([theta[i]*stats.norm.pdf(np.linalg.norm(
        sim_stats[i]-obs_stats)) for i in range(len(sim_stats))])

    return numerator*denominator**-1


def sbil_knn(obs_stats, sim_stats, prior_theta):
    """Return an SBIL estimate using accept-reject (k nearest neighbors)"""
    # Calculate distances of simulated statistics to observed statistic.
    norms = [np.linalg.norm(obs_stats-sim_stats[i]) for i
             in range(len(sim_stats))]

    # Index of k nearest simulated statistics.
    k_nearest = np.argsort(norms)[0:round(len(sim_stats)**0.25)]

    # Select the k nearest parameters (measured by the distance of the
    # associated simulated statistic to the observed statistic).
    prior_theta = [prior_theta[:, i] for i in range(len(sim_stats))]
    prior_theta = np.array(prior_theta)
    theta_chosen = prior_theta[k_nearest]

    theta_hat = sum(theta_chosen)/len(theta_chosen)

    return theta_hat


def denom(theta, beta_mean, sigma_mean, sim_stats, obs_stats):
    weight = isam.isam_w(theta, beta_mean, sigma_mean)
    smoothed_dist = stats.norm.pdf(np.linalg.norm(sim_stats-obs_stats))

    return weight*smoothed_dist

# Nearly identical copy of above function to avoid having to recreate argument
# list for pool.starmap.
def numer(theta, beta_mean, sigma_mean, sim_stats, obs_stats):
    weight = isam.isam_w(theta, beta_mean, sigma_mean)
    smoothed_dist = stats.norm.pdf(np.linalg.norm(sim_stats-obs_stats))

    return theta*weight*smoothed_dist
