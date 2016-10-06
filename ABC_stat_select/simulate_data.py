#!/usr/bin/env python
"""Simulate data from parameterized models"""

import numpy as np
import math
import random

__author__ = "yasc"
__date_created__ = "06 July 2016"


def prior_draw(n_regressors, s, density="uniform", beta_range=[-2, 2],
               sigma_range=[0, 5]):
    """Return parameters drawn from a prior distributon."""
    if density == "uniform":
        beta = beta_range[0]+(beta_range[1]-beta_range[0])*np.random.rand(
            n_regressors+1, s)

        sigma = sigma_range[1]*np.random.rand(1, s)

    return np.matrix(np.vstack((beta, sigma)))


def sim_lin_reg(beta, sigma, obs_x, s, intercept=True):
    """Return data simulated from a linear model."""

    # Tranpose the vector containing the beta parameters if necessary.
    if beta.shape[1] > 1:
        beta = beta.T
    
    u = np.random.randn(obs_x.shape[0], s)*sigma  # Matrix containing errors.

    # Add column of "1"s to observed regressors if the
    # model includes an intercept.
    if intercept:
        obs_x = np.hstack((np.zeros((obs_x.shape[0], 1)) + 1, obs_x))

    # Matrix containing repeated column vectors of
    # explained component of linear model.
    x = np.tile(obs_x*beta, s)

    return x + u


def sim_sv(t, mu_y, mu_s, phi_1, phi_2, eta):
    """Return simulated SV time series from given parameters"""
    # Inititalize sigma.
    sigma = []
    y = []
    y.append(1)
    y.append(1)
    sigma.append(mu_s)
    sigma.append(mu_s)

    for i in range(2, t+2):
        sigma.append(mu_s + phi_1*(sigma[i-1]) + np.random.randn()*eta)
        y.append(mu_y + phi_2*y[i-1] + math.exp(sigma[i]/2)*np.random.randn())

    return y[2:len(y)]


def sim_svj(t, beta, alpha, kappa, mu_y, mu_s, phi_1, phi_2, eta):
    """Return simulated SV with jumps time series from given parameters"""
    sigma = []
    sigma.append(mu_s)
    sigma.append(mu_s)
    
    y = []
    k = [0, 0]
    q = [0, 0]

    for i in range(2, t + 2):
        k.append(np.random.randn()*beta + alpha)
        q.append(int(random.random() < kappa))
        sigma.append(mu_s + phi_1*(sigma[i-1]-mu_s) + phi_2*(sigma[i-2]-mu_s)
                     + np.random.randn()*eta)
        y.append(k[i]*q[i] + mu_y + math.exp(sigma[i]/2)*np.random.randn())
    sigma = sigma[2:t+2]

    return y


def sim_ARCH(t, mu_y, mu_s, phi_1, phi_2):
    """Return simulated ARCH time series from given parameters"""
    # Inititalize sigma.
    sigma = []
    y = []
    y.append(1)
    y.append(1)
    sigma.append(mu_s)
    sigma.append(mu_s)

    for i in range(2, t+2):
        sigma.append(mu_s + phi_1*(y[i-1]**2) + phi_2*(y[i-2]**2))
        y.append(mu_y + (np.sqrt(sigma[i]))*np.random.randn())

    return y  # y[2:len(y)]


def sim_AR_2(t, mu_y, mu_s, phi_1, phi_2):
    """Return simulated AR(2) time series from given parameters"""
    # Inititalize sigma.
    sigma = []
    y = []
    y.append(1)
    y.append(1)
    sigma.append(mu_s)
    sigma.append(mu_s)

    for i in range(2, t+2):
        y.append(mu_y + phi_1*y[i-1] + phi_2*y[i-2] + np.random.randn())

    return y  # y[2:len(y)]
