#!/usr/bin/env python
"""Calculate summary statistics from data"""

import numpy as np
import simulate_data as sim
from scipy import stats
from sklearn import linear_model
#from matplotlib import pyplot as plt

__author__ = "yasc"
__date_created__ = "07 July 2016"


def sv_stats(delta, y):
    """Calculate statistics from stochastic volatility time series."""

    y_sq = [y[i]**2 for i in range(len(y))]

    z = np.array([])
    z = np.append(z, np.mean(y))  # Mean of returns.
    z = np.append(z, np.std(y))  # Standard deviation of returns.
    z = np.append(z, stats.skew(y))  # Skewness of returns.
    z = np.append(z, stats.kurtosis(y))  # Kurtosis of returns.
    
    z = np.append(z, np.mean(y_sq))  # Mean of realized variation.
    z = np.append(z, np.std(y_sq))  # Standard deviation of realized variation.
    z = np.append(z, stats.skew(y_sq))  # Skewness of realized variation.
    z = np.append(z, stats.kurtosis(y_sq))  # Kurtosis of realized variation.

    z = np.append(z, stats.pearsonr(y, y_sq)[0])  # Correlation: y, rv.

    # Generate lags.
    y_1 = np.matrix(np.delete(np.roll(np.array(y), 1), [0, 1, 2, 3])).T
    y_1_2 = np.square(y_1)
    
    y_2 = np.matrix(np.delete(np.roll(np.array(y), 2), [0, 1, 2, 3])).T
    y_2_2 = np.square(y_2)

    y_3 = np.matrix(np.delete(np.roll(np.array(y), 3), [0, 1, 2, 3])).T
    y_3_2 = np.square(y_2)

    y_4 = np.matrix(np.delete(np.roll(np.array(y), 4), [0, 1, 2, 3])).T
    y_4_2 = np.square(y_4)

    y = np.array(y[4:len(y)])
    y_sq = np.square(y)
    
    y = np.matrix(y).T
    y_sq = np.matrix(y_sq).T

    x = np.hstack([y_1, y_1_2, y_2, y_2_2])

    aux_reg_1 = linear_model.LinearRegression()  # First auxiliary regression.
    aux_reg_2 = linear_model.LinearRegression()  # Second auxiliary regression.
    aux_reg_1.fit(x, y)
    aux_reg_2.fit(x, y_sq)

    res_1 = np.array(y - aux_reg_1.predict(x))
    ste_1 = np.sqrt(sum(res_1**2)/(len(res_1)-1))

    res_2 = np.array(y_sq - aux_reg_2.predict(x))
    ste_2 = np.sqrt(sum(res_2**2)/(len(res_2)-1))

    corr_res = stats.pearsonr(res_1, res_2)

    z = np.append(z, aux_reg_1.intercept_)
    z = np.append(z, aux_reg_1.coef_)
    z = np.append(z, ste_1)
    z = np.append(z, aux_reg_2.intercept_)
    z = np.append(z, aux_reg_2.coef_)
    z = np.append(z, ste_2)

    z = np.append(z, corr_res[0])

    return z[delta > 0]


def lin_reg_stats(delta, y, obs_x, n_noise=5, intercept=True):
    """Calculate summary statistics from linear model."""

    n = obs_x.shape[1]  # Number of regressors (exluding the intercept).
    # Add column of "1"s to observed regressors if the
    # model includes an intercept.
    if intercept:
        obs_x = np.hstack((np.zeros((obs_x.shape[0], 1)) + 1, obs_x))

    z = np.array([])  # Initialize vector of evaluated statistics.

    obs_x_2 = np.matrix(np.delete(obs_x, 0, 1).A**2)
    obs_x_1_2 = np.concatenate([obs_x, obs_x_2], 1)
    obs_x_3 = np.matrix(np.delete(obs_x, 0, 1).A**3)
    obs_x_1_2_3 = np.concatenate([obs_x_1_2, obs_x_3], 1)

    # OLS estimate of linear regressors.
    if sum(delta[0:(n+2)]) > 0:
        beta_hat = np.linalg.inv(obs_x.T*obs_x)*(obs_x.T*y)
        sigma_hat = np.sqrt(((y-obs_x*beta_hat).T*(y-obs_x*beta_hat)) /
                            (obs_x.shape[0]-n-1))
        z = np.append(z, np.append(beta_hat.A, sigma_hat))
    else:
        padding = np.array([float('nan')]*(2+n))
        z = np.concatenate((z, padding), 0)

    # OLS estimate of linear and quadratic regressors.
    if sum(delta[(n+2):(n*3+4)]) > 0:
        # Transform observed regressors.
        beta_hat = np.linalg.inv(obs_x_1_2.T*obs_x_1_2)*(obs_x_1_2.T*y)
        sigma_hat = np.sqrt(((y-obs_x_1_2*beta_hat).T *
                             (y-obs_x_1_2*beta_hat))/(obs_x.shape[0] - 2*n-1))
        z = np.append(z, np.append(beta_hat.A, sigma_hat))
    else:
        padding = np.array([float('nan')]*(2+n*2))
        z = np.concatenate((z, padding), 0)

    # OLS estimate of linear, quadratic, and cubic regressors.
    if sum(delta[(n*3+4):(n*6+6)]) > 0:
        # Transform observed regressors.
        beta_hat = np.linalg.inv(obs_x_1_2_3.T*obs_x_1_2_3)*(obs_x_1_2_3.T*y)
        sigma_hat = np.sqrt(((y-obs_x_1_2_3*beta_hat).T *
                             (y-obs_x_1_2_3*beta_hat))/(obs_x.shape[0]-3*n-1))
        z = np.append(z, np.append(beta_hat.A, sigma_hat))
    else:
        padding = np.array([float('nan')]*(2+n*3))
        z = np.concatenate((z, padding), 0)

    # Noise statistics (standard normal white noise).
    if sum(delta[(n*6+6):(n*6+6+n_noise)]) > 0:
        noise = np.array([np.random.randn(1, 1) for i in
                          range(n_noise)]).flatten()
        z = np.concatenate((z, noise), 0)
    else:
        padding = np.array([float('nan')]*(n_noise))
        z = np.concatenate((z, padding), 0)

    return z[delta > 0]
