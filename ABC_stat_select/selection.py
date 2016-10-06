#!/usr/bin/env python
"""Selection procedures (local and global)"""

import numpy as np
import multiprocessing as mp
import time
import random
import anneal
import itertools
from multiprocessing import pool, Process  # Redundant?

import simulate_data as sim
import summary_stats as sum_stat
import estimators as estim
import assess_sv_estimation as a

theta_initial = []
stats_initial = []
sample_dim = []
theta_out = np.matrix([])
theta_out_sigma = np.matrix([])
theta_in_sigma = np.matrix([])
stats_out = []
local_knn_estimate = []
obs_stats_standard = []
criterion = ""


class BinaryAnnealer(anneal.Annealer):

    def move(self):
        i = 0
        while i == 0 or sum(self.state) == 0:
            n = random.randrange(len(self.state))
            self.state[n] = 1 - self.state[n]
            i += 1
            
    def energy(self):
        return obj_f(self.state, criterion)


def select_sv(y, s, r, ar, criterion_f):
    """Statistic selection procedure for SV process"""
    # s: In sample size.
    # r: Out sample size.
    # ar:

    # Start timer.
    start = time.time()
    print("Started at:", time.strftime("%H:%M:%S on %d-%m-%Y"))
    
    t = len(y)

    delta = np.array([1]*22)

    # Calculate statistics from observed sample.
    obs_stats = sum_stat.sv_stats(delta, y)
    
    # Generate out theta.
    global theta_out
    theta_out = [generate_theta_sv(ar) for i in range(r)]
    theta_out = np.matrix(theta_out).T

    # Generate out sample of time series.
    out_y = [sim.sim_sv(t, theta_out[0, i], theta_out[1, i], theta_out[2, i],
                        theta_out[3, i], 1) for i in range(r)]

    # Generate out sample statistics.
    global stats_out
    stats_out = [sum_stat.sv_stats(delta, out_y[i]) for i
                 in range(len(out_y))]

    # Standardize the out-of-sample parameters.
    theta_out_mean = sum(theta_out.T)/r
    u = sum([np.square(theta_out[:, i]-theta_out_mean.T) for i in range(r)])/r
    theta_out = np.hstack([(theta_out[:, i] - theta_out_mean.T)/np.sqrt(u)
                           for i in range(r)])
    global theta_out_sigma
    theta_out_sigma = np.sqrt(u)

    # Standardize the out-of-sample statistics.
    stats_out_mean = sum(stats_out)/r
    u = sum([np.square(stats_out[i]-stats_out_mean) for i in range(r)])/r
    stats_out = [(stats_out[i] - stats_out_mean)/np.sqrt(u) for i in range(r)]

    # Generate in theta.
    global theta_initial
    theta_initial = [generate_theta_sv(ar) for i in range(s)]
    theta_initial = np.matrix(theta_initial).T

    # Generate in sample of time series.
    in_y = [sim.sim_sv(t, theta_initial[0, i], theta_initial[1, i],
                       theta_initial[2, i], theta_initial[3, i], 1)
            for i in range(s)]

    # Generate in sample statistics.
    global stats_initial
    stats_initial = [sum_stat.sv_stats(delta, in_y[i]) for i
                     in range(s)]

    # Standardize parameters used for estimation.
    theta_initial_mean = sum(theta_initial.T)/s
    u = sum([np.square(theta_initial[:, i]-theta_initial_mean.T)
             for i in range(s)])/s
    theta_initial = np.hstack([(theta_initial[:, i]
                                - theta_initial_mean.T)/np.sqrt(u)
                               for i in range(s)])

    global theta_in_sigma
    global theta_in_mean
    theta_in_sigma = np.sqrt(u)
    theta_in_mean = theta_initial_mean

    # Standardize statistics for estimation.
    stats_initial_mean = sum(stats_initial)/s
    u = sum([np.square(stats_initial[i]-stats_initial_mean)
             for i in range(s)])/s
    stats_initial = [(stats_initial[i] - stats_initial_mean)/np.sqrt(u) for
                     i in range(s)]

    # Standardize statistics calculated from observed sample.
    global obs_stats_standard
    obs_stats_standard = (obs_stats - stats_initial_mean)/np.sqrt(u)

    # Guess delta.
    delta = np.around(np.random.rand(1, 22)).flatten().astype(int)

    global criterion
    criterion = criterion_f

    #  Summary of sample sizes.
    print("Out sample size:", r)
    print("In sample size:", s)
    print("Observed sample size:", len(y))
    print("Number of persistence parameters:", ar)

    # Initiate binary annealing algorithm and assign tuning parameters.
    print("Initial guess for delta is:", delta)
    print("Criterion function is:", criterion)
    opt = BinaryAnnealer(delta)
    opt.Tmax = 0.10  # 0.25
    opt.Tmin = 0.0001  # 0.001
    opt.steps = len(delta)*10
    opt.updates = len(delta)
    print("Simmulated annealing parameters are:")
    print("Tmax:", opt.Tmax)
    print("Tmin:", opt.Tmin)
    print("Beginning simulated annealing at:",
          time.strftime("%H:%M:%S on %d-%m-%Y"))
    results = opt.anneal()

    results_file = open("annealing_results.txt", "a")
    results_file.write("### Results from annealing run finished at: " +
                       time.strftime("%H:%M:%S on %d-%m-%Y") + " ###\n")
    results_file.write(str(results) + "\n")
    results_file.close()

    print("Final selection and minimum are: ", results)
    print("Finished at :", time.strftime("%H:%M:%S on %d-%m-%Y"))
    print("Total time to run (in minutes):", (time.time()-start)/60, "\n")

    return results


def select_lin_reg(obs_y, obs_x, s, r, q, criterion_f):
    """Selection procedure for linear regression"""
    # r: out of sample simulation size.
    # s: within sample simulation size.
    # q: size of initial importance sampler draw.

    # Start timer.
    t = time.time()

    # Sample dimensions.
    global sample_dim
    sample_dim.append(obs_x.shape[0])
    sample_dim.append(obs_x.shape[1])

    # Candidate statistic vector:
    # (6*sample_dim beta) (3*2 intercepts and error term variances)
    # (5 noise random variables)
    n_candidates = 6*sample_dim[1] + 6 + 5
    delta = np.array([0]*n_candidates)
    delta[0:len(delta)] = 1  # Inititalize delta.

    # Print starting message.
    print("### Beginning summary statistic selection from", n_candidates,
          "candidate summary statistics. ###")
    
    # Calculate full set of summary statistics from observed data.
    obs_stats = sum_stat.lin_reg_stats(delta, obs_y, obs_x)

    # Generate simulated samples from the linear model.
    generate_lin_reg_sample(delta, s, r, sample_dim, obs_stats)

    # Initial guess for delta.
    delta = np.around(np.random.rand(1, n_candidates)).flatten().astype(int)
    
    #  Summary of sample sizes.
    print("Out sample size:", r)
    print("In sample size:", s)
    print("Observed sample size:", sample_dim[0])
    print("Number of regressors:", sample_dim[1])

    global criterion
    criterion = criterion_f

    # Initiate binary annealing algorithm and assign tuning parameters.
    print("Initial guess for delta is:", delta)
    print("Criterion function is:", criterion)
    opt = BinaryAnnealer(delta)
    opt.Tmax = 0.25  # 0.25
    opt.Tmin = 0.001  # 0.001
    opt.steps = len(delta)*10
    opt.updates = len(delta)
    print("Simmulated annealing parameters are:")
    print("Tmax:", opt.Tmax)
    print("Tmin:", opt.Tmin)
    print("Beginning simulated annealing at:",
          time.strftime("%H:%M:%S on %d-%m-%Y"))
    results = opt.anneal()

    results_file = open("annealing_results.txt", "a")
    results_file.write("### Results from annealing run finished at: " +
                       time.strftime("%H:%M:%S on %d-%m-%Y") + " ###\n")
    results_file.write(str(results) + "\n")
    results_file.close()

    print("Final selection and minimum are: ", results)
    print("Finished at :", time.strftime("%H:%M:%S on %d-%m-%Y"))
    print("Total time to run (in minutes):", (time.time()-t)/60, "\n")
    return


def obj_f(delta, criterion_type):
    """Objective functions for the different types of selection procedures"""
    if criterion_type == 'integrated_kl':
        # Integrated Kullback-Leibler distance criterion.
        criterion = -criterion_f(delta, criterion_type)

    if criterion_type == 'abs_dev_knn':
        # Integrated Bayesian loss of posterior mean estimate
        # (estimated by k-nn).
        a = 0
        criterion = (1+a*sum(delta))*criterion_f(
            delta, criterion_type)

    if criterion_type == 'local_abs_dev':
        # Bayesian loss of posterior mean estimate.
        a = 0  # sum(delta)/len(delta)
        sim_stats = [stats_initial[i][delta > 0]
                     for i in range(len(stats_initial))]
        global local_knn_estimate
        local_knn_estimate = estim.sbil_knn(obs_stats_standard[delta > 0],
                                            sim_stats, theta_initial)
        criterion = (1+a*sum(delta))*criterion_f(
            delta, criterion_type)

    if criterion_type == 'local_kl':
        criterion = -criterion_f(delta, criterion_type)

    return criterion


def criterion_f(delta, criterion_type='abs_dev_knn'):
    """Function calls to the different criteria associated with the different
    selection procedures (with the spawning of multiple processes)"""
    stats_out_filtered = [stats_out[i][delta > 0]
                          for i in range(len(stats_out))]
    
    if criterion_type == 'local_kl':
        # Find k nearest statistics from R sample (full candidate set).
        norms = [np.linalg.norm(stats_out[i] - obs_stats_standard)
                 for i in range(len(stats_out))]
        k_nearest_index = np.argsort(norms)[0:round(len(  # 0.01
            stats_out_filtered)**.25)]
        k_nearest_out_theta = [theta_out.T[i] for i in k_nearest_index]
        obs_stats_standard_filtered = obs_stats_standard[delta > 0]
        # Estimate posterior density for each of the k nearest thetas.
        p = pool.Pool()

        results = p.map_async(local_kl_star, zip(itertools.repeat(delta),
                                                 k_nearest_out_theta,
                                                 itertools.repeat(
                                                     obs_stats_standard_filtered)))

        p.close()
        p.join()
        results.wait()
        posterior_densities = results.get()
        numerator = sum(np.log(posterior_densities))
        denominator = len(k_nearest_out_theta)
                
        return numerator/denominator

    if criterion_type == 'local_abs_dev':
        norms = [np.linalg.norm(stats_out_filtered[i] -
                                obs_stats_standard[delta > 0])
                 for i in range(len(stats_out_filtered))]
        k_nearest_stats = np.argsort(norms)[0:round(len(  # 0.01
            stats_out_filtered)**0.25)]
        k_nearest_theta = [theta_out.T[i] for i in k_nearest_stats]
        k_nearest_norms_std = [1 for i in k_nearest_stats]  #[norms[i]/largest_norm for i in k_nearest_stats]
        p = pool.Pool()
        
        results = p.map_async(local_abs_dev_star, zip(k_nearest_theta,
                                                      itertools.repeat(delta),
                                                      k_nearest_norms_std))
        p.close()
        p.join()
        results.wait()
        num_denom = results.get()
        numerator = [num_denom[i][0] for i in range(len(num_denom))]
        denominator = [num_denom[i][1] for i in range(len(num_denom))]
        criterion = sum(numerator)/sum(denominator)
    
    if criterion_type == 'integrated_kl':
        p = pool.Pool()
        results = p.map_async(integrated_kl_star, zip(theta_out.T,
                                                      itertools.repeat(delta),
                                                      stats_out_filtered))
        p.close()
        p.join()
        results.wait()
        posterior_densities = results.get()
        criterion = sum(np.log(posterior_densities))/len(posterior_densities)
    
    if criterion_type == 'abs_dev_knn':
        p = pool.Pool()
        results = p.map_async(abs_dev_knn_star, zip(theta_out.T,  # theta_out.T
                                                    itertools.repeat(delta),
                                                    stats_out_filtered))
        p.close()
        p.join()
        
        results.wait()
        norms = results.get()
        criterion = sum(norms)/len(norms)

    if criterion_type == '':
        print("Loss function is L2-norm.")
        manager = mp.Manager()
        norms = manager.list()
        procs = []
        chunk_size = int(len(stats_out)/p_n)
        print("Chunk size is: ", chunk_size)
        for i in range(4):
            fargs = (theta_out[:, i*chunk_size:(i+1)*chunk_size], delta,
                     stats_out[i*chunk_size:(i+1)*chunk_size], s,
                     q, i, norms)
            p = Process(target=norm_L2_mp, args=fargs)
            procs.append(p)

        print("Spawned these processes: ", procs)

        for p in procs:
            p.start()

        for p in procs:
            p.join()

    return criterion


def local_kl_star(args):
    """Intermediate function call to expand arguments
    (required for map_async())"""
    return estim.sbil_pd_accept_reject(*args)


def local_abs_dev_star(args):
    """Intermediate function call to expand arguments
    (required for map_async())"""
    return local_abs_dev(*args)


def integrated_kl_star(args):
    """Intermediate function call to expand arguments
    (required for map_async())"""
    return integrated_kl(*args)


def abs_dev_knn_star(args):
    """Intermediate function call to expand arguments
    (required for map_async())"""
    return abs_dev_knn(*args)


def local_abs_dev(theta_r, delta, norm):
    abs_dev = (1/theta_r.shape[1])*sum(abs(
        theta_r.T-local_knn_estimate))
    #u = np.linalg.norm(stat_r - obs_stats_standard[delta > 0])#/0.01
    #kernel = (1/math.sqrt((2*math.pi)))*math.exp(-u/2)
    #kernel = int(u < 2)
    kernel = 1  # (3/4)*(1-norm**2)
    return abs_dev*kernel, kernel

def integrated_kl(theta_r, delta, stat_r):
    posterior_density = estim.sbil_pd_accept_reject(delta, theta_r.T, stat_r)
    return posterior_density


def abs_dev_knn(theta_out, delta, obs_stat):

    sim_stats = [stats_initial[i][delta > 0]
                 for i in range(len(stats_initial))]
    knn_estimate = estim.sbil_knn(obs_stat, sim_stats, theta_initial)
    abs_dev = (1/theta_out.shape[1])*sum(theta_in_sigma.T*abs(
        theta_out.T-knn_estimate))
    return abs_dev


def norm_L2_mp(theta_out, delta, stats_out, s, q, i, norms):
    L2_norm = []
    for i in range(len(stats_out)):
        L2_norm.append(np.linalg.norm(theta_out[:, i] - np.matrix(estim.sbil_kernel(
        delta, stats_out[i], s, q, i)).T))
    print("Finished one set!")
    print(L2_norm)
    norms.append(sum(L2_norm))
    return


def theta_and_stats(delta, s, sample_size, n):
    """Draw parameter vectors and simulate associated statistics"""
    # Generate draws from the prior.
    theta = sim.prior_draw(n,  s)

    # Generate data for regressors.
    sim_x = np.matrix(np.random.randn(sample_size, s*n))
    sim_x_iter = [sim_x[:, i*n:(i+1)*n] for i in range(s)]

    # Simulate data from the linear model.
    p = pool.Pool()
    theta_iter = [theta[0:n+1, i] for i in range(s)]
    sigma = [theta[n+1, i] for i in range(s)]
    sim_y = p.map(sim_lin_reg_star, zip(theta_iter, sigma, sim_x_iter,
                                        itertools.repeat(1)))
    p.close()
    p.join()

    # Calculate summary statistics from simulated data.
    p = pool.Pool()
    sim_stats = p.map(lin_reg_stats_star, zip(itertools.repeat(delta),
                                              sim_y, sim_x_iter))
    p.close()
    p.join()

    return [theta, sim_stats]


def sim_lin_reg_star(args):
    return sim.sim_lin_reg(*args)


def lin_reg_stats_star(args):
    return sum_stat.lin_reg_stats(*args)


def generate_lin_reg_sample(delta, s, r, sample_dim, obs_stats):
    """Draw from the prior and simulate associated summary statistics for
    cross validation"""
    global theta_out
    global stats_out
    result = theta_and_stats(delta, r, sample_dim[0], sample_dim[1])
    theta_out = result[0]
    stats_out = result[1]

    # Standardize the out-of-sample parameters.
    theta_out_mean = sum(theta_out.T)/r
    u = sum([np.square(theta_out[:, i]-theta_out_mean.T) for i in range(r)])/r
    theta_out = np.hstack([(theta_out[:, i] - theta_out_mean.T)/np.sqrt(u)
                           for i in range(r)])
    global theta_out_sigma
    theta_out_sigma = np.sqrt(u)

    # Standardize the out-of-sample statistics.
    stats_out_mean = sum(stats_out)/r
    u = sum([np.square(stats_out[i]-stats_out_mean) for i in range(r)])/r
    stats_out = [(stats_out[i] - stats_out_mean)/np.sqrt(u) for i in range(r)]

    # Draw from the prior and caclulate summary statistics for estimation.
    result = theta_and_stats(delta, s, sample_dim[0], sample_dim[1])
    global theta_initial
    global stats_initial
    theta_initial = result[0]
    stats_initial = result[1]

    # Standardize parameters used for estimation.
    theta_initial_mean = sum(theta_initial.T)/s
    u = sum([np.square(theta_initial[:, i]-theta_initial_mean.T)
             for i in range(s)])/s
    global theta_in_sigma
    theta_in_sigma = np.sqrt(u)
    theta_initial = np.hstack([(theta_initial[:, i]
                                - theta_initial_mean.T)/np.sqrt(u)
                               for i in range(s)])
    
    # Standardize statistics for estimation.
    stats_initial_mean = sum(stats_initial)/s
    u = sum([np.square(stats_initial[i]-stats_initial_mean)
             for i in range(s)])/s
    stats_initial = [(stats_initial[i] - stats_initial_mean)/np.sqrt(u) for
                     i in range(s)]

    # Standardize statistics calculated from observed sample.
    global obs_stats_standard
    obs_stats_standard = (obs_stats - stats_initial_mean)/np.sqrt(u)


def generate_theta_sv(ar):
    """Return a parameter vector draw from the prior for the SV process"""
    theta_out = []

    mu_y = 0
    while abs(mu_y) < 0.5:
        mu_y = np.random.uniform(-6, 6)

    theta_out.append(mu_y)  # Expected return.
    theta_out.append(np.random.uniform(0, 3))  # Mu_s.
    theta_out.append(np.random.uniform(0, 0.7))  # Phi_1
    theta_out.append(np.random.uniform(0, 0.7))  # Phi_2

    # Initialize autoregressive coefficients in lag polynomial.
    #p = [-1, -1, 1]

    #while sum(abs(np.roots(p)) < 1.2) != 0:
        #p = [np.random.randn() for i in range(ar)]
        #p = [np.random.uniform(0, 1)*-1 for i in range(ar)]
        #p.append(1)

    #theta_out.append(-1*p[1])  # Phi_1
    #theta_out.append(-1*p[0])  # Phi_2
    #theta_out.append(-1*p[1])
    #theta_out.append(-1*p[0])
    #theta_out.append(np.random.rand()*2)  # Sigma_eta.

    return theta_out
