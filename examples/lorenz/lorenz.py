#!/usr/bin/env python

# Copyright 2019-2020 Steven Mattis and Troy Butler
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as GKDE
from scipy.stats import beta
import scipy.integrate.quadrature as quad
from luq.luq import *
from luq.dynamical_systems import Lorenz

import ipywidgets as wd
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'axes.linewidth': 2})

np.random.seed(123456)


# Model is for a Lorenz system,
# $$x'(t) = \sigma (y-x),$$
# $$y'(t) = x(\rho-z) - y,$$
# $$z'(t) = xy - \beta z,$$
# with $\sigma \in [9, 11]$, $\beta \in [2, 3]$, and $\rho \in [25, 31]$.

# A ***true*** distribution of $x(0), y(0), z(0), \sigma, \rho$, and $\beta$ are defined by (non-uniform) Beta distributions and used to generate a set of time series data.

# An ***initial*** uniform distribution is assumed and updated by the true time series data.


# Uniformly sample the parameter samples to form a "prediction" or "test" set
num_samples = int(2E3)

num_params = 3
num_ics = 3

params = np.random.uniform(size=(num_samples, num_params))
ics = np.random.uniform(size=(num_samples, num_ics))

param_range = np.array([[9.0, 11.0], # sigma
                        [2.0, 3.0], # beta
                        [25.0, 31.0]]) # rho
ic_range = np.array([[2.0, 2.0], # x(0)
                     [1.0, 1.0], # y(0)
                     [1.5, 1.5]]) # z(0)
params = param_range[:, 0] + (param_range[:, 1] - param_range[:, 0]) * params
ics = ic_range[:, 0] + (ic_range[:, 1] - ic_range[:, 0]) * ics
param_labels = [r'$\sigma$', r'$\beta$', r'$\rho$']
ic_labels = [r'$x(0)$', r'$y(0)$', r'$z(0)$']

# Construct the predicted time series data
num_time_preds = int(501)  # number of predictions (uniformly spaced) between [time_start,time_end]
time_start = 0.0
time_end = 6.0
times = np.linspace(time_start, time_end, num_time_preds)

phys = Lorenz()
predicted_time_series = phys.solve(ics=ics, params=params, t_eval=times)

# Simulate an observed Beta distribution of time series data
num_obs = int(3E2)

true_a = 2
true_b = 2

# data generating distributions
params_obs = np.random.beta(size=(num_obs, num_params), a=true_a, b=true_b)
ics_obs = np.random.beta(size=(num_obs, num_ics), a=true_a, b=true_b)
params_obs = param_range[:, 0] + (param_range[:, 1] - param_range[:, 0]) * params_obs
ics_obs = ic_range[:, 0] + (ic_range[:, 1] - ic_range[:, 0]) * ics_obs

observed_time_series = phys.solve(ics=ics_obs, params=params_obs, t_eval=times)


# Use LUQ to learn dynamics and QoIs
learn = LUQ(predicted_time_series, observed_time_series, times)

# time array indices over which to use
time_start_idx = 0
time_end_idx = num_time_preds-1

num_filtered_obs = 16

# Filter data with piecewise linear splines
learn.filter_data(time_start_idx=time_start_idx,
                  time_end_idx=time_end_idx,
                  num_filtered_obs=num_filtered_obs,
                  tol=5.0e-2,
                  min_knots=3,
                  max_knots=10)


# learn and classify dynamics
learn.dynamics(kwargs={'n_clusters': 2, 'n_init': 10})


fig = plt.figure(figsize=(10,8))

chosen_obs = [0, 8, 10]
colors = ['r', 'g', 'b']

for i, c in zip(chosen_obs,colors):
    plt.plot(learn.times[time_start_idx:time_end_idx],
             learn.observed_time_series[i,time_start_idx:time_end_idx],
             color=c,
             linestyle='none',
             marker='.',
             markersize=10,
             alpha=0.25)
    
for i in chosen_obs:
    num_i_knots = int(0.5*(2+len(learn.obs_knots[i])))
    knots = np.copy(learn.obs_knots[i][num_i_knots:])
    knots = np.insert(knots, 0, learn.times[time_start_idx])
    knots = np.append(knots, learn.times[time_end_idx])
    plt.plot(knots,
             learn.obs_knots[i][:num_i_knots],
             'k',
             linestyle='dashed',
             markersize=15,
             marker='o',
             linewidth=2)
    
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Approximating Dynamics')
plt.show()


fig = plt.figure(figsize=(10,8))

for i, c in zip(chosen_obs,colors):
    plt.plot(learn.times[time_start_idx:time_end_idx],
             learn.observed_time_series[i,time_start_idx:time_end_idx],
             color=c,
             linestyle='none',
             marker='.',
             markersize=10,
             alpha=0.25)
    
for i in chosen_obs:
    plt.plot(learn.filtered_times,
             learn.filtered_obs[i,:],
             'k',
             linestyle='none',
             marker='s',
             markersize=12)
    
plt.xlabel('$t$')
plt.ylabel('$y(t)$')
plt.title('Generating Filtered Data')
plt.show()

# Plot clusters of predicted time series

for j in range(learn.num_clusters):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,8),
                                   gridspec_kw={'width_ratios': [1, 1]}) 
    ax1.scatter(np.tile(learn.filtered_times,num_samples).reshape(num_samples,
                                                                  num_filtered_obs),
                learn.filtered_predictions, 50, c='gray', marker='.', alpha=0.2)
    idx = np.where(learn.predict_labels == j)[0]
    ax1.scatter(np.tile(learn.filtered_times,len(idx)).reshape(len(idx),
                                                               num_filtered_obs),
                learn.filtered_predictions[idx,:],
                50, c='b', marker='o', alpha=0.2)
    ax1.set(title='Cluster ' + str(j+1) + ' in data')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$y(t)$')
    
    ax2.scatter(params[:,1], params[:,2], 30, c='gray', marker='.', alpha=0.2)
    ax2.scatter(params[idx,1], params[idx,2], 50, c='blue', marker='o')
    ax2.set(title='Cluster ' + str(j+1) + ' in parameters')
    ax2.set_ylabel('$\omega_0$')
    ax2.set_xlabel('$c$')
    fig.tight_layout
    plt.show()


# Plot oberved and predicted clusters

for j in range(learn.num_clusters):
    fig = plt.figure(figsize=(10,8))
    plt.scatter(np.tile(learn.filtered_times,num_samples).reshape(num_samples,
                                                                  num_filtered_obs), 
                learn.filtered_predictions,
                10, c='gray', marker='.', alpha=0.2)
    idx = np.where(learn.predict_labels == j)[0]
    plt.scatter(np.tile(learn.filtered_times,len(idx)).reshape(len(idx),
                                                               num_filtered_obs), 
                learn.filtered_predictions[idx,:],
                20, c='b', marker='o', alpha=0.3)
    idx = np.where(learn.obs_labels == j)[0]    
    plt.scatter(np.tile(learn.filtered_times,len(idx)).reshape(len(idx),
                                                               num_filtered_obs), 
                learn.filtered_obs[idx, :],
                50, c='r', marker='s', alpha=0.2)
    plt.title('Classifying filtered observations')
    plt.xlabel('$t$')
    plt.ylabel('$y(t)$')
    bottom, top = plt.gca().get_ylim()
    props = dict(boxstyle='round', facecolor='gray', alpha=0.2)
    plt.text(1, (top-bottom)*0.1+bottom, 
             'Cluster ' + str(j+1), 
             {'color': 'k', 'fontsize': 20},
             bbox=props)
    plt.text
    fig.tight_layout
    plt.show()


# Find best KPCA transformation for given number QoI and transform time series data
predict_map, obs_map = learn.learn_qois_and_transform(num_qoi=3)


def plot_gap(all_eig_vals, n, cluster):
    fig = plt.figure(figsize=(10,10))
    fig.clear()
    #Plotting until maximum number of knots
    eig_vals = all_eig_vals[cluster].lambdas_[0:10]
    plt.semilogy(np.arange(np.size(eig_vals))+1,
                 eig_vals/np.sum(eig_vals)*100,
                 Marker='.', MarkerSize=20, linestyle='')
    plt.semilogy(np.arange(np.size(eig_vals))+1,
                 eig_vals[n]/np.sum(eig_vals)*100*np.ones(np.size(eig_vals)),
                 'k--')
    plt.semilogy(np.arange(np.size(eig_vals))+1,
                 eig_vals[n+1]/np.sum(eig_vals)*100*np.ones(np.size(eig_vals)),
                 'r--')
    plt.text(n+1, eig_vals[n]/np.sum(eig_vals)*150, 
             r'%2.3f' %(np.sum(eig_vals[0:n+1])/np.sum(eig_vals)*100) + '% of variation explained by first ' + '%1d' %(n+1) + ' PCs.',
             {'color': 'k', 'fontsize': 20})
    plt.text(n+2, eig_vals[n+1]/np.sum(eig_vals)*150, 
             r'Order of magnitude of gap is %4.2f.' %(np.log10(eig_vals[n])-np.log10(eig_vals[n+1])),
             {'color': 'r', 'fontsize': 20})
    s = 'Determining QoI for cluster #%1d' %(cluster+1)
    plt.title(s)
    plt.xlabel('Principal Component #')
    plt.ylabel('% of Variation')
    plt.xlim([0.1, np.size(eig_vals)+1])
    plt.ylim([0,500])
    plt.show()


plot_gap(all_eig_vals=learn.kpcas, n=1, cluster=0)
plot_gap(all_eig_vals=learn.kpcas, n=1, cluster=1)

# Generate kernel density estimates on new QoI
learn.generate_kdes()
# Calculate rejection rates for each cluster and print averages.
r_vals = learn.compute_r()

param_marginals = []
ic_marginals = []
true_param_marginals = []
true_ic_marginals = []
lam_ptr = []
cluster_weights = []
for i in range(learn.num_clusters):
    lam_ptr.append(np.where(learn.predict_labels == i)[0])
    cluster_weights.append(len(np.where(learn.obs_labels == i)[0]) / num_obs)

for i in range(0, num_params):
    true_param_marginals.append(GKDE(params_obs[:,params_to_vary[i]]))
    param_marginals.append([])
    for j in range(learn.num_clusters):
        param_marginals[i].append(GKDE(params[lam_ptr[j], i], weights=learn.r[j]))
    

def unif_dist(x, p_range):
    y = np.zeros(x.shape)
    val = 1.0/(p_range[1] - p_range[0])
    for i, xi in enumerate(x):
        if xi < p_range[0] or xi >  p_range[1]:
            y[i] = 0
        else:
            y[i] = val
    return y


for i in range(0, num_params):
    fig = plt.figure(figsize=(10,10))
    fig.clear()
    x_min = min(min(params[:, i]), min(params_obs[:, i]))
    x_max = max(max(params[:, i]), max(params_obs[:, i]))
    delt = 0.25*(x_max - x_min)
    x = np.linspace(x_min-delt, x_max+delt, 100)
    plt.plot(x, unif_dist(x, param_range[i, :]),
         label = 'Initial', linewidth=2)
    mar = np.zeros(x.shape)
    for j in range(learn.num_clusters):
        mar += param_marginals[i][j](x) * cluster_weights[j]
    plt.plot(x, mar, label = 'Updated', linewidth=4, linestyle='dashed')
    plt.plot(x, true_param_marginals[i](x),
             label = 'Data-generating', linewidth=4, linestyle='dotted')
    plt.title('Densities for parameter ' + param_labels[i], fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

    
# Compute TV metric between densities

def param_init_error(x):
    return np.abs(unif_dist(x,param_range[param_num, :])-true_param_marginals[param_num](x))

# for i in range(params.shape[1]):
for i in range(0, num_params):
    param_num=i
    TV_metric = quad(param_init_error,param_range[i,0],param_range[i,1],maxiter=1000)
    print(TV_metric)

def param_update_KDE_error(x):
    mar = np.zeros(x.shape)
    for j in range(learn.num_clusters):
        mar += param_marginals[param_num][j](x) * cluster_weights[j]
    return np.abs(mar-true_param_marginals[param_num](x))

for i in range(0,num_params):
    param_num = i
    TV_metric = quad(param_update_KDE_error,
                     param_range[i,0],
                     param_range[i,1],
                     maxiter=1000)
    print(TV_metric)

def KDE_error(x):
    true_beta = beta(a=true_a,
                     b=true_b,
                     loc=param_range[i,0],
                     scale=param_range[i,1]-param_range[i,0])
    return np.abs(true_beta.pdf(x)-true_param_marginals[param_num](x))

for i in range(0,num_params):
    param_num=i
    TV_metric = quad(KDE_error,param_range[i,0],param_range[i,1],maxiter=1000)
    print(TV_metric)