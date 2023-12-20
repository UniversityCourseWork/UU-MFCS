import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import seaborn as sns

from gbm_data import Geometric_Brownian_Motion

def loglikelihood(Xt, dt, mu, sigma):
    n = Xt.size
    var = sigma**2
    t1 = - (np.log(sigma))
    t2 = - (0.5 / (var * dt)) * (Xt - ((mu - (0.5 * var)) * dt))**2
    return np.sum(t1 + t2)

def acceptance_criteria(Xt, dt, current_mu, current_sigma, proposed_mu, proposed_sigma, q):
    Ly = loglikelihood(Xt, dt, proposed_mu, proposed_sigma)
    Lx = loglikelihood(Xt, dt, current_mu, current_sigma)
    delta_Lxy = Ly - Lx
    # In order to avoid overflow issues    
    accept_prob = min(1, q * np.exp(delta_Lxy)) if delta_Lxy < 0 else 1
    return accept_prob

def perform_single_sweep(Xt, current_mu, current_sigma, dt):
    #q = np.exp(np.random.normal(0, current_sigma))
    # p = np.exp(np.random.normal(0, 0.1))
    # q = np.exp(np.random.normal(0, 0.1))
    # proposed_mu =  current_mu * p
    # proposed_sigma = current_sigma * q #np.exp(np.random.normal() * 0.1)
    proposed_mu = current_mu + np.random.normal(0, 0.01)
    proposed_sigma = current_sigma * np.exp(np.random.normal(0, 0.01))

    # Compute acceptance probability
    accept_prob = acceptance_criteria(Xt, dt, current_mu, current_sigma, proposed_mu, proposed_sigma, 1.0)

    # Accept or Reject based on probability
    if np.random.random() < accept_prob:
        return proposed_mu, proposed_sigma
    else:
        return current_mu, current_sigma

def mcmc(Xt, init_mu, init_sigma, dt, num_sweeps):
    log_X = np.log(Xt)
    log_dX = log_X[1:] - log_X[:-1]

    hist_mu = np.zeros(num_sweeps, dtype=np.float64) 
    hist_sigma = np.zeros(num_sweeps, dtype=np.float64) 
    hist_mu[0] = init_mu
    hist_sigma[0] = init_sigma

    # Perform sweeps
    for sweep in range(1, num_sweeps):
        # Perform metroplis step
        hist_mu[sweep], hist_sigma[sweep] = perform_single_sweep(log_dX, hist_mu[sweep-1], hist_sigma[sweep-1], dt)

    return hist_mu, hist_sigma

# Define auto-correlation function for
# numpy arrays using numpy function
def autocorr(x, max_lag):
    corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in range(max_lag)]
    return np.array(corr)

if __name__ == "__main__":
    X0 = 1.0
    mu = 0.6
    sigma = 0.2
    N = [50, 100, 250, 500, 1000, 2500]
    delta_T = 0.5
    #T_max = delta_T * N
    project_path = "Module 2 - Foundations of probabilistic modelling/miniproject-3/"

    # Testing GBM implementation
    gbm_paths = [
        Geometric_Brownian_Motion(mu = mu, sigma = sigma, dt = delta_T, n_steps = nsteps, X0 = 1.0).generate_path() for nsteps in N
    ]

    # Run MCMC algorithm
    hist_results = [
        mcmc(gbm_path, 0.01, 0.5, delta_T, 10_000) for gbm_path in gbm_paths
    ]

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].title.set_text(r"Metropolis Convergence")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Parameter Value")
    ax[0].plot(hist_results[2][0], label="Estimated $(\mu)$")
    ax[0].plot([mu]*len(hist_results[2][0]), linewidth = 2, label="Original $(\mu)$", color="Green")
    ax[0].plot(hist_results[2][1], label="Estimated $(\sigma)$")
    ax[0].plot([sigma]*len(hist_results[2][1]), linewidth = 2, label="Original $(\sigma)$", color="Red")
    ax[0].grid(which="both")
    ax[0].legend(loc="upper right")

    ax[1].title.set_text(r"Autocorrelation Function")
    ax[1].set_xlabel("Lag")
    ax[1].set_ylabel("Correlation")
    ax[1].plot(autocorr(hist_results[2][0][1000:], 150), label="$\mu$")
    ax[1].plot(autocorr(hist_results[2][1][1000:], 150), label="$\sigma$")
    ax[1].grid(which="both")
    ax[1].legend(loc="upper right")
    
    plt.savefig(project_path+"figures/mcmc.pdf", bbox_inches='tight', dpi=300)

    print(f"Ergodic Average of Parameter (mu) = {np.mean(hist_results[2][0][1000::50])}")
    print(f"Ergodic Average of Parameter (sigma) = {np.mean(hist_results[2][1][1000::50])}")

    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for indx, item in enumerate(hist_results):
        sns.kdeplot(item[0], ax = ax[0], label = f"N={N[indx]}")
        sns.kdeplot(item[1], ax = ax[1], label = f"N={N[indx]}")

    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('$\mu$')
    ax[1].set_xlabel('$\sigma$')
    fig.suptitle(r"Distribution of $\mu$ and $\sigma$ for different sample size.")
    plt.savefig(project_path+"figures/mcmc_distribution.pdf", bbox_inches='tight', dpi=300)
