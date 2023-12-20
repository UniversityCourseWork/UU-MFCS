import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from gbm_data import Geometric_Brownian_Motion

def loglikelihood(Xt, dt, mu, sigma):
    n = Xt.size
    var = sigma**2
    t1 = - (np.log(sigma))
    t2 = - (0.5 / (var * dt)) * (Xt - ((mu - (0.5 * var)) * dt))**2
    return np.sum(t1 + t2)

def mle_estimator(Xt, T):
    log_X = np.log(Xt)
    dX = log_X[1:] - log_X[:-1]
    pX =  log_X[-1] - log_X[0]
    N = dX.size
    va_hat = - ((pX**2) / (N*T)) + ((1/T)*np.sum(dX**2))
    mu_hat = (pX/T) + (0.5 * va_hat)
    return mu_hat, np.sqrt(va_hat)

def perform_tests():
    plt.rcParams.update({'font.size': 16})
    project_path = "Module 2 - Foundations of probabilistic modelling/miniproject-3/"
    ########################################################################
    ###  Test 1: Impact of delta_T
    ########################################################################
    X0 = 1.0
    mu = 0.3
    sigma = 0.1
    delta_T = np.arange(0.1, 2.0, 0.1)
    #n_steps = 100
    #T_max = delta_T * n_steps
    T_max = 50.0
    n_steps =  T_max * np.reciprocal(delta_T)

    # Generate all paths
    gbm_paths = [Geometric_Brownian_Motion(mu = mu, sigma = sigma, dt=dt, n_steps = int(steps), X0 = X0).generate_path(seed=32) for (dt, steps) in zip(delta_T, n_steps)]

    # Compute MLE estimate for each path
    x_hats = [mle_estimator(path, T_max) for path in gbm_paths]
    errors_mu = [abs(est_mu-mu) for est_mu, _ in x_hats]
    errors_sigma = [abs(est_sigma-sigma) for _, est_sigma in x_hats]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.tick_params(which="both", direction="out", labelcolor='black', top=True, bottom=True, left=True, right=True)
    ax.minorticks_on()
    ax.set_xlabel(r"Step size $(\Delta t)$")
    ax.set_ylabel(r"Absolute Error $(E)$")

    ax.plot(delta_T, errors_mu, marker="o", label=r"$| \hat{\mu} - \mu |$")
    ax.plot(delta_T, errors_sigma, marker="o", label=r"$| \hat{\sigma} - \sigma |$")
    ax.legend(loc="upper right")
    plt.savefig(project_path+"figures/error_deltaT.pdf", bbox_inches='tight', dpi=300)

    ########################################################################
    ###  Test 2: Impact of data size
    ########################################################################
    X0 = 1.0
    mu = 0.3
    sigma = 0.1
    n_steps = np.array([10, 50, 100, 250, 500, 750, 1000]) #np.arange(10, 1_000, 50)
    delta_T = 0.1
    T_max = delta_T * n_steps

    # Generate all paths
    gbm_paths = [Geometric_Brownian_Motion(mu = mu, sigma = sigma, dt=delta_T, n_steps = steps, X0 = X0).generate_path(seed=64) for steps in n_steps]

    # Compute MLE estimate for each path
    x_hats = [mle_estimator(path, T) for path, T in zip(gbm_paths, T_max)]
    errors_mu = [abs(est_mu-mu) for est_mu, _ in x_hats]
    errors_sigma = [abs(est_sigma-sigma) for _, est_sigma in x_hats]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.tick_params(which="both", direction="out", labelcolor='black', top=False, bottom=True, left=True, right=False)
    ax.minorticks_on()
    ax.set_xlabel(r"No. of datapoints $(N)$")
    ax.set_ylabel(r"Absolute Error $(E)$")

    ax.plot(n_steps, errors_mu, marker="o", label=r"$| \hat{\mu} - \mu |$")
    ax.plot(n_steps, errors_sigma, marker="o", label=r"$| \hat{\sigma} - \sigma |$")
    plt.xscale('log')
    plt.legend(loc="upper right")
    plt.savefig(project_path+"figures/error_datasize.pdf", bbox_inches='tight', dpi=300)

    ########################################################################
    ###  Test 3: Impact of constant end time
    ########################################################################
    X0 = 1.0
    mu = 0.5
    sigma = 1.5
    delta_T = np.arange(0.1, 2.0, 0.1)
    #delta_T = delta_T.T
    T_max = np.arange(10.0, 500, 10)

    # Create mesh grid
    X, Y = np.meshgrid(delta_T, T_max, indexing='ij')
    Z1 = -5 * np.ones_like(X)
    Z2 = -5 * np.ones_like(X)

    for i in range(delta_T.size):
        for j in range(T_max.size):
            n_steps = int(Y[i, j] / X[i, j])
            gen_path = Geometric_Brownian_Motion(mu = mu, sigma = sigma, dt=X[i, j], n_steps = n_steps, X0 = X0).generate_path(seed=64)
            # Compute MLE estimate
            if gen_path.size > 2:
                est_mu, est_sigma = mle_estimator(gen_path, Y[i, j])
                # Compute and save errors
                Z1[i, j] = abs(est_mu-mu)
                Z2[i, j] = abs(est_sigma-sigma)

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 1, figsize=(20, 20), subplot_kw={"projection": "3d"})
    ax.set_xlabel(r"Step size $(\Delta t)$")
    ax.set_ylabel(r"Sampling Interval $T_{max}$")
    ax.set_zlabel(r"$| \hat{\mu} - \mu |$")
    surf1 = ax.plot_surface(X, Y, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.view_init(25, 40)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf1, shrink=0.5, aspect=10)
    plt.savefig(project_path+"figures/error_Tmax_mu.pdf", bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(1, 1, figsize=(20, 20), subplot_kw={"projection": "3d"})
    ax.set_xlabel(r"Step size $(\Delta t)$")
    ax.set_ylabel(r"Sampling Interval $T_{max}$")
    ax.set_zlabel(r"$| \hat{\sigma} - \sigma |$")
    surf2 = ax.plot_surface(X, Y, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.view_init(25, 40)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf2, shrink=0.5, aspect=10)
    plt.savefig(project_path+"figures/error_Tmax_sigma.pdf", bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    perform_tests()
    
    ####################### Unit Tests #########################
    # X0 = 1.0
    # mu = 0.3
    # sigma = 0.1
    # n_steps = 100000
    # delta_T = 0.01 
    # T_max = delta_T * n_steps

    # # Testing GBM implementation
    # gbm_path = Geometric_Brownian_Motion(mu = mu, sigma = sigma, dt = delta_T, n_steps = n_steps, X0 = 1.0).generate_path()

    # print(mle_estimator(gbm_path, T_max))
    # print(loglikelihood(gbm_path, 0.5, sigma, dt=delta_T))