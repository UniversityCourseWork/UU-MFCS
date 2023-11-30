import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Geometric_Brownian_Motion:
    """dX = mu*X*dt + sigma*X*dW"""
    def __init__(self, mu, sigma, dt, n_steps, X0 = 1.0):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = dt
        self.t = np.array([i*self.dt for i in range(n_steps)], dtype=np.float64)  #np.arange(0, self.T+self.dt, self.dt)
        self.X0 = X0
    
    def generate_path(self, seed = None):
        if seed is not None:
            np.random.seed(seed=seed)
        dW = np.sqrt(self.dt) * np.random.randn(self.n_steps)
        Wt = np.cumsum(dW)
        Xt = self.X0 * np.exp( (self.mu - 0.5 * (self.sigma**2)) * self.t + self.sigma * Wt)
        return Xt
    
    def get_expection(self):
        EX = self.X0 * np.exp(self.mu * self.t)
        return EX
    
    def get_variance(self):
        VarX = (self.X0**2) * np.exp(2 * self.mu * self.t) * (np.exp(self.t * self.sigma**2) - 1)
        return VarX

if __name__ == "__main__":
    X0 = 1.0
    n_steps = 500
    delta_T = 0.01 
    T_max = delta_T * n_steps
    project_path = "Module 2 - Foundations of probabilistic modelling/miniproject-3/"

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Testing GBM implementation

    ## mu = 0.5, sigma = 0.01
    mu = 0.5
    sigma = 0.05
    gbm = Geometric_Brownian_Motion(mu = mu, sigma = sigma, dt = delta_T, n_steps = n_steps, X0 = 1.0)
    ax[0, 0].tick_params(which="both", direction="out", labelcolor='black', top=False, bottom=True, left=True, right=False)
    ax[0, 0].minorticks_on()
    ax[0, 0].set_xlabel(r"Time Stamp $(t)$")
    ax[0, 0].set_ylabel(r"Data Samples $(X_t)$")
    ax[0, 0].plot(gbm.get_expection(), linewidth=3, color = "r", label="Expectation")
    for i in range(5):
        ax[0, 0].plot(gbm.generate_path())
    ax[0, 0].title.set_text(r"$\mu = 0.50, \sigma = 0.05$")
    ax[0, 0].legend(loc="upper left")

    ## mu = 0.5, sigma = 0.25
    mu = 0.5
    sigma = 0.25
    gbm = Geometric_Brownian_Motion(mu = mu, sigma = sigma, dt = delta_T, n_steps = n_steps, X0 = 1.0)
    ax[0, 1].tick_params(which="both", direction="out", labelcolor='black', top=False, bottom=True, left=True, right=False)
    ax[0, 1].minorticks_on()
    ax[0, 1].set_xlabel(r"Time Stamp $(t)$")
    ax[0, 1].set_ylabel(r"Data Samples $(X_t)$")
    ax[0, 1].plot(gbm.get_expection(), linewidth=3, color = "r", label="Expectation")
    for i in range(5):
        ax[0, 1].plot(gbm.generate_path())
    ax[0, 1].title.set_text(r"$\mu = 0.50, \sigma = 0.25$")
    ax[0, 1].legend(loc="upper left")

    ## mu = 1.5, sigma = 0.5
    mu = 1.5
    sigma = 0.5
    gbm = Geometric_Brownian_Motion(mu = mu, sigma = sigma, dt = delta_T, n_steps = n_steps, X0 = 1.0)
    ax[1, 0].tick_params(which="both", direction="out", labelcolor='black', top=False, bottom=True, left=True, right=False)
    ax[1, 0].minorticks_on()
    ax[1, 0].set_xlabel(r"Time Stamp $(t)$")
    ax[1, 0].set_ylabel(r"Data Samples $(X_t)$")
    ax[1, 0].plot(gbm.get_expection(), linewidth=3, color = "r", label="Expectation")
    for i in range(5):
        ax[1, 0].plot(gbm.generate_path())
    ax[1, 0].title.set_text(r"$\mu = 1.50, \sigma = 0.50$")
    ax[1, 0].legend(loc="upper left")

    ## mu = 1.0, sigma = 0.25
    mu = 1.0
    sigma = 0.25
    gbm = Geometric_Brownian_Motion(mu = mu, sigma = sigma, dt = delta_T, n_steps = n_steps, X0 = 1.0)
    ax[1, 1].tick_params(which="both", direction="out", labelcolor='black', top=False, bottom=True, left=True, right=False)
    ax[1, 1].minorticks_on()
    ax[1, 1].set_xlabel(r"Time Stamp $(t)$")
    ax[1, 1].set_ylabel(r"Data Samples $(X_t)$")
    ax[1, 1].plot(gbm.get_expection(), linewidth=3, color = "r", label="Expectation")
    for i in range(5):
        ax[1, 1].plot(gbm.generate_path())
    ax[1, 1].title.set_text(r"$\mu = 1.00, \sigma = 0.25$")
    ax[1, 1].legend(loc="upper left")
    plt.savefig(project_path+"figures/gbm_trajectories.pdf", bbox_inches='tight', dpi=300)
    