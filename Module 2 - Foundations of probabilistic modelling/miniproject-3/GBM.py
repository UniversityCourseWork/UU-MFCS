
class Geometric_Brownian_Motion:
    """dS = mu*S*dt + sigma*S*dW"""
    def __init__(self, mu, sigma, n_paths, n_steps, t, T, S_0):
        self.mu = mu
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.t = t
        self.T =T
        self.S_0 = S_0
    
    def get_paths(self):
        """Returns the paths, S, for the Geometric Brownian Motion using Euler-Maruyama method"""
        dt = self.T/self.n_steps
        dW = np.sqrt(dt)*np.random.randn(self.n_paths, self.n_steps)
        dS = (self.mu-0.5*self.sigma**2)*dt + self.sigma*dW
        
        dS = np.insert(dS, 0, 0, axis=1)
        S = np.cumsum(dS, axis=1)
        
        S = self.S_0*np.exp(S)
        return S
    
    def get_expection(self):
        """Returns the expectation, E[S], of the Geometric Brownian Motion"""
        ES = self.S_0*np.exp(self.mu*self.t)
        return ES
    
    def get_variance(self):
        """Returns the variance, Var[S], of the Geometric Brownian Motion"""
        VarS = (self.S_0**2)*np.exp(2*self.mu*self.t)*(np.exp(self.t*self.sigma**2)-1)
        return VarS
    
    def simulate(self, plot_expected=False):
        """Returns the plot of the random paths taken by the Geometric Brownian Motion"""
        plotting_df = pd.DataFrame(self.get_paths().transpose())
        if plot_expected==True:
            plotting_df["Expected Path"]=self.get_expection()
        fig = px.line(plotting_df, labels={"value":"Value of S", "variable":"Paths"})
        return fig.show()