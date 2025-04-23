from filters import BaseFilter
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.special import softmax
from scipy.integrate import solve_ivp

from .enkf import EnsembleKalmanFilter
# from sklearn.preprocessing import StandardScaler

class VP:
    def __init__(self, sigma_max):
        # self.beta_min = sigma_min # = 0 
        self.beta_max = sigma_max

    def _beta_t(self, t): 
        # self.beta_min + t * (self.beta_max - self.beta_min)
        return self.beta_max * t
    
    def _alpha_t(self, t):
        # t*self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)
        return 0.5 * t**2 * self.beta_max
    
    def _drift(self, x, t):
        return -0.5*self._beta_t(t)*x
    
    def _marginal_prob_mean(self, t):
        return np.exp(-0.5 * self._alpha_t(t))
    
    def _marginal_prob_std(self, t):
        # np.sqrt(1 - np.exp(-self._alpha_t(t)))
        # put if to stop at min bandwidth stop integration
        # root finder to finnd the t for smallest time
        return np.sqrt(1 - self._marginal_prob_mean(t)**2)
    
    def _diffusion_coeff(self, t):
        # transform into g(t)
        # -> self._beta_t(t)
        return self._beta_t(t)
    
class VE:
    def __init__(self, sigma_max):
        self.sigma_min = 0
        self.sigma_max = sigma_max

    def _drift(self, x, t): # f(x, t)
        return np.zeros(x.shape)

    def _marginal_prob_mean(self, t): # m(t)
        return 1.0 # np.ones((1,))

    def _marginal_prob_std(self, t): # sigma(t)
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def _diffusion_coeff(self, t): # g(t)
        # sigma_min * ((sigma_max/sigma_min)**t) 
        return 2 * (self._marginal_prob_std(t)) * (self.sigma_max - self.sigma_min) 
    
def odeint_sampler(get_score, schedule, latents, sigma_min, N_tsteps):
    # init_T = 1.0 * torch.ones(batch_size)
    # latents = N(0, 1)
    init_x = schedule._marginal_prob_std(1) * latents 
    # print(latents)
    # print(init_x.reshape(-1))
    
    def ode_func(t, x):        
        g = schedule._diffusion_coeff(t)
        f = schedule._drift(x.reshape(latents.shape), t)
        rhs = f - 0.5 * g * get_score(x, t) # TODO make stochastic??
        return rhs.flatten() # (N, d)
    
    # Run the black-box ODE solver
    err_tol = 1e-5



    t_eval = np.linspace(1, 0, N_tsteps)
    # t_eval = 10 ** (np.linspace(0, -3, N_tsteps))
    t_index = np.argmin(np.abs(schedule._marginal_prob_std(t_eval) - sigma_min))
    t_end = t_eval[t_index]

    res = solve_ivp(ode_func, (1, t_end), init_x.flatten(), rtol=err_tol, atol=err_tol, method='RK45', dense_output=True, t_eval=t_eval[:t_index + 1] )

    # for i, t in enumerate(t_eval[:-1]):
    #     # print(i)
    #     init_x = init_x - (schedule._drift(init_x, t) - 0.5 * schedule._diffusion_coeff(t) * get_score(init_x, t)) * (t_eval[i] - t_eval[i+1])

    # init_x = init_x - (schedule._drift(init_x, t_eval[-1]) - 0.5 * schedule._diffusion_coeff(t_eval[-1]) * get_score(init_x, t_eval[-1])) * (t_eval[-1])
    # # res = x + ode_func()
    # print(init_x)
    # return init_x
    return res.y[:,-1]

# * * * 

class KernelDensityEstimation(BaseFilter):

    def __init__(self, 
                 obs_op, 
                 random_sd, 
                 scheduler, 
                 sigma_min, 
                 sigma_max, 
                 sigma_y, 
                 N_tsteps=500,
                 hybrid=False):

        np.random.seed(random_sd)
        self.random_sd = random_sd

        self.obs_op = obs_op
        # self.T = T # end time NOTE change variable? or needed?
        self.N_tsteps = N_tsteps # Number of pseudo-time

        self.scheduler = scheduler # "VE" (variance exploding) or "VP" (variance preserving)
        
        self.sigma_min = sigma_min

        self.sigma_max = sigma_max
        self.sigma_y = sigma_y

        self.hybrid = hybrid

    def __str__(self):

        if self.hybrid == True:
            return f"KDE_{self.scheduler}_Hybrid"

        return f"KDE_{self.scheduler}"
    
    def update(self, predicted_states, observation):
        # print(predicted_states)

        # sample_dists = pdist(predicted_states)
        # print(f"Before EnKF: {predicted_states[0]}")
        # print("Making update")
        if self.hybrid is True:
            enkf_filter_params = {'obs_op': self.obs_op,
                                  'random_sd': self.random_sd}
            enkf_filter = EnsembleKalmanFilter(**enkf_filter_params)
            # print("EnKF ran")
            predicted_states = enkf_filter.update(predicted_states, observation)

        # print(f"After EnKF: {predicted_states[0]}")

        # Makes this only for the first time and sets it
        if self.sigma_min == "silverman":
            ensemble_size = len(predicted_states)
            dim = len(predicted_states[0])
            states_std = np.mean(np.std(predicted_states, axis=0))
            sigma_min = (4/(dim+2))**(1/(dim+4)) * states_std * ensemble_size**(-1/(dim+4))
            if sigma_min > 0.5: sigma_min = 0.5
            print(f"std = {states_std}, bandwidth = {sigma_min}")
        else:
            sigma_min = self.sigma_min

        # Initialize Schedule
        if self.scheduler == "VE":
            schedule = VE(self.sigma_max)
            # schedule = VE(self.sigma_max, min_dist)
        elif self.scheduler == "VP":
            schedule = VP(self.sigma_max)

        # Call ode integrator
        latents = np.random.normal(0, 1, predicted_states.shape)
        ensemble_size = len(predicted_states)
        dim = len(predicted_states[0])

        predicted_observations = self.obs_op.get_observations(predicted_states)

        # Shift predicted states to zero mean
        predicted_mean = np.mean(predicted_states, axis=0)
        predicted_states = predicted_states - predicted_mean

        # Determine the observation terms
        obs_terms = cdist(predicted_observations, observation.reshape(1, -1), metric='sqeuclidean').flatten() / (2 * self.sigma_y**2)
        obs_terms = obs_terms.reshape((1, ensemble_size))
        obs_terms = np.repeat(obs_terms, ensemble_size, axis=0)

        # Function that is called in sampler that evaluates the score
        def get_score(x, t):
            x = x.reshape((ensemble_size, dim)) # solve_ivp requires 1d array

            state_terms = cdist(x, schedule._marginal_prob_mean(t) * predicted_states, metric='sqeuclidean') / (2 * schedule._marginal_prob_std(t)**2)
            weight_matrix = softmax(-(state_terms + obs_terms), axis=1)

            # Calculate score
            scores = - (1 / schedule._marginal_prob_std(t)**2) * (x - schedule._marginal_prob_mean(t) * np.matmul(weight_matrix, predicted_states))
                                                                        # weight_matrix @ (x - schedule._marginal_prob_mean(t) * predicted_states)
            return scores

        # print("solving ode")
        results = odeint_sampler(get_score, schedule, latents, sigma_min, self.N_tsteps)
        results = results.reshape((ensemble_size, dim))
        results = results + predicted_mean
        # results = scaler_states.inverse_transform(results) # Un-normalizes the results

        return results
