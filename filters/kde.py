from filters import BaseFilter

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.integrate import solve_ivp

class KDE(BaseFilter):

    def __str__(self): return f"KDE"

    def __init__(self, filter_args):
        
        # Filter Parameters
        self.ensemble_size = filter_args['ensemble_size']
        self.scheduler     = filter_args['scheduler']
        self.h_x_min       = filter_args['h_x_min']
        self.h_x_max       = filter_args['h_x_max']
        self.h_y           = filter_args['h_y']
        self.N_tsteps      = filter_args['N_tsteps']
    
    def update(self, predicted_states, predicted_observations, observation):

        # Initialize Schedule
        if self.scheduler == "VE":
            schedule = self.VE(self.h_x_min, self.h_x_max)
        elif self.scheduler == "VP":
            schedule = self.VP(self.h_x_max)

        predicted_mean = np.mean(predicted_states, axis=0)
        predicted_states = predicted_states - predicted_mean

        # Determine the observation term
        obs_terms = cdist(predicted_observations, observation.reshape(1, -1), metric='sqeuclidean').flatten() / (2 * self.h_y**2)
        obs_terms = obs_terms.reshape((1, self.ensemble_size))
        obs_terms = np.repeat(obs_terms, self.ensemble_size, axis=0)

        # Function that is called in sampler that evaluates the score
        def get_score(x, t):

            x = x.reshape(predicted_states.shape) # solve_ivp requires 1d array

            state_terms = cdist(x, schedule._marginal_prob_mean(t) * predicted_states, metric='sqeuclidean') / (2 * schedule._marginal_prob_std(t)**2)
            weight_matrix = softmax(-(state_terms + obs_terms), axis=1)
            
            # Calculate score
            scores = - (1 / schedule._marginal_prob_std(t)**2) * (x - schedule._marginal_prob_mean(t) * np.matmul(weight_matrix, predicted_states))
                                                 
            return scores
        
        latents = np.random.normal(0, 1, predicted_states.shape)

        results = self._odeint_sampler(get_score, schedule, latents, self.h_x_min, self.N_tsteps)
        results = results.reshape(predicted_states.shape)
        updated_states = results + predicted_mean

        return updated_states

    def _odeint_sampler(self, get_score, schedule, latents, sigma_min, N_tsteps):

        init_x = schedule._marginal_prob_std(1) * latents 
        
        def ode_func(t, x):        
            g = schedule._diffusion_coeff(t)
            f = schedule._drift(x.reshape(latents.shape), t)
            rhs = f - 0.5 * g * get_score(x, t) 
            return rhs.flatten() # (N, d)
        
        # Run the black-box ODE solver
        err_tol = 1e-5

        t_eval = np.linspace(1, 0, N_tsteps)
        # t_index = np.argmin(np.abs(schedule._marginal_prob_std(t_eval) - sigma_min))
        # t_end = t_eval[t_index]

        res = solve_ivp(ode_func, (1, 0), init_x.flatten(), rtol=err_tol, atol=err_tol, method='RK45', dense_output=True, t_eval=t_eval)

        return res.y[:,-1]

    class VE: # Variance Exploding
        def __init__(self, sigma_min, sigma_max):
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max

        def _drift(self, x, t): # f(x, t)
            return np.zeros(x.shape)

        def _marginal_prob_mean(self, t): # m(t)
            return 1.0 # np.ones((1,))

        def _marginal_prob_std(self, t): # sigma(t)
            return self.sigma_min + t * (self.sigma_max - self.sigma_min)

        def _diffusion_coeff(self, t): # g(t)
            return 2 * (self._marginal_prob_std(t)) * (self.sigma_max - self.sigma_min) 

    class VP: # Variance Preserving
        def __init__(self, sigma_max):
            self.beta_max = sigma_max

        def _beta_t(self, t): 
            return self.beta_max * t
        
        def _alpha_t(self, t):
            return 0.5 * t**2 * self.beta_max
        
        def _drift(self, x, t):
            return -0.5 * self._beta_t(t) * x
        
        def _marginal_prob_mean(self, t):
            return np.exp(-0.5 * self._alpha_t(t))
        
        def _marginal_prob_std(self, t):

            return np.sqrt(1 - self._marginal_prob_mean(t)**2)
        
        def _diffusion_coeff(self, t):
            return self._beta_t(t)
    