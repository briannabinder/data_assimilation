from models import BaseModel

import numpy as np

class Lorenz63(BaseModel):

    def __str__(self): return f"L63_{self.obs_operator_name}"

    def __init__(self, model_args, obs_operator):

        # Model Parameters
        super().__init__(model_args['global_args'])

        self.sigma = model_args['local_args']['sigma']
        self.rho   = model_args['local_args']['rho']
        self.beta  = float(eval(model_args['local_args']['beta']))

        self.initial_noise = model_args['local_args']['initial_noise']
        if isinstance(self.initial_noise, str): self.initial_noise = eval(self.initial_noise)
        self.process_noise = model_args['local_args']['process_noise']
        if isinstance(self.process_noise, str): self.process_noise = eval(self.process_noise)

        self.integration_dt = model_args['local_args']['integration_dt']

        # Observation Operator Parameters
        self.obs_operator_name = obs_operator['name']
        self.obs_operator_args = obs_operator['args']

    @property
    def initial_true_state(self):
        # Returns a single initial state sampled from a Gaussian -> [1 x 3]
        return np.random.normal(loc=0, scale=self.initial_noise, size=(1, self.state_dim))
    
    @property
    def initial_ensemble(self):
        # Returns an ensemble of initial states sampled from a Gaussian -> [N x 3]
        return np.random.normal(loc=0, scale=self.initial_noise, size=(self.ensemble_size, self.state_dim))
    
    def predict(self, states, time_span, apply_noise=True):
        """
        Propogates the given states forward in time.

        dx/dt = s(y - x), 
        dy/dt = x(r - z) - y,
        dz/dt = xy - bz
        
        Args:
            states (array):      Ensemble of states to propogate   -> [N x 3]
            time_span (tuple):   (time_start, time_end)
            apply_noise (bool):  If True, applies noise

        Returns:
            predicted_states (array): Ensemble of predicted states -> [N x 3]
        """
        
        time_start, time_end = time_span
        states = np.atleast_2d(states)

        # Compute derivatives
        def derivatives(states, time): 

            derivatives = np.zeros_like(states)

            xs, ys, zs = states[:,0], states[:,1], states[:,2]

            derivatives[:,0] = self.sigma * (ys - xs)
            derivatives[:,1] = xs * (self.rho - zs) - ys
            derivatives[:,2] = xs * ys - self.beta * zs

            return derivatives
        
        # Euler integration step
        times = np.append(np.arange(time_start, time_end, self.integration_dt), time_end)
        predicted_states = states
        for t_start, t_end in zip(times[:-1], times[1:]):
            predicted_states = predicted_states + derivatives(predicted_states, t_start) * (t_end - t_start)

        # Apply noise
        if apply_noise == True:
            predicted_states = predicted_states + np.random.normal(loc=0, scale=self.process_noise, size=predicted_states.shape)
        
        return predicted_states

    def observe(self, states, apply_noise=True):
        """
        Applies the observation operator to the given states.

        Args:
            states (array):      Ensemble of states to observe     -> [N x 3]
            apply_noise (bool):  If True, applies noise

        Returns:
            observations (array): Ensemble of observations         -> [N x dy]
        """

        states = np.atleast_2d(states)

        if self.obs_operator_name == 'PartialGauss':
            return self.partialGauss(states, self.obs_operator_args, apply_noise=apply_noise)
        
        elif self.obs_operator_name == 'FullGauss':
            return self.fullGauss(states, self.obs_operator_args, apply_noise=apply_noise)
            
        else:
            error_str = f"\n[ERROR] Unknown Observation Operator: {self.obs_operator_name}"
            raise ValueError(error_str)

    @staticmethod
    def partialGauss(states, args, apply_noise):
        """
        x = (x_1, x_2, x_3)
        y = x_3 + e where e ~ N(0, sigma^2)
        """

        # Arguments
        idx = args['idx']
        sigma = args['sigma']
        if isinstance(sigma, str): sigma = eval(sigma)

        # Make observations h(x)
        observations = states[:,idx]

        # Apply noise
        if apply_noise == True:
            observations = observations + np.random.normal(loc=0, scale=sigma, size=observations.shape)

        return observations.reshape(-1, 1) # Ensures observation shape is (N, 1) not (N,)
    
    @staticmethod
    def fullGauss(states, args, apply_noise):
        """
        x = (x_1, x_2, x_3)
        y = x + e where e ~ N(0, sigma^2)
        """

        # Arguments
        sigma = args['sigma']
        if isinstance(sigma, str): sigma = eval(sigma)

        # Make observations h(x)
        observations = states

        # Apply noise
        if apply_noise == True:
            observations = observations + np.random.normal(loc=0, scale=sigma, size=observations.shape)

        return observations

