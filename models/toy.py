from models import BaseModel

import numpy as np

class Toy(BaseModel):

    def __str__(self): return f"TOY_{self.obs_operator_name}"

    def __init__(self, model_args, obs_operator):

        # Model Parameters
        super().__init__(model_args['global_args'])

        self.initial_noise = model_args['local_args']['initial_noise']
        self.process_noise = model_args['local_args']['process_noise']

        # Observation Operator Parameters
        self.obs_operator_name = obs_operator['name']
        self.obs_operator_args = obs_operator['args']
    
    @property
    def initial_true_state(self):
        # Returns a single initial state sampled from a bimodal -> [1 x 1]
        return self._generate_initial_particles(1)
    
    @property
    def initial_ensemble(self):
        # Returns an ensemble of initial states sampled from a bimodal -> [N x 1]
        return self._generate_initial_particles(self.ensemble_size)
    
    def predict(self, states, time_span, apply_noise=True):
        """
        Propogates the given states forward in time.

        x_n+1 = x_n + e where e ~ N(0, sigma^2)

        Args:
            states (array):      Ensemble of states to propogate   -> [N x 1]
            time_span (tuple):   (time_start, time_end)
            apply_noise (bool):  If True, applies noise

        Returns:
            predicted_states (array): Ensemble of predicted states -> [N x 1]
        """

        predicted_states = states

        # Apply noise
        if apply_noise == True:
            predicted_states = predicted_states + np.random.normal(loc=0.0, scale=self.process_noise, size=states.shape)
        
        return predicted_states

    def observe(self, states, apply_noise=True):
        """
        Applies the observation operator to the given states.

        Args:
            states (array):      Ensemble of states to observe     -> [N x 1]
            apply_noise (bool):  If True, applies noise

        Returns:
            observations (array): Ensemble of observations         -> [N x dy]
        """
        
        states = np.atleast_2d(states)
        
        if self.obs_operator_name == 'AbsGauss':
            return self.absGauss(states, self.obs_operator_args, apply_noise=apply_noise)
        else:
            raise ValueError(f"\n[ERROR] Unknown Observation Operator: {self.obs_operator_name}")

    @staticmethod
    def absGauss(states, args, apply_noise):
        """
        y = |x| + e where e ~ N(0, sigma^2)
        """

        # Arguments
        sigma = args['sigma']

        # Make observations h(x)
        observations = np.abs(states)

        # Apply noise
        if apply_noise == True:
            observations = observations + np.random.normal(loc=0, scale=sigma, size=observations.shape)

        return observations
    
    def _generate_initial_particles(self, num_particles):
        
        # Randomly assign each particle to a mode of the bimodal
        modes = [-1, 1]
        mode_choices = np.random.choice([0, 1], size=num_particles)

        # Allocate array
        initial_particles = np.zeros((num_particles, self.state_dim))

        # Sample each particle based on its assigned mode
        for i, mode_choice in enumerate(mode_choices):
            initial_particles[i] = np.random.normal(loc=modes[mode_choice], scale=self.initial_noise, size=self.state_dim)

        return initial_particles