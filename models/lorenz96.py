from models import BaseModel

import numpy as np

class Lorenz96(BaseModel):

    def __str__(self): return f"L96_{self.obs_operator_name}"

    def __init__(self, model_args, obs_operator):

        # Model Parameters
        super().__init__(model_args['global_args'])

        self.constant_forcing = model_args['local_args']['constant_forcing']

        self.initial_noise = model_args['local_args']['initial_noise']
        if isinstance(self.initial_noise, str): self.initial_noise = eval(self.initial_noise)

        self.integration_dt = model_args['local_args']['integration_dt']

        # Observation Operator Parameters
        self.obs_operator_name = obs_operator['name']
        self.obs_operator_args = obs_operator['args']
    
    @property
    def initial_true_state(self):
        # Returns a single initial state -> [1 x 40]
        return np.random.normal(loc=0, scale=self.initial_noise, size=(1, self.state_dim))

    @property
    def initial_ensemble(self):
        # Returns an ensemble of initial states -> [N x 40]
        return np.random.normal(loc=0, scale=self.initial_noise, size=(self.ensemble_size, self.state_dim))

    def predict(self, states, time_span, apply_noise=True):
        """
        Propogates the given states forward in time.

        dX_{j}/dt = (X_{j+1} - X_{j-2}) * X_{j-1} - X_{j} + F ; j = 1, ..., 40
        
        Args:
            states (array):      Ensemble of states to propogate   -> [N x 3]
            time_span (tuple):   (time_start, time_end)
            apply_noise (bool):  If True, applies noise

        Returns:
            predicted_states (array): Ensemble of predicted states -> [N x 3]
        """

        time_start, time_end = time_span
        states = np.atleast_2d(states)

        def derivatives(states, time):

            states_after = np.roll(states, -1, axis=1)
            states_1prev = np.roll(states,  1, axis=1)
            states_2prev = np.roll(states,  2, axis=1)

            return (states_after - states_2prev) * states_1prev - states + self.constant_forcing

        times = np.append(np.arange(time_start, time_end, self.integration_dt), time_end)
        predicted_states = states
        for t_start, t_end in zip(times[:-1], times[1:]):
            predicted_states = predicted_states + derivatives(predicted_states, t_start) * (t_end - t_start)

        # Apply noise
        if apply_noise == True: pass
        
        return predicted_states

    def observe(self, states, apply_noise=True):
        """
        Applies the observation operator to the given states.

        Args:
            states (array):      Ensemble of states to observe     -> [N x 40]
            apply_noise (bool):  If True, applies noise

        Returns:
            observations (array): Ensemble of observations         -> [N x 20]
        """
        states = np.atleast_2d(states)

        if self.obs_operator_name == 'OddGauss': 
            return self.oddGauss(states, self.obs_operator_args, apply_noise=apply_noise)
        
        elif self.obs_operator_name == 'EvenGauss': 
            return self.evenGauss(states, self.obs_operator_args, apply_noise=apply_noise)
        
        else:
            error_str = f"\n[ERROR] Unknown Observation Operator: {self.obs_operator_name}"
            raise ValueError(error_str)


    @staticmethod
    def oddGauss(states, args, apply_noise): 

        # Arguments
        sigma = args['sigma']
        if isinstance(sigma, str): sigma = eval(sigma)

        # Make observations h(x)
        observations = states[:, 1::2]

        # Apply noise
        if apply_noise == True:
            observations = observations + np.random.normal(loc=0, scale=sigma, size=observations.shape)

        return observations

    @staticmethod
    def evenGauss(states, args, apply_noise): 

        # Arguments
        sigma = args['sigma']
        if isinstance(sigma, str): sigma = eval(sigma)

        # Make observations h(x)
        observations = states[:, 0::2]

        # Apply noise
        if apply_noise == True:
            observations = observations + np.random.normal(loc=0, scale=sigma, size=observations.shape)

        return observations
