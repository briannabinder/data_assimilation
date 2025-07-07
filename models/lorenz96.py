from models import BaseModel

import numpy as np

class Lorenz96(BaseModel):

    def __str__(self): return f"L96_{self.obs_operator_name}"

    def __init__(self, model_args, obs_operator):

        # Model Parameters
        super().__init__(model_args['global_args'])

        # TODO

        # Observation Operator Parameters
        self.obs_operator_name = obs_operator['name']
        self.obs_operator_args = obs_operator['args']
    
    @property
    def initial_true_state(self):
        # Returns a single initial state -> [1 x dx]
        return initial_true_state

    @property
    def initial_ensemble(self):
        # Returns an ensemble of initial states -> [N x dx]
        return initial_ensemble

    def predict(self, states, time_span, apply_noise=True):
        pass

    def observe(self, states, apply_noise=True):
        """
        Applies the observation operator to the given states.

        Args:
            states (array):      Ensemble of states to observe     -> [N x dx]
            apply_noise (bool):  If True, applies noise

        Returns:
            observations (array): Ensemble of observations         -> [N x dy]
        """
        states = np.atleast_2d(states)


        pass

    @staticmethod
    def Gauss(states, args, apply_noise): 
        pass
