from filters import BaseFilter

import numpy as np

class ENKF(BaseFilter):

    def __str__(self): return "ENKF"

    def __init__(self, filter_args):

        # Filter Parameters
        self.ensemble_size = filter_args['ensemble_size']
    
    def update(self, predicted_states, predicted_observations, observation):

        state_mean       = predicted_states.mean(axis=0, keepdims=True)
        observation_mean = predicted_observations.mean(axis=0, keepdims=True)

        a = predicted_states - state_mean 
        b = predicted_observations - observation_mean

        C_xy = 1/self.ensemble_size * a.T @ b
        C_yy = 1/self.ensemble_size * b.T @ b

        dy = predicted_observations.shape[1]
        K  = C_xy @ np.linalg.inv(C_yy + np.eye(dy) * 1e-6)

        updated_states = predicted_states + (K @ (observation - predicted_observations).T).T
        
        return updated_states
