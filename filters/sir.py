from filters import BaseFilter

import numpy as np

class SIR(BaseFilter):

    def __str__(self): return "SIR"

    def __init__(self, filter_args):

        # Filter Parameters
        self.ensemble_size     = filter_args['ensemble_size']
        self.observation_op    = filter_args['observation_op']
        self.observation_noise = filter_args['observation_noise']
    
    def update(self, predicted_states, predicted_observations, observation):

        rng = np.random.default_rng()

        residual = observation - self.observation_op(predicted_states, apply_noise=False)
        W = np.sum(residual * residual, axis=1) / (2 * self.observation_noise**2)
        W = W - np.min(W)

        weight = np.exp(-W).T
        weight = weight/np.sum(weight)

        index = rng.choice(np.arange(self.ensemble_size), self.ensemble_size, p = weight)

        updated_states = predicted_states[index,:]

        return updated_states
