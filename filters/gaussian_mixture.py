import numpy as np
import multiprocessing
from .base_filter import BaseEnsembleFilter
from scipy.spatial.distance import cdist
from scipy.special import softmax

class GaussianMixture(BaseEnsembleFilter):
  
    def update(self, predicted_states, predicted_obs, actual_obs, sigmas, gammas):

        self.predicted_states = predicted_states
        self.predicted_obs = predicted_obs
        self.actual_obs = actual_obs

        N_tsteps = len(sigmas)
        self.N_particles = len(predicted_states)

        self.sigma_y = 2

        self.updated_states = np.random.normal(0, sigmas[-1], predicted_states.shape)

        for t in range(N_tsteps-2, -1, -1):
            
            sigma_x = sigmas[t]
            gamma = gammas[t]

            weight_matrix = self.get_weight_matrix(sigma_x)

            for i in range(self.N_particles):

                weights = weight_matrix[i]
                score = np.sum(weights[:, np.newaxis] * (predicted_states - self.updated_states[i]) / sigma_x**2, axis=0)

                self.updated_states[i] = self.updated_states[i] + (gamma / (2 * N_tsteps)) * score

        return self.updated_states

    def get_weight_matrix(self, sigma_x):

        # Fast multiple method

        weight_matrix = np.zeros((self.N_particles, self.N_particles))

        obs_terms = cdist(self.predicted_obs, self.actual_obs.reshape(1, -1), metric='sqeuclidean').flatten() / (2 * self.sigma_y**2)
        state_terms = cdist(self.updated_states, self.predicted_states, metric='sqeuclidean') / (2 * sigma_x**2)

        weight_matrix = -(state_terms + obs_terms[None, :])

        return softmax(weight_matrix, axis=1)
