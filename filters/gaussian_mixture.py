import numpy as np
from .base_filter import BaseEnsembleFilter

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


    # def __init__(self, pred_states, pred_obs, real_obs, sigmas, gammas):
    #     self.pred_states = pred_states
    #     self.pred_obs = pred_obs
    #     self.real_obs = real_obs

    #     N_tsteps = len(sigmas)
    #     self.N_particles = len(pred_states)

    #     self.sigma_y = 2

    #     # * * * 

    #     print("UPDATING STATE")

    #     self.update_states = np.random.normal(0, sigmas[-1], pred_states.shape)

    #     for t in range(N_tsteps-2, -1, -1):
            
    #         sigma_x = sigmas[t]
    #         gamma = gammas[t]

    #         weight_matrix = self.get_weight_matrix(sigma_x)

    #         for i in range(self.N_particles):

    #             weights = weight_matrix[i]
    #             score = np.sum(weights[:, np.newaxis] * (pred_states - self.update_states[i]) / sigma_x**2, axis=0)

    #             self.update_states[i] = self.update_states[i] + (gamma / (2 * N_tsteps)) * score


    def get_weight_matrix(self, sigma_x):

        weight_matrix = np.zeros((self.N_particles, self.N_particles))

        for i in range(self.N_particles):

            updated_state = self.updated_states[i]

            for j in range(self.N_particles):

                pred_state = self.predicted_states[j]
                pred_ob = self.predicted_obs[j]

                # TODO: cdist or pdist
                state_term = -np.dot(updated_state - pred_state, updated_state - pred_state) / (2 * sigma_x**2)
                ob_term = -np.dot(self.actual_obs - pred_ob, self.actual_obs - pred_ob) / (2 * self.sigma_y**2)

                weight_matrix[i][j] = np.exp(state_term + ob_term)

                # print(f"({i}, {j}): {state_term}")

        row_sums = weight_matrix.sum(axis=1, keepdims=True)

        # print(f"sigma_x = {sigma_x}: {row_sums}")

        return weight_matrix / row_sums