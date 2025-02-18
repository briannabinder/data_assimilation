from filters import BaseFilter
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.special import softmax

class KernelDensityEstimation(BaseFilter):

    def __init__(self, sigma_config, sigma_scale, obs_op, random_sd, N_tsteps=500):

        self.N_tsteps = N_tsteps # Number of pseudo-time

        self.sigma_config = sigma_config
        self.obs_op = obs_op
        self.sigma_scale = sigma_scale
        np.random.seed(random_sd)

    def __str__(self):
        return "KDE"
    
    def update(self, predicted_states, observation, logger=None):

        self.predicted_states = predicted_states
        self.N_particles = len(predicted_states)

        predicted_obs = self.obs_op.get_observations(predicted_states)
        # self.observation = observation

        bw_method = self.sigma_config.get("method")

        if bw_method == "manual":

            # Obtain parameter values
            sigma_min_x = self.sigma_config.get("sigma_min_x")

            if sigma_min_x == "min_dist":
                distances = pdist(self.predicted_states, metric='euclidean')
                sigma_min_x = np.min(distances)
                if logger != None: logger.debug(f"sigma_x min (min dist) = {sigma_min_x}")

            sigma_max_x = self.sigma_config.get("sigma_max_x")
            sigma_y = self.sigma_config.get("sigma_y")
            # print(self.sigma_obs)
            if sigma_min_x is None and sigma_y is None:
                raise ValueError("For 'manual', 'sigma_start', 'sigma_end', and 'sigma_obs' are required parameters.")
            
            distances = pdist(self.predicted_states, metric='euclidean')
            # print(np.min(distances))
            
            # sigmas = np.linspace(sigma_start, sigma_end, self.N_tsteps)
            # sigmas = np.logspace(np.log10(sigma_start), np.log10(sigma_end), self.N_tsteps)

        # elif bw_method == "silverman":

        #     # Obtain parameter values
        #     sigma_end = self.sigma_config.get("sigma_end")
        #     if sigma_end is None:
        #         raise ValueError("For 'silverman', 'sigma_end' are required parameters.")
            
        #     estimated_std = np.std(predicted_states)
        #     silvermans_bw = 1.06 * estimated_std * (len(predicted_states))**(-1/5)
        #     print(f"Silvermans Bandwidth: {silvermans_bw}")

        #     sigmas = np.linspace(silvermans_bw, sigma_end, self.N_tsteps)
        #     self.sigma_obs = silvermans_bw
            
        else:
            raise ValueError("Invalid bandwidth method. Supported methods are 'manual' and 'silverman'.")

        if self.sigma_scale == "linear":
            sigmas = np.linspace(sigma_min_x, sigma_max_x, self.N_tsteps)
            taus = np.linspace(0, 1, self.N_tsteps)

        elif self.sigma_scale == "log": # TODO fix
            sigmas = np.logspace(np.log10(sigma_min_x), np.log10(sigma_max_x), self.N_tsteps)
            taus = np.logspace(0, 1, self.N_tsteps, base=10)

        # Determine the gamma values TODO might be different for logspace...
        taus = np.linspace(0, 1, self.N_tsteps)
        dtau = taus[1] - taus[0]  # Time step (assuming uniform spacing)
        dsigma_dtau = np.gradient(sigmas, dtau)
        gammas = 2 * sigmas * dsigma_dtau

        self.updated_states = np.random.normal(0, sigmas[-1], predicted_states.shape)

        obs_terms = cdist(predicted_obs, observation.reshape(1, -1), metric='sqeuclidean').flatten() / (2 * sigma_y**2)

        for t in range(self.N_tsteps-2, -1, -1):
            
            sigma_pred = sigmas[t]
            gamma = gammas[t]

            weight_matrix = self.get_weight_matrix(sigma_pred, obs_terms)

            for i in range(self.N_particles):

                weights = weight_matrix[i]
                score = np.sum(weights[:, np.newaxis] * (predicted_states - self.updated_states[i]) / sigma_pred**2, axis=0)

                self.updated_states[i] = self.updated_states[i] + (gamma / (2 * self.N_tsteps)) * score

        return self.updated_states

    def get_weight_matrix(self, sigma_pred, obs_terms):

        weight_matrix = np.zeros((self.N_particles, self.N_particles))

        state_terms = cdist(self.updated_states, self.predicted_states, metric='sqeuclidean') / (2 * sigma_pred**2)

        weight_matrix = -(state_terms + obs_terms[None, :])

        return softmax(weight_matrix, axis=1)
