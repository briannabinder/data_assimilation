from filters import BaseFilter
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax

class EnsembleKalmanFilter(BaseFilter):
    def __init__(self, obs_op, random_sd):

        self.obs_op = obs_op
        np.random.seed(random_sd)

    def __str__(self):
        return "EnKF"
    
    def update(self, predicted_states, observation):

        self.predicted_states = predicted_states
        self.N_particles = len(predicted_states)

        predicted_obs = self.obs_op.get_observations(predicted_states)
        # self.observation = observation

        self.updated_states = np.zeros(predicted_states.shape)

        cov_predicted_states = np.cov(predicted_states.T)
        kalman_gain = np.linalg.inv(cov_predicted_states + (self.obs_op.std**2)*np.eye(len(predicted_states[0])))
        
        self.updated_states = predicted_states + ((observation - predicted_obs) @ kalman_gain.T ) @ cov_predicted_states.T
        
        return self.updated_states
                                                                                                             
                   
