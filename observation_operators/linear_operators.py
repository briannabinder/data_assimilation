import numpy as np

np.random.seed(10)

def linear_gaussian(states, mean, std):
    # Expected states to be a numpy array
    noise = np.random.normal(mean, std, states.shape)

    return states + noise

# TODO: make ABC
class LinearGaussian():

    def __init__(self, mean, std, random_sd):

        self.mean = mean
        self.std = std

        np.random.seed(random_sd)

    def get_observations(self, states):
        # Expected states to be a numpy array
        noise = np.random.normal(self.mean, self.std, states.shape)

        return states + noise