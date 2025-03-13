import numpy as np

class LinearGaussian():
    """
    A class representing a linear Gaussian observation operator.
    This adds Gaussian noise to the given states.
    """

    def __init__(self, mean, std, random_sd):
        """
        Initializes the LinearGaussian observation model.

        Parameters:
        - mean (float): The mean of the Gaussian noise.
        - std (float): The standard deviation of the Gaussian noise.
        - random_sd (int): The random seed for reproducibility.
        """

        np.random.seed(random_sd)

        self.mean = mean
        self.std = std

    def get_observations(self, states):
        # Expected states to be a numpy array
        noise = np.random.normal(self.mean, self.std, states.shape)

        return states + noise
    
