import numpy as np

class LinearGaussian():
    """
    A class representing a linear Gaussian observation operator.
    This adds Gaussian noise to the given states.
    """
    def __str__(self): return "linear"

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
    
class CubicGaussian():
    """
    A class representing a linear Gaussian observation operator.
    This adds Gaussian noise to the given states.
    """
    def __str__(self): return "cubic"

    def __init__(self, mean, std, random_sd):
        """
        Initializes the CubicGaussian observation model.

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

        return (states ** 3) + noise

class SparseGaussian():
    """
    A class representing a linear Gaussian sparse observation operator.
    This only observes every other component of the state and adds 
    Gaussian noise.
    """
    def __str__(self): return "sparse"
    
    def __init__(self, mean, std, random_sd):
        """
        Initializes the SparseGaussian observation model.

        Parameters:
        - mean (float): The mean of the Gaussian noise.
        - std (float): The standard deviation of the Gaussian noise.
        - random_sd (int): The random seed for reproducibility.
        """

        np.random.seed(random_sd)

        self.mean = mean
        self.std = std

    def get_observations(self, states):

        # Observes every other component
        if states.ndim == 1:
            sparse_states = states[::2]
        else:
            sparse_states = states[:, ::2]

        noise = np.random.normal(self.mean, self.std, sparse_states.shape)

        return sparse_states + noise


