from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):

    def __init__(self, global_args):

        # Dimensions
        self.state_dim       = global_args['state_dim']             # dx  (int)
        self.observation_dim = global_args['observation_dim']       # dy  (int)

        # Time
        self.initial_time = global_args['initial_time']             # t_0 (float)
        self.dt           = global_args['dt']                       # dt  (float)
        self.T_steps      = global_args['T_steps']                  # T   (int)
        
        # Ensemble Size
        self.ensemble_size = global_args['ensemble_size']           # N   (int)

    @abstractmethod
    def predict(self, states, time_span, apply_noise=True):
        """
        Propogates the given states forward in time.

        Args:
            states (array):      Ensemble of states to propogate   -> [N x dx]
            time_span (tuple):   (time_start, time_end)
            apply_noise (bool):  If True applies noise

        Returns:
            predicted_states (array): Ensemble of predicted states -> [N x dx]
        """
        pass

    @abstractmethod
    def observe(self, states, apply_noise=True):
        """
        Applies the observation operator to the given states.

        Args:
            states (array):      Ensemble of states to observe     -> [N x dx]
            apply_noise (bool):  If True applies noise

        Returns:
            observations (array): Ensemble of observations         -> [N x dy]
        """
        pass

    def generate_data(self, initial_state, initial_time):
        """
        Generates the true states and associated noisy and clean observations. 

        Args:
            initial_state (array):  Initial true state             -> [1 x dx]
            initial_time (float):   Starting time

        Returns:
            true_states (array):         True states               -> [T+1 x dx]
            observations (array):        Noisy observations        -> [T+1 x dy]
            observations_clean (array):  Clean Observations        -> [T+1 x dy]
            times (array):               Time steps                -> [T+1]
        """
        
        # Create time steps [t_0, t_T]
        times = np.linspace(initial_time, initial_time + (self.T_steps * self.dt), self.T_steps + 1)

        # Allocate arrays
        true_states        = np.zeros((self.T_steps + 1, self.state_dim))        # [T+1 x dx]
        observations       = np.zeros((self.T_steps + 1, self.observation_dim))  # [T+1 x dy]
        observations_clean = np.zeros((self.T_steps + 1, self.observation_dim))  # [T+1 x dy]

        # Initialize arrays at t = 0
        true_states[0]        = initial_state
        observations[0]       = np.nan         # No observations at t = 0
        observations_clean[0] = np.nan         # No observations at t = 0

        for t in range(1, self.T_steps + 1): # t : [1, T]

            time_start, time_end = times[t-1], times[t]

            # Predict next state (no process noise)
            true_states[t] = self.predict(true_states[t-1], (time_start, time_end), apply_noise=False)

            # Generate noisy and clean observations
            observations[t]       = self.observe(true_states[t])
            observations_clean[t] = self.observe(true_states[t], apply_noise=False)

        return true_states, observations, observations_clean, times
