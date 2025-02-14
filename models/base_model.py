from abc import ABC, abstractmethod

class BaseModel(ABC):

    def __init__(self, obs_dt, T_steps, ensemble_size):

        # Required Model Parameters
        self.T_steps = T_steps
        self.ensemble_size = ensemble_size

        self.obs_dt = obs_dt # *

    @abstractmethod
    def predict(self, start_states, start_time, logger=None):
        pass


