from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):

    def __init__(self, 
                 initial_ensemble, 
                 initial_time,
                 obs_dt, 
                 T_steps, 
                 ensemble_size, 
                 random_sd):

        np.random.seed(random_sd) # Sets Random Seed

        # Required Model Parameters
        self.T_steps = T_steps
        self.ensemble_size = ensemble_size

        self.obs_dt = obs_dt # *

        self.initial_ensemble = initial_ensemble # Can be given or not
        self.initial_time = initial_time # If not given set to 0 

        self.predicted_states = [initial_ensemble]
        self.updated_states = [initial_ensemble]
        self.times = [initial_time]

    @abstractmethod
    def predict(self, start_states, start_time):
        pass

    @abstractmethod
    def post_process(self):
        pass

    def add_prediction(self, predicted_states): self.predicted_states.append(predicted_states)

    def add_update(self, updated_states): self.updated_states.append(updated_states)

    def add_time(self): self.times.append(self.times[-1] + self.obs_dt)


