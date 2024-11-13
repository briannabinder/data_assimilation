import numpy as np
from abc import ABC, abstractmethod

class BaseEnsembleFilter(ABC):

    @abstractmethod
    def update(self, predicted_states, predicted_obs, actual_obs, *args, **kwargs):
        pass

    def filter(self, model, end_time, dt, *args, **kwargs):

        times = np.arange(0, end_time + dt, dt)
        observations, obs_ts = model.observations

        states = []

        for i, time in enumerate(times):

            print(f"{i}, {time}")

            if time == 0:

                print("Get Initialize states")

                states.append(model.init_states)

            else:

                print("Predicting")

                current_states = states[i-1]

                predicted_states = model.predict(current_states, times[i-1], time)

                if time in obs_ts:
                    print("Observation exists at this time")

                    predicted_obs = model.observation_operator(predicted_states)

                    obs_index = int(np.where(obs_ts == time)[0])
                    actual_obs = observations[obs_index]
                    print(f"observation = {actual_obs}")

                    updated_states = self.update(predicted_states, predicted_obs, actual_obs, *args, **kwargs)
                    states.append(updated_states)
                else:
                    print("Observation unavailable")
                    states.append(predicted_states)

        return states, times
                

            


        