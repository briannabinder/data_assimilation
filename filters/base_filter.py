from abc import ABC, abstractmethod

class BaseFilter(ABC):

    @abstractmethod
    def update(self, predicted_states, predicted_observations, observation):
        """
        Updates an ensemble of predicted states with a given observation.

        Args:
            predicted_states (array):        Ensemble of predicted states                    -> [N x dx]
            predicted_observations (array):  Associated observations of the predicted states -> [N x dy]
            observation (array):             Actual observation at current time step         -> [dy]

        Returns:
            updated_states (array): Ensemble of updated states                               -> [N x dx]
        """
        pass