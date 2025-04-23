from abc import ABC, abstractmethod

class BaseFilter(ABC):

    @abstractmethod
    def update(self, predicted_states, observation):
        pass