from abc import ABC, abstractmethod

class Loss(ABC):

    @staticmethod
    @abstractmethod
    def calculate_loss():
        raise NotImplementedError