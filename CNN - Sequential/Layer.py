from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    def __init__(self):
        # Initialize outputMatrix as None until it is computed
        self.outputMatrix: np.ndarray = None
        self.changeInCostOverInputMatrix : np.ndarray = None

    @abstractmethod
    def forwardPropagate(self, input: np.ndarray) -> None:
        """
        Performs forward propagation on the input.
        """
        pass

    @abstractmethod
    def backPropagate(self, output: np.ndarray, nextLayer=None) -> None:
        """
        Performs backward propagation using the output and (optionally) the next layer.
        """
        pass

    @abstractmethod
    def combineOutput(self) -> np.ndarray:
        """
        Combines intermediate outputs into the final output matrix.
        """
        pass