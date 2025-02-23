from abc import abstractmethod
import numpy as np

class Layer:


    
    @abstractmethod
    def forwardPropogate(self,input):
        pass

    @abstractmethod
    def backPropogate(self,outpu,nextLayer = None):
        pass

    @abstractmethod
    def updateWeightsAndBiases(self,learningRate):
        pass

    @abstractmethod
    def combineOutput(self):
        pass

    def __init__(self):
        self.outputMatrix : np.ndarray
        pass

