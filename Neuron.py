from Weight import Weight


class Neuron:
    
    def __init__(self,bias : float):

        self.zValue : float = 0
        self.aValue : float = 0
        self.inputValue : float = 0
        self.bias : float = 0
        self.error : float= 0
        self.layerNumber : int = 0
        self.layerPosition : int = 0
        self.forwardWeights : list[Weight] = []  
        self.bias  : float = bias
        self.changeInBias : float = 0
        pass
