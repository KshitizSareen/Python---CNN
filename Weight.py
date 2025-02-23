class Weight:
    def __init__(self,weight: float):
        self.weight: float = weight
        self.prevNeuron = None
        self.nextNeuron = None
        self.changeInWeight : float = 0
        pass