import numpy as np

from Layer import Layer
from Layers.FullyConnectedLayer import FullyConnectedLayer
import static_functions

class Network:

    def __init__(self,learningRate):
        self.layers : list[Layer] = []

        learningRate : float
        self.learningRate = learningRate
        self.outputVector : np.ndarray
        pass

    def forwardPropogate(self,input: np.ndarray):
        for layer in self.layers:
            layer.forwardPropogate(input)
            input = layer.outputMatrix
        self.outputVector = input
        pass

    def backwardPropogate(self,output):
        k=len(self.layers)-1
        for layer in self.layers[::-1]:
            if k==len(self.layers)-1:
                layer.backPropogate(output)
            else:
                layer.backPropogate(output,self.layers[k+1])
            k-=1
        pass

    def updateWeightsAndBiases(self):
        for layer in self.layers:
            layer.updateWeightsAndBiases(self.learningRate)
        pass

    def calculateError(self,output):
        lastLayer = self.layers[-1]
        lastLayerValues = []
        if isinstance(lastLayer,FullyConnectedLayer):
            for neuron in lastLayer.neurons:
                lastLayerValues.append(neuron.inputValue)
        return static_functions.meanSquareError(np.array(lastLayerValues),output)


    def addLayer(self,layer: Layer):
        self.layers.append(layer)

    def trainImage(self,input,output):
        self.forwardPropogate(input)
        error=self.calculateError(output)
        self.backwardPropogate(output)
        self.updateWeightsAndBiases()
        print(error)

    
    def trainNetwork(self,iterations: int,images : []): # type: ignore
        for i in range(iterations):
            k=0
            for image in images:
                self.trainImage(image[0],image[1])
                k+=1
                print(k)