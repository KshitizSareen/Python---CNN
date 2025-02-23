from Layer import Layer
from Neuron import Neuron
from Weight import Weight
import numpy as np
from static_functions import relu_single,ErrorDerivative,reluDerivative

class FullyConnectedLayer(Layer):

    def __init__(self,numberOfNeurons: int,prevLayer: Layer,isFirstLayer : bool = False,isLastLayer: bool = False):
        self.numberOfNuerons : int = numberOfNeurons
        self.neurons : list[Neuron] = []
        self.weights : list[Weight] = []
        self.inputVector : np.ndarray 
        self.prevLayer = prevLayer

        self.isFirstLayer : bool
        self.isLastLayer : bool
        self.isFirstLayer = isFirstLayer
        self.isLastLayer = isLastLayer

        for i in range(numberOfNeurons):
            self.neurons.append(Neuron( 0 if isFirstLayer else 0.5))

        if(not isFirstLayer):
            if isinstance(prevLayer, FullyConnectedLayer):
                for neuron in prevLayer.neurons:
                    for i in range(numberOfNeurons):
                        weight = Weight(0.5)
                        neuron.forwardWeights.append(weight)
                        self.weights.append(weight)
                        weight.nextNeuron = self.neurons[i]
        pass

    def createAandWMatrix(self):
        aMatrix = np.zeros((len(self.neurons),1))
        wMatrix = np.zeros((len(self.neurons[0].forwardWeights),len(self.neurons)))
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            aMatrix[i,0] = neuron.inputValue
            for j in range(len(neuron.forwardWeights)):
                wMatrix[j,i] = neuron.forwardWeights[j].weight
        return (aMatrix,wMatrix)

    def forwardPropogate(self,input):
        self.inputVector = self.transformInput(input)
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            neuron.zValue = self.inputVector[i] if self.isFirstLayer else  self.inputVector[i]+neuron.bias
            neuron.inputValue = neuron.zValue if self.isFirstLayer else  relu_single(neuron.zValue)
        
        self.combineOutput()
        pass

    def backPropogate(self,output,nextLayer : Layer = None ):
        errorForNextLayer = 0
        if not self.isLastLayer and isinstance(nextLayer,FullyConnectedLayer):
            for neuron in nextLayer.neurons:
                errorForNextLayer += neuron.error
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            error = 0
            if self.isLastLayer:
                error =  ErrorDerivative(neuron.inputValue,output[i])
            else:
                error = errorForNextLayer
            reluDerivativeValue = reluDerivative(neuron.zValue)
            neuron.error = error*reluDerivativeValue
            neuron.changeInBias = neuron.error
            prevLayer= self.prevLayer
            if isinstance(prevLayer,FullyConnectedLayer):
                prevLayerNeurons = prevLayer.neurons
                for prevNeuron in prevLayerNeurons:
                    for weight in prevNeuron.forwardWeights:
                        if isinstance(weight,Weight):
                            weight.changeInWeight = neuron.error*prevNeuron.inputValue

            

    def updateWeightsAndBiases(self,learningRate):
        for weight in self.weights:
            weight.weight = weight.weight - learningRate*weight.changeInWeight
        for neuron in self.neurons:
            neuron.bias = neuron.bias - learningRate*neuron.changeInBias

    def combineOutput(self):
        aMatrix,wMatrix = self.createAandWMatrix()
        if not self.isLastLayer:
            self.outputMatrix = np.dot(wMatrix,aMatrix)
        else:
            self.outputMatrix = aMatrix
        pass

    def transformInput(self,input: np.ndarray):
        return input.reshape(self.numberOfNuerons)
