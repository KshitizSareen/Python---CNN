from Layer import Layer
from Neuron import Neuron
from Weight import Weight
import numpy as np
from static_functions import relu_single, error_derivative, relu_derivative

class FullyConnectedLayer(Layer):

    def __init__(self, numberOfNeurons: int, prevLayer: Layer, 
                 isFirstLayer: bool = False, isLastLayer: bool = False):
        self.numberOfNeurons = numberOfNeurons
        self.neurons: list[Neuron] = []
        self.weights: list[Weight] = []
        self.inputVector: np.ndarray = None
        self.prevLayer = prevLayer

        self.isFirstLayer = isFirstLayer
        self.isLastLayer = isLastLayer
        self.outputMatrix : np.ndarray = None

        # Initialize neurons with a starting bias (0 for first layer, else 0.5)
        for i in range(numberOfNeurons):
            self.neurons.append(Neuron(0 if isFirstLayer else 0.5))

        # If not the first layer, create weights connecting neurons from the previous layer
        if not isFirstLayer:
            if isinstance(prevLayer, FullyConnectedLayer):
                for neuron in prevLayer.neurons:
                    for i in range(numberOfNeurons):
                        weight = Weight(0.5)
                        neuron.forwardWeights.append(weight)
                        self.weights.append(weight)
                        weight.nextNeuron = self.neurons[i]
                        weight.prevNeuron = neuron

    def createAandWMatrix(self):
        """Creates the activation (A) matrix and the weight (W) matrix."""
        aMatrix = np.zeros((len(self.neurons), 1))
        # Assume every neuron has the same number of forward weights if available.
        if self.neurons and hasattr(self.neurons[0], 'forwardWeights'):
            numWeights = len(self.neurons[0].forwardWeights)
        else:
            numWeights = 0
        wMatrix = np.zeros((numWeights, len(self.neurons)))
        
        for i, neuron in enumerate(self.neurons):
            aMatrix[i, 0] = neuron.inputValue
            for j, weight_obj in enumerate(neuron.forwardWeights):
                wMatrix[j, i] = weight_obj.weight
        
        return aMatrix, wMatrix

    def forwardPropagate(self, input: np.ndarray):
        """Forward propagates the input through the layer."""
        self.inputVector = self.transformInput(input)
        for i, neuron in enumerate(self.neurons):
            if self.isFirstLayer:
                neuron.zValue = self.inputVector[i]
                neuron.inputValue = neuron.zValue
            else:
                neuron.zValue = self.inputVector[i] + neuron.bias
                neuron.inputValue = relu_single(neuron.zValue)
        self.combineOutput()

    def backPropagate(self, output, nextLayer: Layer = None):
        """Back propagates the error through the layer."""
        errorForNextLayer = 0
        if not self.isLastLayer and isinstance(nextLayer, FullyConnectedLayer):
            for neuron in nextLayer.neurons:
                errorForNextLayer += neuron.error

        for i, neuron in enumerate(self.neurons):
            # For the last layer, calculate error based on the target output.
            if self.isLastLayer:
                error = error_derivative(neuron.inputValue, output[i])
            else:
                error = errorForNextLayer
            reluDerivativeValue = relu_derivative(neuron.zValue)
            neuron.error = error * reluDerivativeValue
            neuron.changeInBias = neuron.error
            

        # Update change in weight for each weight coming from the previous layer
        if isinstance(self.prevLayer, FullyConnectedLayer):
            for weight in self.weights:
                nextNeuron = weight.nextNeuron
                prevNeuron = weight.prevNeuron
                weight.changeInWeight = nextNeuron.error * prevNeuron.inputValue

    def updateWeightsAndBiases(self, learningRate: float):
        """Updates weights and biases using the specified learning rate."""
        for weight in self.weights:
            print("Weight pre update is "+str(weight.weight),"Change in weight is "+str(weight.changeInWeight))
            weight.weight -= learningRate * weight.changeInWeight
            print("Updated weight is "+str(weight.weight))
        for neuron in self.neurons:
            print("neuron bias pre update is "+str(neuron.bias),"Change in bias is "+str(neuron.changeInBias))
            neuron.bias -= learningRate * neuron.changeInBias
            print("Updated bias is "+str(neuron.bias))

    def combineOutput(self):
        """Combines the layer's outputs into a matrix."""
        aMatrix, wMatrix = self.createAandWMatrix()
        if not self.isLastLayer:
            self.outputMatrix = np.dot(wMatrix, aMatrix)
        else:
            self.outputMatrix = aMatrix

    def transformInput(self, input: np.ndarray):
        """Reshapes the input to match the number of neurons in this layer."""
        return input.reshape(self.numberOfNeurons)
