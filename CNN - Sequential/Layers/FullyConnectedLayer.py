from Layer import Layer
import numpy as np
from static_functions import relu_derivative,relu,softmax,cross_entropy_to_softmax_derivative

class FullyConnectedLayer(Layer):

    def __init__(self, numberOfNeurons: int, prevLayer: Layer, 
                 learningRate : float,
                  isLastLayer: bool = False):
        self.numberOfNeurons = numberOfNeurons
        self.neurons: np.array = np.random.randn(numberOfNeurons)*learningRate
        self.weights: np.array = None
        self.bias : np.array = np.zeros(numberOfNeurons)
        self.inputVector: np.ndarray = None
        self.learningRate : float = learningRate
        self.prevLayer = prevLayer

        self.isLastLayer = isLastLayer
        self.outputMatrix : np.ndarray = np.zeros(numberOfNeurons)
        self.preActivationMatrix : np.ndarray = None
        
        self.weights = np.random.randn(numberOfNeurons,len(prevLayer.outputMatrix.flatten()))
        self.errorVector : np.ndarray = None


    def forwardPropagate(self, input: np.ndarray):
        """Forward propagates the input through the layer."""
        self.inputVector = self.prevLayer.outputMatrix.flatten()
        self.preActivationMatrix = np.dot(self.weights,self.inputVector) + self.bias
        if not self.isLastLayer:
            leakyReLU = np.vectorize(relu)
            self.outputMatrix = leakyReLU(self.preActivationMatrix)
        else:
            self.outputMatrix = softmax(self.preActivationMatrix)


    def backPropagate(self, output, nextLayer: Layer = None):
        """Back propagates the error through the layer."""

        if self.isLastLayer:
            self.errorVector = cross_entropy_to_softmax_derivative(self.preActivationMatrix,output)
        else:
            errorForNextLayer = 0
            if isinstance(nextLayer, FullyConnectedLayer):
                errorForNextLayer = np.sum(nextLayer.errorVector)
                leakyReLU_derivative = np.vectorize(relu_derivative)
                self.errorVector = leakyReLU_derivative(self.preActivationMatrix) * np.full(self.numberOfNeurons,errorForNextLayer)

        self.changeInCostOverInputMatrix = np.dot(np.transpose(self.weights),self.errorVector)
        
        changeInWeightMatrix = self.errorVector.reshape(-1,1)*self.inputVector
        changeInBiasMatrix = self.errorVector.copy()
        self.weights = self.weights - (self.learningRate*changeInWeightMatrix)
        self.bias = self.bias - (self.learningRate*changeInBiasMatrix)

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


    def transformInput(self, input: np.ndarray):
        """Reshapes the input to match the number of neurons in this layer."""
        return input.reshape(self.numberOfNeurons)
    
    def combineOutput(self) -> np.ndarray:
        """
        Combines the output matrices from each image into a single 3D tensor.
        """
        pass
