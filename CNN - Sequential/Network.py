import numpy as np
from Layer import Layer
from Layers.FullyConnectedLayer import FullyConnectedLayer
import static_functions
import time
import csv

class Network:
    def __init__(self):
        self.layers: list[Layer] = []

    def forwardPropogate(self, input: np.ndarray):
        """
        Forward propagates the input through each layer of the network.
        """
        for layer in self.layers:
            layer.forwardPropagate(input)
            input = layer.outputMatrix
        self.outputVector = input

    def backwardPropogate(self, output):
        """
        Backward propagates the error through the network.
        For the last layer, error is computed using the provided output.
        For previous layers, error is computed using the next layer.
        """
        # Iterate over layers in reverse order
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                # Last layer: use the output directly
                layer.backPropagate(output)
            else:
                # Other layers: pass next layer as additional argument
                nextLayer = self.layers[i + 1]
                layer.backPropagate(output, nextLayer)
            

    def updateWeightsAndBiases(self):
        """
        Updates the weights and biases of all layers using the learning rate.
        """
        for layer in self.layers:
            layer.updateWeightsAndBiases(self.learningRate)

    def calculateError(self, output):
        """
        Calculates the mean squared error of the network output.
        Assumes that the last layer is a FullyConnectedLayer.
        """
        lastLayer = self.layers[-1]
        return static_functions.cross_entropy_loss(lastLayer.outputMatrix, output)

    def addLayer(self, layer: Layer):
        """
        Adds a new layer to the network.
        """
        self.layers.append(layer)

    def trainImage(self, input: np.ndarray, output):
        """
        Trains the network on a single input-output pair.
        """
        self.forwardPropogate(input)
        error = self.calculateError(output)
        self.backwardPropogate(output)
        return error


    def trainNetwork(self, iterations: int, images: list):
        """
        Trains the network over a number of iterations.
        Each element in 'images' should be a tuple or list containing
        the input and the target output.
        """
        for _ in range(iterations):
            count = 0
            totalError = 0
            for image in images:
                start_time = time.time()
                totalError+=self.trainImage(image[0], image[1])
                count += 1
            print("Total error is "+str(totalError) + " after "+str(_)+" epochs")
            with open("Error Progression.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([totalError, _])
