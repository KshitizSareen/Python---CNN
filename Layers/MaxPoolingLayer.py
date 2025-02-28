import numpy as np
from Layer import Layer
from Layers.ConvolutedLayer import ConvolutedLayer
from Layers.FullyConnectedLayer import FullyConnectedLayer
from Neuron import Neuron
from collections import deque
from static_functions import full_convolve2d

class MaxPoolingLayer(Layer):
    def __init__(self, numRows: int, numCols: int):
        # Dictionary mapping image index to a list of (row, col) positions.
        self.recordedPositions: dict[int, list[tuple[int, int]]] = {}
        self.outputMatrices: list[np.ndarray] = []
        self.numRows = numRows
        self.numCols = numCols
        self.inputMatrix: np.ndarray = None  # Will be assigned during forward propagation
        self.outputMatrix: np.ndarray = None

    def findMaxValueAndPosition(self, imageSection: np.ndarray) -> tuple[float, int, int]:
        """
        Finds the maximum value in the given section of the image and returns
        the value along with its (row, col) position (local indices).
        """
        rows, cols = imageSection.shape
        maxValue = float("-inf")
        maxIndexRow = -1
        maxIndexCol = -1
        for i in range(rows):
            for j in range(cols):
                if imageSection[i, j] > maxValue:
                    maxValue = imageSection[i, j]
                    maxIndexRow = i
                    maxIndexCol = j
        return maxValue, maxIndexRow, maxIndexCol

    def maxPooling(self, image: np.ndarray, imageNo: int) -> np.ndarray:
        """
        Applies max pooling to a single image.
        Records the global positions of maximum values.
        """
        inRows, inCols = image.shape
        outRows = inRows - self.numRows + 1
        outCols = inCols - self.numCols + 1
        outputMatrix = np.zeros((outRows, outCols))
        for i in range(outRows):
            for j in range(outCols):
                # Extract a subregion for pooling.
                section = image[i:i+self.numRows, j:j+self.numCols]
                maxValue, localMaxRow, localMaxCol = self.findMaxValueAndPosition(section)
                # Convert local indices to global indices.
                globalRow = i + localMaxRow
                globalCol = j + localMaxCol
                self.recordedPositions[imageNo].append((globalRow, globalCol))
                outputMatrix[i, j] = maxValue
        return outputMatrix

    def forwardPropagate(self, input: np.ndarray):
        """
        Performs forward propagation through the max pooling layer.
        Expects an input tensor with shape (rows, cols, numberOfImages).
        """
        self.inputMatrix = input
        numberOfImages = self.inputMatrix.shape[2]
        self.outputMatrices = []
        self.recordedPositions = {}
        for i in range(numberOfImages):
            self.recordedPositions[i] = []
        for i in range(numberOfImages):
            image = self.inputMatrix[:, :, i]
            maxOutputMatrix = self.maxPooling(image, i)
            self.outputMatrices.append(maxOutputMatrix)
        self.combineOutput()

    def combineOutput(self) -> np.ndarray:
        """
        Combines the output matrices from each image into a single 3D tensor.
        """
        self.outputMatrix = np.stack(self.outputMatrices, axis=2)
        return self.outputMatrix

    def transformOutput(self):
        # If needed, implement any transformation on output before passing to the next layer.
        pass

    def backPropagate(self, output, nextLayer = None):
        """
        Performs back propagation through the max pooling layer.
        Gathers error signals from the next layer and assigns them to the positions
        corresponding to the maximum values from the forward pass.
        """
        errors = deque()
        if isinstance(nextLayer, FullyConnectedLayer):
            for neuron in nextLayer.neurons:
                totalError = 0
                for weight in neuron.forwardWeights:
                    nextNeuron = weight.nextNeuron
                    totalError += weight.weight * nextNeuron.error
                errors.append(totalError)
        elif isinstance(nextLayer, ConvolutedLayer):
            gradMatrix = np.zeros(self.outputMatrix.shape)
            for kernel in nextLayer.kernels:
                rotatedFilter = np.flipud(np.fliplr(kernel.filter))
                full_convoluted_matrix = full_convolve2d(kernel.errorMatrix, rotatedFilter)
                numChannels = gradMatrix.shape[2]
                for i in range(numChannels):
                    gradMatrix[:,:,i]+=full_convoluted_matrix
        gradMatrix=gradMatrix.flatten()
        for i in range(len(gradMatrix)):
            errors.append(gradMatrix[i])
                    
        numOfImages = self.inputMatrix.shape[2]
        # Initialize a gradient matrix with zeros.
        gradInput = np.zeros_like(self.inputMatrix)


        for i in range(numOfImages):
            for pos in self.recordedPositions[i]:
                if errors:
                    gradInput[pos[0], pos[1], i] = errors.popleft()
        self.inputMatrix = gradInput

    def updateWeightsAndBiases(self, learningRate: float):
        # Max pooling layers typically do not have weights or biases.
        pass
