import numpy as np
from Layer import Layer
from collections import deque

class MaxPoolingLayer(Layer):
    def __init__(self, numRows: int, numCols: int,stride: int,prevLayer : Layer = None):
        # Dictionary mapping image index to a list of (row, col) positions.
        self.stride = stride
        self.recordedPositions: dict[int, list[tuple[int, int]]] = {}
        self.outputMatrices: list[np.ndarray] = []
        self.numRows = numRows
        self.numCols = numCols
        self.inputMatrix: np.ndarray = None  # Will be assigned during forward propagation
        self.outputMatrix: np.ndarray = np.zeros(shape=(int((prevLayer.outputMatrix.shape[0]-numRows)/stride)+1,int((prevLayer.outputMatrix.shape[1]-numCols)/stride)+1,prevLayer.outputMatrix.shape[2]))

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
        outRows = int((inRows - self.numRows)/self.stride) + 1
        outCols = int((inCols - self.numCols)/self.stride) + 1
        outputMatrix = np.zeros((outRows, outCols))
        out_y=0
        i=0
        while i+self.numRows<=inRows:
            j=0
            out_x = 0
            while j+self.numCols<=inCols:
                # Extract a subregion for pooling.
                section = image[i:i+self.numRows, j:j+self.numCols]
                maxValue, localMaxRow, localMaxCol = self.findMaxValueAndPosition(section)
                # Convert local indices to global indices.
                globalRow = i + localMaxRow
                globalCol = j + localMaxCol
                self.recordedPositions[imageNo].append((globalRow, globalCol))
                outputMatrix[out_y, out_x] = maxValue
                j+=self.stride
                out_x+=1
            i+=self.stride
            out_y+=1
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

    def backPropagate(self, output, nextLayer : Layer = None):
        """
        Performs back propagation through the max pooling layer.
        Gathers error signals from the next layer and assigns them to the positions
        corresponding to the maximum values from the forward pass.
        """
        errors = deque()
        errorMatrix = nextLayer.changeInCostOverInputMatrix.flatten()
        for i in range(len(errorMatrix)):
            errors.append(errorMatrix[i])
                    
        numOfImages = self.inputMatrix.shape[2]
        # Initialize a gradient matrix with zeros.
        self.changeInCostOverInputMatrix = np.zeros(shape=self.inputMatrix.shape)


        for i in range(numOfImages):
            for pos in self.recordedPositions[i]:
                if errors:
                    self.changeInCostOverInputMatrix[pos[0], pos[1], i] = errors.popleft()
