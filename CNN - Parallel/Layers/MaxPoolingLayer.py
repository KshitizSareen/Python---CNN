import concurrent.futures
import numpy as np
from Layer import Layer
from collections import deque
from multiprocessing import Pool,Manager

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
        # This returns the flat index of the maximum element.
        # Then we convert that to 2D indices via unravel_index.
        flat_index = np.argmax(imageSection)
        maxValue = imageSection.flat[flat_index]  # or imageSection.ravel()[flat_index]
        
        # Convert flat index to (row, col)
        maxIndexRow, maxIndexCol = np.unravel_index(flat_index, imageSection.shape)
        
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

    
    def MaxPoolingProcess(self,image: np.ndarray,imageNo:int,recordedPositions : dict):
        recordedPositions[imageNo] = []
        maxOutputMatrix = self.maxPooling(image, imageNo)
        return maxOutputMatrix

    def forwardPropagate(self, input: np.ndarray):
        """
        Performs forward propagation through the max pooling layer.
        Expects an input tensor with shape (rows, cols, numberOfImages).
        """
        self.inputMatrix = input
        numberOfImages = self.inputMatrix.shape[2]
        self.outputMatrices = []
        self.recordedPositions = Manager().dict()
        poolArray = [(self.inputMatrix[:,:,i],i,self.recordedPositions) for i in range(numberOfImages)]
        with Pool() as pool:
            self.outputMatrices = pool.starmap(self.MaxPoolingProcess,poolArray)
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
        errorMatrix = nextLayer.changeInCostOverInputMatrix.flatten()
        errors = deque(errorMatrix)  # short way of building a deque

        numOfImages = self.inputMatrix.shape[2]
        self.changeInCostOverInputMatrix = np.zeros_like(self.inputMatrix)

        for i in range(numOfImages):
            for pos in self.recordedPositions[i]:
                if errors:
                    self.changeInCostOverInputMatrix[pos[0], pos[1], i] = errors.popleft()

