import numpy as np
from Layer import Layer
from Layers.ConvolutedLayer import ConvolutedLayer
from Layers.FullyConnectedLayer import FullyConnectedLayer
from Neuron import Neuron
from collections import deque
from static_functions import full_convolve2d

class MaxPoolingLayer(Layer):
    

    def __init__(self,numRows:int,numCols:int):
        self.recordedPositions: dict[list[int]] = {}
        self.outputMatrices : list[np.ndarray] = []
        self.numRows = numRows
        self.numCols = numCols
        self.inputMatrix : np.ndarray 

    def findMaxValueAndPostion(self,imageSection):
        rows,cols = imageSection.shape
        maxValue = float("-inf")
        maxIndexRow = -1
        maxIndexCol = -1
        for i in range(rows):
            for j in range(cols):
                if imageSection[i,j]>maxValue:
                    maxValue = imageSection[i,j]
                    maxIndexRow = i
                    maxIndexCol = j
        
        return (maxValue,maxIndexRow,maxIndexCol)

    def maxPooling(self,image: np.ndarray,imageNo: int):
        rows,cols = image.shape
        rows = rows - self.numRows+1
        cols = cols - self.numCols+1
        outputMatrix = np.zeros((rows,cols))
        for i in range(rows):
            for j in range(cols):
                maxValue,maxIndexRow,maxIndexCol = self.findMaxValueAndPostion(image[i:i+self.numRows,j:j+self.numCols])
                self.recordedPositions[imageNo].append([maxIndexRow,maxIndexCol])
                outputMatrix[i,j] = maxValue
        
        return outputMatrix

    def forwardPropogate(self,input : np.ndarray):
        self.inputMatrix = input
        numberOfImages = self.inputMatrix.shape[2]
        self.outputMatrices : list[np.ndarray] = []
        self.recordedPositions: dict[list[int]] = {}
        for i in range(numberOfImages):
            self.recordedPositions[i]= []
        for i in range(numberOfImages):
            image = self.inputMatrix[:,:,i]
            maxOutputMatrix = self.maxPooling(image,i)
            self.outputMatrices.append(maxOutputMatrix)
        self.combineOutput()


    def combineOutput(self):
        self.outputMatrix = np.stack(self.outputMatrices,axis=2)
        return self.outputMatrix
        

    def transformOutput(self):
        pass

    def backPropogate(self,output,nextLayer = None):
        errors = deque([])
        if isinstance(nextLayer,FullyConnectedLayer):
            for neuron in nextLayer.neurons:
                totalError = 0
                for weight in neuron.forwardWeights:
                    nextNeuron = weight.nextNeuron
                    totalError += weight.weight*nextNeuron.error
                errors.append(totalError)
        if isinstance(nextLayer,ConvolutedLayer):
            for kernel in nextLayer.Kernels:
                rotatedFilter = np.flipud(np.fliplr(kernel.filter))
                full_convoluted_matrix = full_convolve2d(kernel.errorMatrix,rotatedFilter).flatten()
                for i in range(len(full_convoluted_matrix)):
                    errors.append(full_convoluted_matrix[i])
        numOfImages = self.inputMatrix.shape[2]        
        self.inputMatrix = np.zeros(self.inputMatrix.shape)
        for i in range(numOfImages):
            for position in self.recordedPositions[i]:
                self.inputMatrix[position[0],position[1],i] = errors.popleft()
                
    def updateWeightsAndBiases(self,learningRate):
        pass
                    
        
        pass