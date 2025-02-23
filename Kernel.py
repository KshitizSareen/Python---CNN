import numpy as np
from static_functions import relu,reluDerivativeMatrix,convolveMatrixWithFilter


class Kernel():

    def __init__(self,numRows: int,numCols: int,inputNumRows: int,inputNumCols):
        self.filter : np.ndarray = np.ones((numRows, numCols))
        self.zMatrix  : np.ndarray
        self.activationMatrix : np.ndarray
        self.errorMatrix: np.ndarray
        self.outputNumRows : int = inputNumRows-numRows+1
        self.outputNumCols : int = inputNumCols - numCols +1
        self.numRows : int = numRows
        self.numCols : int = numCols
        self.bias : int = 1
        self.preBiasMatrix : np.ndarray
        self.inputMatrix : np.ndarray
        self.weightChangeMatrix : np.ndarray
        self.biasChange : np.ndarray

    def convolveImage(self,image: np.ndarray):
        convolvedOutput : np.ndarray = np.zeros((self.outputNumRows,self.outputNumCols))
        for i in range(self.zMatrix.shape[0]):
            for j in range(self.zMatrix.shape[1]):
                convolvedValue = np.sum((image[i:i+self.numRows,j:j+self.numCols] * self.filter))
                convolvedOutput[i,j] = convolvedValue
        return convolvedOutput
    
    def Convolve(self,input: np.ndarray):
        shape = input.shape[2]
        self.inputMatrix = input
        self.zMatrix  : np.ndarray = np.zeros((input.shape[0]-self.numRows+1,input.shape[1]-self.numCols+1))
        for i in range(shape):
            self.zMatrix+=self.convolveImage(input[:,:,i])
        self.preBiasMatrix = self.zMatrix.copy()
        self.zMatrix += 1
    
    
    def applyActivation(self):
        self.activationMatrix = relu(self.zMatrix)

    def setErrorMatrix(self,nextLayerMatrix: np.ndarray):
        self.errorMatrix = nextLayerMatrix*reluDerivativeMatrix(self.zMatrix)
        self.weightChangeMatrix = convolveMatrixWithFilter(self.preBiasMatrix,self.errorMatrix)
        self.biasChange = np.sum(self.errorMatrix)
    
    def updateKernelAndBias(self,learningRate):
        self.filter = self.filter - learningRate*(self.weightChangeMatrix)
        self.bias = self.bias - learningRate*self.biasChange