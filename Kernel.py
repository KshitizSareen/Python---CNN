import numpy as np
from static_functions import sigmoid, sigmoid_derivative_matrix, convolve_matrix_with_filter

class Kernel:
    def __init__(self, numRows: int, numCols: int, inputNumRows: int, inputNumCols: int):
        """
        Initialize the kernel (filter) for convolution.

        :param numRows: Number of rows in the kernel.
        :param numCols: Number of columns in the kernel.
        :param inputNumRows: Number of rows in the input image.
        :param inputNumCols: Number of columns in the input image.
        """
        # Initialize the filter weights to ones.
        self.filter: np.ndarray = np.ones((numRows, numCols))
        
        # These attributes will be set during forward/back propagation.
        self.zMatrix: np.ndarray = None
        self.activationMatrix: np.ndarray = None
        self.errorMatrix: np.ndarray = None
        self.preBiasMatrix: np.ndarray = None
        self.inputMatrix: np.ndarray = None
        self.weightChangeMatrix: np.ndarray = None
        self.biasChange: float = 0.0
        
        # Calculate the dimensions of the output after convolution.
        self.outputNumRows: int = inputNumRows - numRows + 1
        self.outputNumCols: int = inputNumCols - numCols + 1
        
        self.numRows: int = numRows
        self.numCols: int = numCols
        
        # Initialize bias (set to 1.0 by default)
        self.bias: float = 1.0

    def convolveImage(self, image: np.ndarray) -> np.ndarray:
        """
        Convolve a single-channel image with the kernel filter.

        :param image: 2D numpy array representing one channel of the input image.
        :return: A 2D numpy array containing the convolved output.
        """
        convolvedOutput: np.ndarray = np.zeros((self.outputNumRows, self.outputNumCols))
        for i in range(self.outputNumRows):
            for j in range(self.outputNumCols):
                # Extract the region from the input corresponding to the kernel size.
                region = image[i:i+self.numRows, j:j+self.numCols]
                convolvedValue = np.sum(region * self.filter)
                convolvedOutput[i, j] = convolvedValue
        return convolvedOutput
    
    def convolve(self, input: np.ndarray):
        """
        Convolve the input with the kernel across all channels and add the bias.

        :param input: A 3D numpy array with shape (height, width, channels).
        """
        numChannels = input.shape[2]
        self.inputMatrix = input
        # Initialize zMatrix with zeros.
        self.zMatrix = np.zeros((input.shape[0] - self.numRows + 1, input.shape[1] - self.numCols + 1))
        
        # Sum the convolution results over all channels.
        for channel in range(numChannels):
            self.zMatrix += self.convolveImage(input[:, :, channel])
        
        # Save the pre-bias activations for use in weight updates.
        self.preBiasMatrix = self.zMatrix.copy()
        # Add the bias value to the convolution result.
        self.zMatrix += self.bias
    
    def applyActivation(self):
        """
        Apply the ReLU activation function to the zMatrix.
        """
        self.activationMatrix = sigmoid(self.zMatrix)
    
    def setErrorMatrix(self, nextLayerMatrix: np.ndarray):
        """
        Compute the error matrix for the kernel based on the derivative of the ReLU activation
        and calculate weight and bias changes.

        :param nextLayerMatrix: The error matrix propagated from the next layer.
        """
        print("nextLayer Matrix is "+str(nextLayerMatrix))
        print("z Matrix is "+str(self.zMatrix))
        self.errorMatrix = nextLayerMatrix * sigmoid_derivative_matrix(self.zMatrix)
        print("Error matrix is "+str(self.errorMatrix))
        summedMatrix = np.sum(self.inputMatrix, axis=2)
        print("Summed Matrix is "+str(summedMatrix))
        self.weightChangeMatrix = convolve_matrix_with_filter(summedMatrix, self.errorMatrix)
        print("Weight change matrix is "+str(self.weightChangeMatrix))
        self.biasChange = np.sum(self.errorMatrix)
        print("Bias change is "+str(self.biasChange))
    
    def updateKernelAndBias(self, learningRate: float):
        """
        Update the kernel filter weights and bias using the computed gradients.

        :param learningRate: The learning rate used for weight updates.
        """
        self.filter = self.filter - learningRate * self.weightChangeMatrix
        self.bias = self.bias - learningRate * self.biasChange
