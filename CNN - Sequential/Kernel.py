import numpy as np
from static_functions import relu, relu_derivative_matrix, convolve_matrix_with_filter,full_convolve,relu_derivative,dilate_filter,space_array

class Kernel:
    def __init__(self, numRows: int, numCols: int, inputNumRows: int, inputNumCols: int,learningRate : float,stride : int):
        """
        Initialize the kernel (filter) for convolution.

        :param numRows: Number of rows in the kernel.
        :param numCols: Number of columns in the kernel.
        :param inputNumRows: Number of rows in the input image.
        :param inputNumCols: Number of columns in the input image.
        """
        # Initialize the filter weights to ones.
        self.filter: np.ndarray = np.random.randn(numRows, numCols)*learningRate
        
        # These attributes will be set during forward/back propagation.
        self.zMatrix: np.ndarray = None
        self.activationMatrix: np.ndarray = None
        self.errorMatrix: np.ndarray = None
        self.preBiasMatrix: np.ndarray = None
        self.inputMatrix: np.ndarray = None
        self.weightChangeMatrix: np.ndarray = None
        self.biasChange: float = 0.0
        self.changeOverInputMatrix : np.ndarray = None
        self.stride = stride
        
        # Calculate the dimensions of the output after convolution.
        self.outputNumRows: int = int((inputNumRows - numRows)/stride) + 1
        self.outputNumCols: int = int((inputNumCols - numCols)/stride) + 1
        
        self.numRows: int = numRows
        self.numCols: int = numCols
        self.learningRate : float = learningRate
        
        # Initialize bias (set to 1.0 by default)
        self.bias: float = np.random.randn()

    def convolveImage(self, image: np.ndarray) -> np.ndarray:
        """
        Convolve a single-channel image with the kernel filter.

        :param image: 2D numpy array representing one channel of the input image.
        :return: A 2D numpy array containing the convolved output.
        """
        convolvedOutput: np.ndarray = np.zeros((self.outputNumRows, self.outputNumCols))
        i=0
        out_y=0
        while i+self.numRows<=self.inputMatrix.shape[0]:
            j=0
            out_x=0
            while j+self.numCols<=self.inputMatrix.shape[1]:
                # Extract the region from the input corresponding to the kernel size.
                region = image[i:i+self.numRows, j:j+self.numCols]
                convolvedValue = np.sum(region * self.filter)
                convolvedOutput[out_y, out_x] = convolvedValue
                j+=self.stride
                out_x+=1
            i+=self.stride
            out_y+=1
        return convolvedOutput
    
    def convolve(self, input: np.ndarray):
        """
        Convolve the input with the kernel across all channels and add the bias.

        :param input: A 3D numpy array with shape (height, width, channels).
        """
        numChannels = input.shape[2]
        self.inputMatrix = input
        # Initialize zMatrix with zeros.
        self.zMatrix = np.zeros(shape=(self.outputNumRows,self.outputNumCols))
        
        # Sum the convolution results over all channels.
        for channel in range(numChannels):
            self.zMatrix += self.convolveImage(input[:, :, channel])
        
        # Save the pre-bias activations for use in weight updates.
        self.preBiasMatrix = self.zMatrix.copy()
    
    def applyActivation(self):
        """
        Apply the ReLU activation function to the zMatrix.
        """
        leakyReLU = np.vectorize(relu)
        self.activationMatrix = leakyReLU(self.zMatrix)
    
    def setErrorMatrix(self, nextLayerMatrix: np.ndarray):
        """
        Compute the error matrix for the kernel based on the derivative of the ReLU activation
        and calculate weight and bias changes.

        :param nextLayerMatrix: The error matrix propagated from the next layer.
        """
        leakyReLU_derivative = np.vectorize(relu_derivative)
        self.errorMatrix = nextLayerMatrix * leakyReLU_derivative(self.zMatrix)
        summedMatrix = np.sum(self.inputMatrix, axis=2)
        weightChangeMatrix = convolve_matrix_with_filter(summedMatrix,self.errorMatrix,self.filter,self.stride)
        changeOverInputMatrix = full_convolve(self.inputMatrix,self.errorMatrix, self.filter,self.stride)
        self.filter = self.filter - (self.learningRate*weightChangeMatrix)
        self.changeOverInputMatrix = changeOverInputMatrix


