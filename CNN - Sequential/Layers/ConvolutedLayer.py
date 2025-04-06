import numpy as np
from Kernel import Kernel
from Layer import Layer

class ConvolutedLayer(Layer):
    def __init__(self, numberOfKernels: int, numRows: int, numCols: int,
                 inputNumRows: int, inputNumCols: int,learningRate: float,stride: int):
        self.numberOfKernels: int = numberOfKernels
        self.stride = stride
        # Initialize kernels as a list of Kernel objects.
        self.kernels: list[Kernel] = []
        for i in range(numberOfKernels):
            self.kernels.append(Kernel(numRows, numCols, inputNumRows, inputNumCols,learningRate,stride))
        self.outputMatrix: np.ndarray = np.zeros(shape=(int((inputNumRows-numRows)/stride)+1,int((inputNumCols-numCols)/stride)+1,numberOfKernels))
        self.inputMatrix: np.ndarray = None


    def forwardPropagate(self, input: np.ndarray):
        """
        Performs convolution on the input using each kernel,
        applies activation, and combines the results.
        """
        self.inputMatrix = input
        for kernel in self.kernels:
            # Call the kernel's convolution function.
            kernel.convolve(input)  # Assuming the Kernel class has a method named 'convolve'
            kernel.applyActivation()
        self.combineOutput()

    def backPropagate(self, output, nextLayer : Layer=None):
        """
        Back propagates the error by setting the error matrix for each kernel.
        The error for each kernel is taken from the corresponding slice in nextLayer.inputMatrix.
        """
        self.changeInCostOverInputMatrix = np.zeros(shape=self.inputMatrix.shape)
        numChannels = self.changeInCostOverInputMatrix.shape[2]
        nextLayerChangeOverInputMatrix = nextLayer.changeInCostOverInputMatrix.reshape(self.outputMatrix.shape)
        # Assuming nextLayer.inputMatrix has shape (rows, cols, numberOfKernels)
        for k, kernel in enumerate(self.kernels):
            kernel.setErrorMatrix(nextLayerChangeOverInputMatrix[:, :, k])
            for num in range(numChannels):
                self.changeInCostOverInputMatrix[:,:,num]+=kernel.changeOverInputMatrix

    def combineOutput(self):
        """
        Combines the activation matrices of all kernels into a single output matrix.
        """
        activationMatrices = [kernel.activationMatrix for kernel in self.kernels]
        self.outputMatrix = np.stack(activationMatrices, axis=2)
