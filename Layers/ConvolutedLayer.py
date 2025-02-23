import numpy as np
from Kernel import Kernel
from Layer import Layer

class ConvolutedLayer(Layer):
    def __init__(self, numberOfKernels: int, numRows: int, numCols: int,
                 inputNumRows: int, inputNumCols: int):
        self.numberOfKernels: int = numberOfKernels
        # Initialize kernels as a list of Kernel objects.
        self.kernels: list[Kernel] = []
        for i in range(numberOfKernels):
            self.kernels.append(Kernel(numRows, numCols, inputNumRows, inputNumCols))
        self.outputMatrix: np.ndarray = None
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

    def backPropagate(self, output, nextLayer=None):
        """
        Back propagates the error by setting the error matrix for each kernel.
        The error for each kernel is taken from the corresponding slice in nextLayer.inputMatrix.
        """
        # Assuming nextLayer.inputMatrix has shape (rows, cols, numberOfKernels)
        for k, kernel in enumerate(self.kernels):
            kernel.setErrorMatrix(nextLayer.inputMatrix[:, :, k])

    def updateWeightsAndBiases(self, learningRate: float):
        """
        Updates the weights and biases of each kernel using the specified learning rate.
        """
        for kernel in self.kernels:
            kernel.updateKernelAndBias(learningRate)

    def combineOutput(self):
        """
        Combines the activation matrices of all kernels into a single output matrix.
        """
        activationMatrices = [kernel.activationMatrix for kernel in self.kernels]
        self.outputMatrix = np.stack(activationMatrices, axis=2)
