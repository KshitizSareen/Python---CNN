import numpy as np
from Kernel import Kernel
from Layer import Layer
from multiprocessing import Pool

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

    
    def convolveInputProcess(self,kernel : Kernel,input : np.ndarray):
        kernel.convolve(input)
        kernel.applyActivation()
        return kernel
    
    def backPropogateKernel(self,kernel : Kernel, changeOverInputMatrixForNextLayer : np.ndarray):
        kernel.setErrorMatrix(changeOverInputMatrixForNextLayer[:, :])
        return kernel
    
    def setErrorMatrix(self,kernel : Kernel, changeOverInputMatrixForCurrentLayer : np.ndarray,numChannels: int):
        for num in range(numChannels):
            changeOverInputMatrixForCurrentLayer[:,:,num]+=kernel.changeOverInputMatrix
        return changeOverInputMatrixForCurrentLayer
    


    def forwardPropagate(self, input: np.ndarray):
        """
        Performs convolution on the input using each kernel,
        applies activation, and combines the results.
        """
        self.inputMatrix = input
        poolArray = [(kernel,input) for kernel in self.kernels]
        with Pool() as pool:
            self.kernels = pool.starmap(self.convolveInputProcess,poolArray)
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
        poolArray = [(kernel,nextLayerChangeOverInputMatrix[:,:,k]) for k,kernel in enumerate(self.kernels)]
        with Pool() as pool:
            self.kernels = pool.starmap(self.backPropogateKernel,poolArray)
        poolArray = [(kernel,self.changeInCostOverInputMatrix,numChannels) for kernel in self.kernels]
        with Pool() as pool:
            self.changeInCostOverInputMatrixForEachKernel = pool.starmap(self.setErrorMatrix,poolArray)
        self.changeInCostOverInputMatrix = np.sum(self.changeInCostOverInputMatrixForEachKernel,axis=0)

    def combineOutput(self):
        """
        Combines the activation matrices of all kernels into a single output matrix.
        """
        activationMatrices = [kernel.activationMatrix for kernel in self.kernels]
        self.outputMatrix = np.stack(activationMatrices, axis=2)
