import numpy as np
from Kernel import Kernel
from Layer import Layer
from static_functions import reluDerivativeMatrix

class ConvolutedLayer(Layer):


    def __init__(self,numberOfKernels : int,numRows: int,numCols : int,inputNumRows: int,inputNumCols:int):
        self.numberOfKernels : int = numberOfKernels

        self.Kernels : list[Kernel] = [] # type: ignore = 
        for i in range(numberOfKernels):
            self.Kernels.append(Kernel(numRows,numCols,inputNumRows,inputNumCols))
    

    def forwardPropogate(self,input : np.ndarray):
        for kernel in self.Kernels:
            kernel.Convolve(input)
            kernel.applyActivation()
        self.combineOutput()
        pass
    


    def backPropogate(self,output,nextLayer = None):
        k=0
        for kernel in self.Kernels:
            kernel.setErrorMatrix(nextLayer.inputMatrix[:,:,k])
            k+=1
        pass

    def updateWeightsAndBiases(self,learningRate):
        for kernel in self.Kernels:
            kernel.updateKernelAndBias(learningRate)
        pass

    def combineOutput(self):
        activationMatrices = []
        for kernel in self.Kernels:
            activationMatrices.append(kernel.activationMatrix)
        self.outputMatrix = np.stack(activationMatrices,axis=2)
        pass