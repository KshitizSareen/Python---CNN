import numpy as np

class MaxPool:



    def __init__(self,numRows:int,numCols:int):
        self.numRows = numRows
        self.numCols = numCols
        self.recordedPositions: list[float] # type: ignore
        pass

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

    def maxPooling(self,image):
        rows,cols = image.shape
        rows = rows - self.numRows
        cols = cols - self.numCols
        outputMatrix = np.zeros((rows,cols))
        for i in range(rows):
            for j in range(cols):
                maxValue,maxIndexRow,maxIndexCol = self.findMaxValueAndPostion(image[i:i+rows,j:j+cols])
                self.recordedPositions((maxIndexRow,maxIndexCol))
                outputMatrix[i,j] = maxValue
        
        return outputMatrix