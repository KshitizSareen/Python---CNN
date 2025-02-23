import numpy as np
def relu(x):
    return np.maximum(0, x)

def relu_single(x):
    return max(0,x)

def ErrorDerivative(input,output):
    return input - output

def reluDerivative(input):
    if input>0:
        return 1
    return 0

def reluDerivativeMatrix(input : np.ndarray):
    return np.where(input > 0, 1, 0)

def meanSquareError(inputVector: np.ndarray,outputVector: np.ndarray):
    return (1/2)*np.mean((outputVector - inputVector) ** 2)

def convolveMatrixWithFilter(matrix: np.ndarray,filter: np.ndarray):
    convolvedOutput : np.ndarray = np.zeros((matrix.shape[0]-filter.shape[0]+1,matrix.shape[1]-filter.shape[1]+1))
    for i in range(convolvedOutput.shape[0]):
        for j in range(convolvedOutput.shape[1]):
            convolvedValue = np.sum((matrix[i:i+filter.shape[0],j:j+filter.shape[1]] * filter))
            convolvedOutput[i,j] = convolvedValue
    return convolvedOutput


def full_convolve2d(input: np.ndarray, filter: np.ndarray) -> np.ndarray:
    
    i_height, i_width = input.shape
    k_height, k_width = filter.shape
    
    # Calculate the dimensions of the output array
    out_height = i_height + k_height - 1
    out_width = i_width + k_width - 1
    
    # Pad the input image with zeros so that the kernel can be applied to every position
    pad_height = k_height - 1
    pad_width = k_width - 1
    padded_image = np.pad(input, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    # Prepare an output array
    output = np.zeros((out_height, out_width))
    
    # Perform the convolution operation
    for i in range(out_height):
        for j in range(out_width):
            # Extract the current region from the padded image
            region = padded_image[i:i+k_height, j:j+k_width]
            # Compute element-wise multiplication and sum the result
            output[i, j] = np.sum(region * filter)
    
    return output