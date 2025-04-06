import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply the ReLU activation function element-wise to the input array.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: An array where each element is max(0, element of x).
    """
    return x * 0.001 if x < 0 else x


def relu_single(x: float) -> float:
    """
    Apply the ReLU activation function to a single value.

    Parameters:
        x (float): A numeric value.

    Returns:
        float: max(0, x).
    """
    return max(0, x)


def error_derivative(input_val: float, output_val: float) -> float:
    """
    Compute the derivative of the error with respect to the output.
    
    This is a simple difference, often used in mean squared error calculations.

    Parameters:
        input_val (float): The predicted value.
        output_val (float): The target value.

    Returns:
        float: The derivative (input_val - output_val).
    """
    return input_val - output_val


def relu_derivative(x: float) -> int:
    """
    Compute the derivative of the ReLU function for a single value.

    Parameters:
        x (float): A numeric value.

    Returns:
        int: 1 if x > 0, otherwise 0.
    """
    return 0.001 if x < 0 else 1


def relu_derivative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the ReLU function element-wise for a matrix.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: An array where each element is 1 if the corresponding element in x is greater than 0, otherwise 0.
    """
    return np.where(x > 0, 1, 0.01)


def mean_square_error(input_vector: np.ndarray, output_vector: np.ndarray) -> float:
    """
    Calculate the mean squared error between two vectors.

    The formula used is: 0.5 * mean((output_vector - input_vector)^2).

    Parameters:
        input_vector (np.ndarray): Predicted values.
        output_vector (np.ndarray): Target values.

    Returns:
        float: The mean squared error.
    """
    return 0.5 * np.mean((output_vector - input_vector) ** 2)




def dilate_filter(filter_: np.ndarray, stride: int) -> np.ndarray:
    """
    Dilates (upsamples) the filter by inserting zeros to account for stride in backpropagation.

    Parameters:
        filter_ (np.ndarray): 2D filter (error gradient).
        stride (int): Stride used in forward convolution.

    Returns:
        np.ndarray: Dilated filter with zeros inserted.
    """
    if stride == 1:
        return filter_  # No dilation needed

    h, w = filter_.shape
    dilated_filter = np.zeros((h * stride - (stride - 1), w * stride - (stride - 1)))
    dilated_filter[::stride, ::stride] = filter_

    return dilated_filter

def convolve_matrix_with_filter(matrix: np.ndarray, errorMatrix: np.ndarray, filter_: np.ndarray, stride: int) -> np.ndarray:
    """
    Convolve a 2D matrix with a given filter using valid convolution.
    The filter (error gradient) is first dilated to correctly backpropagate through a strided convolution.

    Parameters:
        matrix (np.ndarray): 2D input array.
        filter_ (np.ndarray): 2D filter (error gradient).
        stride (int): Stride used in forward pass.

    Returns:
        np.ndarray: The convolved output with shape 
                    (matrix_rows - filter_rows + 1, matrix_cols - filter_cols + 1).
    """

    dfilt = np.zeros(shape=filter_.shape)                # loss gradient of filter

    tmp_y = out_y = 0
    while tmp_y + filter_.shape[0] <= matrix.shape[0]:
        tmp_x = out_x = 0
        while tmp_x + filter_.shape[1] <= matrix.shape[1]:
            patch = matrix[tmp_y:tmp_y + filter_.shape[0], tmp_x:tmp_x + filter_.shape[1]]
            dfilt += np.sum(errorMatrix[out_y, out_x] * patch)
            tmp_x += stride
            out_x += 1
        tmp_y += stride
        out_y += 1
    return dfilt

def space_array(input_array: np.ndarray, step_size: int) -> np.ndarray:
    """
    Expands the input array by inserting zeros based on step_size.

    Parameters:
        input_array (np.ndarray): 2D input array.
        step_size (int): The spacing factor.

    Returns:
        np.ndarray: The expanded output array.
    """
    if step_size == 1:
        return input_array  # No dilation needed

    # Compute new dimensions
    out_rows = (input_array.shape[0] - 1) * step_size + 1
    out_cols = (input_array.shape[1] - 1) * step_size + 1

    # Initialize output array with zeros
    output = np.zeros((out_rows, out_cols))

    # Fill values at spaced indices
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            output[i * step_size, j * step_size] = input_array[i, j]

    return output



def full_convolve(input_matrix : np.ndarray, errorMatrix: np.ndarray, filter_matrix : np.ndarray,stride : int):
    input_dimension = input_matrix.shape       # input dimension

    dout = np.zeros(shape=(input_matrix.shape[0],input_matrix.shape[1]))              # loss gradient of the input to the convolution operation
    #dfilt = np.zeros(self.filters.shape)                # loss gradient of filter

    tmp_y = out_y = 0
    while tmp_y + filter_matrix.shape[0] <= input_dimension[0]:
        tmp_x = out_x = 0
        while tmp_x + filter_matrix.shape[1] <= input_dimension[1]:
            dout[tmp_y:tmp_y + filter_matrix.shape[0], tmp_x:tmp_x + filter_matrix.shape[1]] += errorMatrix[out_y, out_x] * filter_matrix
            tmp_x += stride
            out_x += 1
        tmp_y += stride
        out_y += 1
    return dout  


def generate_random_filters(self, SEED) -> None:
    """
    Generates 'num_filters' random filters with shape (_filterSize, _filterSize)
    drawn from a Gaussian (normal) distribution, and assigns them to self._filters.
    """


    # Generate a random value between low (inclusive) and high (exclusive)
    random_value = np.random.uniform(0, SEED)
    return random_value

def softmax(x : np.ndarray):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy_to_softmax_derivative(x: np.ndarray,output: np.ndarray):
    return output - x

def cross_entropy_loss(x: np.ndarray, output: np.ndarray):
    """
    Computes the cross-entropy loss.

    Parameters:
    x (np.ndarray): The ground truth labels (one-hot encoded or probabilities).
    output (np.ndarray): The predicted probabilities.

    Returns:
    float: The cross-entropy loss.
    """
    # Avoid log(0) issues by adding a small constant (epsilon)
    epsilon = 1e-12
    output = np.clip(x, epsilon, 1. - epsilon)  # Clip values to prevent log(0)
    # Compute cross-entropy loss
    loss = -np.sum(output * np.log(x)) / x.shape[0]  # Averaged over batch size
    
    return loss
