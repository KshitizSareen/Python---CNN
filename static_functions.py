import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply the sigmoid activation function element-wise to the input array.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: An array where each element is transformed by the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_single(x: float) -> float:
    """
    Apply the sigmoid activation function to a single value.

    Parameters:
        x (float): A numeric value.

    Returns:
        float: Sigmoid of x.
    """
    return 1 / (1 + np.exp(-x))


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


def sigmoid_derivative(x: float) -> float:
    """
    Compute the derivative of the sigmoid function for a single value.

    Parameters:
        x (float): A numeric value.

    Returns:
        float: The derivative, computed as sigmoid(x) * (1 - sigmoid(x)).
    """
    sig = 1 / (1 + np.exp(-x))
    return sig * (1 - sig)


def sigmoid_derivative_matrix(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the sigmoid function element-wise for a matrix.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: An array where each element is the derivative of the sigmoid function,
                    computed as sigmoid(x) * (1 - sigmoid(x)).
    """
    sig = 1 / (1 + np.exp(-x))
    return sig * (1 - sig)


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


def convolve_matrix_with_filter(matrix: np.ndarray, filter_: np.ndarray) -> np.ndarray:
    """
    Convolve a 2D matrix with a given filter using valid convolution.

    Parameters:
        matrix (np.ndarray): 2D input array.
        filter_ (np.ndarray): 2D filter array.

    Returns:
        np.ndarray: The convolved output with shape 
                    (matrix_rows - filter_rows + 1, matrix_cols - filter_cols + 1).
    """
    output_rows = matrix.shape[0] - filter_.shape[0] + 1
    output_cols = matrix.shape[1] - filter_.shape[1] + 1
    convolved_output = np.zeros((output_rows, output_cols))
    for i in range(output_rows):
        for j in range(output_cols):
            region = matrix[i:i + filter_.shape[0], j:j + filter_.shape[1]]
            convolved_output[i, j] = np.sum(region * filter_)
    return convolved_output


def full_convolve2d(input_matrix: np.ndarray, filter_: np.ndarray) -> np.ndarray:
    """
    Perform a full 2D convolution between an input matrix and a filter.
    
    In full convolution, the output size is (input_dim + filter_dim - 1) in each dimension.
    The input is padded with zeros so that the filter can be applied to every possible position.

    Parameters:
        input_matrix (np.ndarray): 2D input array.
        filter_ (np.ndarray): 2D filter array.

    Returns:
        np.ndarray: The full convolution output.
    """
    i_height, i_width = input_matrix.shape
    k_height, k_width = filter_.shape

    out_height = i_height + k_height - 1
    out_width = i_width + k_width - 1

    pad_height = k_height - 1
    pad_width = k_width - 1

    padded_image = np.pad(
        input_matrix,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode='constant',
        constant_values=0
    )

    output = np.zeros((out_height, out_width))
    for i in range(out_height):
        for j in range(out_width):
            region = padded_image[i:i + k_height, j:j + k_width]
            output[i, j] = np.sum(region * filter_)
    return output
