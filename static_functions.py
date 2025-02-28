import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply the ReLU activation function element-wise to the input array.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: An array where each element is max(0, element of x).
    """
    return np.maximum(0, x)


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
    return 1 if x > 0 else 0.01


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


def generate_random_filters(self, SEED) -> None:
    """
    Generates 'num_filters' random filters with shape (_filterSize, _filterSize)
    drawn from a Gaussian (normal) distribution, and assigns them to self._filters.
    """


    # Generate a random value between low (inclusive) and high (exclusive)
    random_value = np.random.uniform(0, SEED)
    return random_value
