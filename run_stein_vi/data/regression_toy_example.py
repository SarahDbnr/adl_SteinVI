import jax
import math
from run_stein_vi.data.data_handling import print_data_information


def generate_data(x, key = jax.random.PRNGKey(1)):
    """
    Generates output data for a given input matrix x using a specific polynomial function with noise.

    Args:
        x (jax.numpy.ndarray): An array of shape (n_samples, n_features) where each row represents a sample
                               with features used to compute the output.
        key (jax.random.PRNGKey): A JAX PRNG key used for deterministic selection of samples.

    Returns:
        jax.numpy.ndarray: An array of shape (n_samples,) containing the computed outputs with added noise.
    """
    
    x1 = x[:, 0]
    x2 = x[:, 1]
    function_outputs = x1 ** 3 - x2 ** 3 + 0.5 * x1 * x2
    noise = jax.random.normal(key, shape=function_outputs.shape)
    return function_outputs + noise


def get_regression_toy_example(num_points, input_dimension=2, key = jax.random.PRNGKey(1), val_split=0.1):
    """
    Generates a synthetic regression dataset based on a specified polynomial function with added noise,
    and splits it into training, validation, and test sets.

    Args:
        num_points (int): Total number of data points to generate.
        input_dimension (int, optional): Number of input features for each data point. Defaults to 2.
        key (jax.random.PRNGKey): A JAX PRNG key used for deterministic selection of samples.
        val_split (float): Fraction of the data used for validation during training.

    Returns:
        tuple: A tuple containing six elements:
               - x_train (jax.numpy.ndarray): Training data features.
               - y_train (jax.numpy.ndarray): Training data outputs.
               - x_val (jax.numpy.ndarray): Validation data features.
               - y_val (jax.numpy.ndarray): Validation data outputs.
               - x_test (jax.numpy.ndarray): Test data features.
               - y_test (jax.numpy.ndarray): Test data outputs.
    """
    
    # Split the number of points into training and testing
    num_train = math.floor(0.8 * num_points)
    num_test = num_points - num_train

    # Generate random input data within the specified range
    key, subkey = jax.random.split(key)
    x_train = jax.random.uniform(subkey, shape=(num_train, input_dimension), minval=0, maxval=1)

    key, subkey = jax.random.split(key)
    x_test = jax.random.uniform(subkey, shape=(num_test, input_dimension), minval=0, maxval=1)

    # Generate output data using the true function
    y_train = generate_data(x_train)
    y_test = generate_data(x_test)

    val_size = math.ceil(len(x_train) * val_split)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    print_data_information(x_train, y_train, x_val, y_val, x_test, y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test
