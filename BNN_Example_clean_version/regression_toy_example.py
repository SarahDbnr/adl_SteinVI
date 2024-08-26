import jax
from data_handling import print_data_information, VAL_SPLIT
import math


def true_function(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    mean_outputs = x1 ** 3 - x2 ** 3 + 0.5 * x1 * x2
    return mean_outputs


def get_regression_toy_example(num_points, input_dimension=2):
    key = jax.random.PRNGKey(1)
    # Split the number of points into training and testing
    num_train = math.floor(0.8 * num_points)
    num_test = num_points - num_train

    # Generate random input data within the specified range
    key, subkey = jax.random.split(key)
    x_train = jax.random.uniform(subkey, shape=(num_train, input_dimension), minval=0, maxval=1)

    key, subkey = jax.random.split(key)
    x_test = jax.random.uniform(subkey, shape=(num_test, input_dimension), minval=0, maxval=1)

    # Generate output data using the true function
    y_train = true_function(x_train)
    y_test = true_function(x_test)

    val_size = math.ceil(len(x_train) * VAL_SPLIT)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    print_data_information(x_train, y_train, x_val, y_val, x_test, y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


