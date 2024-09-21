import pytest
import jax
import math
import tensorflow as tf
from optax import adam, exponential_decay

from run_stein_vi.data.data_handling import apply_data_settings_keras

from stein_vi.Classes.SteinVI_BNN_Class import SteinVI_BNN
from run_stein_vi.model.BNN_Model import build_model


@pytest.fixture
def stein_vi_regression_example(get_regression_toy_example):
    key = jax.random.PRNGKey(1)

    regression_toy_example = get_regression_toy_example
    z_train, _, _, _, z_test, y_test = regression_toy_example

    optimizer = adam(
        exponential_decay(
            init_value=0.1,
            transition_steps=20,
            decay_rate=0.95,
            staircase=True
        )
    )

    nnet_model = build_model(output_size=2)

    steinvi_regression = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=True, optimizer=optimizer)

    return steinvi_regression


@pytest.fixture
def stein_vi_multiclass_example():
    key = jax.random.PRNGKey(1)

    mnist = tf.keras.datasets.mnist
    mnist_dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)
    z_train, _, _, _, z_test, y_test = mnist_dataset

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )
    nnet_model = build_model(output_size=10, hidden_layers=(200, 70, 40))

    stein_vi_multiclass = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, optimizer=optimizer,
                                      batch_size=300, num_iterations=30, num_particles=5)

    return stein_vi_multiclass


@pytest.fixture
def get_regression_toy_example():
    """
    Generates a synthetic regression dataset based on a specified polynomial function with added noise,
    and splits it into training, validation, and test sets.

    Args:
        num_points (int): Total number of data points to generate.

    Returns:
        tuple: A tuple containing six elements:
               - x_train (jax.numpy.ndarray): Training data features.
               - y_train (jax.numpy.ndarray): Training data outputs.
               - x_val (jax.numpy.ndarray): Validation data features.
               - y_val (jax.numpy.ndarray): Validation data outputs.
               - x_test (jax.numpy.ndarray): Test data features.
               - y_test (jax.numpy.ndarray): Test data outputs.
    """
    num_points = 100
    val_split = 0.1
    key = jax.random.PRNGKey(1)
    input_dimension = 2

    # Split the number of points into training and testing
    num_train = math.floor(0.8 * num_points)
    num_test = num_points - num_train

    # Generate random input data within the specified range
    key, subkey = jax.random.split(key)
    x_train = jax.random.uniform(subkey, shape=(num_train, input_dimension), minval=0, maxval=1)

    key, subkey = jax.random.split(key)
    x_test = jax.random.uniform(subkey, shape=(num_test, input_dimension), minval=0, maxval=1)

    # Generate output data using the true function
    y_train = generate_data(x_train, key)
    y_test = generate_data(x_test, key)

    val_size = math.ceil(len(x_train) * val_split)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    return x_train, y_train, x_val, y_val, x_test, y_test


def generate_data(x, key):
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


@pytest.fixture
def get_MNIST():
    mnist = tf.keras.datasets.mnist
    mnist_dataset = apply_data_settings_keras(mnist.load_data())
    return mnist_dataset


def apply_data_settings_keras(new_dataset, with_flattening=False, val_split=0.1):
    """
    Processes a dataset obtained from Keras, normalizing and optionally flattening it, and splitting it into training, validation, and test sets. Used for image data with values between 0 and 255.

    Args:
        new_dataset (tuple): A tuple containing training and test datasets as (x_train, y_train), (x_test, y_test).
        with_flattening (bool): If True, flattens the output arrays.
        val_split (float): Fraction of the data used for validation during training.
    Returns:
        tuple: Tuple containing processed training, validation, and test data as (x_train, y_train, x_val, y_val, x_test, y_test).
    """
    (x_train, y_train), (x_test, y_test) = new_dataset
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    if with_flattening:
        y_train = y_train.flatten()
        y_test = y_test.flatten()

    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    return x_train, y_train, x_val, y_val, x_test, y_test
