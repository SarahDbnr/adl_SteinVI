import pytest
import jax
import jax.numpy as jnp
import math
from unittest.mock import patch
from run_stein_vi.data.regression_toy_example import generate_data, get_regression_toy_example  # Replace with actual import path

def test_generate_data_output_shape():
    """Test that generate_data returns the correct output shape."""
    key = jax.random.PRNGKey(0)
    
    # Input array with 100 samples, each with 2 features
    x = jax.random.uniform(key, shape=(100, 2))
    
    # Generate output data
    y = generate_data(x, key)
    
    # Check that the output shape matches the input shape (number of samples)
    assert y.shape == (100,), "The output shape should be (100,) for 100 input samples."

def test_generate_data_noise_effect():
    """Test that generate_data adds noise to the outputs."""
    key = jax.random.PRNGKey(0)
    
    # Input array with 10 samples, each with 2 features
    x = jax.random.uniform(key, shape=(10, 2))
    
    # Generate output data with different keys (different noise)
    y1 = generate_data(x, key)
    key = jax.random.PRNGKey(1)
    y2 = generate_data(x, key)
    
    # Ensure that the outputs are different due to added noise
    assert not jnp.array_equal(y1, y2), "Outputs with different random keys should be different due to noise."

def test_generate_data_consistency():
    """Test that generate_data is consistent when using the same random key."""
    key = jax.random.PRNGKey(0)
    
    # Input array with 10 samples, each with 2 features
    x = jax.random.uniform(key, shape=(10, 2))
    
    # Generate output data with the same key
    y1 = generate_data(x, key)
    y2 = generate_data(x, key)
    
    # Ensure that the outputs are identical since the same key is used
    assert jnp.array_equal(y1, y2), "Outputs with the same random key should be identical."




def test_get_regression_toy_example_data_split():
    """Test that the data is correctly split into training, validation, and test sets."""
    key = jax.random.PRNGKey(0)
    
    # Generate the toy dataset with 100 points
    x_train, y_train, x_val, y_val, x_test, y_test = get_regression_toy_example(100, val_split=0.1, key=key)
    
    # Check that the data is split correctly
    assert len(x_train) == 72, "Training set should contain 72 samples (90% of 80 points)."
    assert len(x_val) == 8, "Validation set should contain 8 samples (10% of 80 points)."
    assert len(x_test) == 20, "Test set should contain 20 samples (20% of total points)."


def test_get_regression_toy_example_determinism():
    """Test that get_regression_toy_example is deterministic given the same random key."""
    key = jax.random.PRNGKey(0)
    
    # Generate the toy dataset twice with the same key
    result_1 = get_regression_toy_example(100, key=key)
    result_2 = get_regression_toy_example(100, key=key)
    
    # Ensure that the two generated datasets are identical
    for arr_1, arr_2 in zip(result_1, result_2):
        assert jnp.array_equal(arr_1, arr_2), "The two datasets should be identical when using the same random key."


@patch('run_stein_vi.data.regression_toy_example.print_data_information')
def test_get_regression_toy_example_output_shape(mock_print_data_information):
    """Test that get_regression_toy_example returns data with correct shapes and prints information."""
    key = jax.random.PRNGKey(0)
    
    # Generate the toy dataset with 100 points
    x_train, y_train, x_val, y_val, x_test, y_test = get_regression_toy_example(100, key=key)
    
    # Check the shapes of the returned data
    assert x_train.shape == (72, 2), "Training data should have 72 samples and 2 features."
    assert y_train.shape == (72,), "Training labels should have 72 samples."
    assert x_val.shape == (8, 2), "Validation data should have 8 samples and 2 features."
    assert y_val.shape == (8,), "Validation labels should have 8 samples."
    assert x_test.shape == (20, 2), "Test data should have 20 samples and 2 features."
    assert y_test.shape == (20,), "Test labels should have 20 samples."
    
    # Check if print_data_information was called once with the correct arguments
    mock_print_data_information.assert_called_once_with(x_train, y_train, x_val, y_val, x_test, y_test)
