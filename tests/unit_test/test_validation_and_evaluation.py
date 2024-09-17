import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from run_stein_vi.model.BNN_Model import FlexibleSimpleNN
from src.algorithm.get_posteriori import link_function
from src.metrics.validation_and_evaluation import (calculate_number_of_different_classified_by_particles,
                                                   get_most_common_class_over_particles,get_most_common_class, compute_confidence_intervals_with_2_neurons)


def test_calculate_number_of_different_classified_by_particles():
    # given
    num_particles = 20
    key = jax.random.PRNGKey(0)
    predictions = jax.random.randint(key, (num_particles, 10), minval=1, maxval=4)
    # when
    num_differently_classified = calculate_number_of_different_classified_by_particles(predictions)
    # then
    first_column = predictions[:, 0]
    argmax_first_column = first_column[jnp.argmax(first_column)]
    print(first_column, argmax_first_column)
    assert first_column[first_column != argmax_first_column].size == num_differently_classified[0]


def test_get_most_common_class_over_particles():
    # given
    num_particles = 20
    key = jax.random.PRNGKey(0)
    predictions = jax.random.randint(key, (num_particles, 10), minval=1, maxval=4)
    # when
    most_common_class_over_particles = get_most_common_class_over_particles(predictions)
    # then
    unique_vals, col_counts = jnp.unique(predictions[:, 0], return_counts=True)
    max_index = jnp.argmax(col_counts)
    assert most_common_class_over_particles[0] == unique_vals[max_index]



def test_get_most_common_class():
    # Test case 1: Single most common class
    column = jnp.array([1, 2, 2, 3, 2])
    expected_most_common = 2
    result = get_most_common_class(column)
    assert result == expected_most_common, f"Expected {expected_most_common}, but got {result}"
    
    # Test case 2: Tie case (should return the first most common by index)
    column = jnp.array([1, 2, 2, 3, 3])
    expected_most_common = 2  # `2` appears first in case of tie with `3`
    result = get_most_common_class(column)
    assert result == expected_most_common, f"Expected {expected_most_common}, but got {result}"

    # Test case 3: All elements are the same
    column = jnp.array([1, 1, 1, 1, 1])
    expected_most_common = 1
    result = get_most_common_class(column)
    assert result == expected_most_common, f"Expected {expected_most_common}, but got {result}"
    
    # Test case 4: Each element appears once
    column = jnp.array([1, 2, 3, 4, 5])
    expected_most_common = 1  # Returns the first element in case of all unique elements
    result = get_most_common_class(column)
    assert result == expected_most_common, f"Expected {expected_most_common}, but got {result}"
    
    # Test case 5: Large array with clear majority
    column = jnp.array([5] * 100 + [3] * 10 + [1] * 5)
    expected_most_common = 5
    result = get_most_common_class(column)
    assert result == expected_most_common, f"Expected {expected_most_common}, but got {result}"

    # Test case 6: Edge case with a single element array
    column = jnp.array([42])
    expected_most_common = 42
    result = get_most_common_class(column)
    assert result == expected_most_common, f"Expected {expected_most_common}, but got {result}"



def test_compute_confidence_intervals_with_2_neurons():
    # Set up the neural network model
    model = FlexibleSimpleNN(
        hidden_layers=[2],  # Example with 1 hidden layer of 2 neurons
        output_size=2,  # Output size for regression (mean and variance)
        activation=nn.relu,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros,
        use_for_regression=True
    )
    
    # Initialize random key and input data
    key = random.PRNGKey(0)
    input_size = 4
    dz = jnp.ones((1, input_size))  # Input data of ones, batch size 1

    # Initialize model parameters
    params = model.init(key, dz)
    
    # Simulate multiple particles for Bayesian Neural Networks
    num_particles = 5
    particles = jax.vmap(lambda _: ravel_pytree(params)[0])(jnp.arange(num_particles))

    # Create a mock tree_def function that flattens and unflattens the parameters
    _, tree_def = ravel_pytree(params)
    
    # Prepare a mock output structure that mimics a model output
    class MockOutput:
        def __init__(self, particles):
            self.particles = particles
    
    out = MockOutput(particles)
    
    # Run the function being tested
    mean_star, variance_star = compute_confidence_intervals_with_2_neurons(model, tree_def, out, dz)
    
    # Check that the outputs have the correct shape
    assert mean_star.shape == (), f"Expected mean_star shape (), but got {mean_star.shape}"
    assert variance_star.shape == (), f"Expected variance_star shape (), but got {variance_star.shape}"
    
    # Check that mean_star and variance_star are numeric and finite
    assert jnp.isfinite(mean_star).all(), "mean_star contains non-finite values."
    assert jnp.isfinite(variance_star).all(), "variance_star contains non-finite values."
    
    # Check that variance_star is non-negative (as variances must be)
    assert jnp.all(variance_star >= 0), "variance_star contains negative values, which is invalid for variance."
    
    # Calculate expected values for mean_star and variance_star manually
    predictions, precision = jax.vmap(lambda p: model.predict(tree_def(p), dz))(out.particles)
    predictions = predictions.squeeze()
    precision = precision.squeeze()
    expected_mean_star = predictions.mean(0)  # Averaging over particles
    
    squared_means_i = jnp.square(predictions)
    variance_i = jax.vmap(lambda p: link_function(p))(precision)
    expected_variance_star = variance_i.mean(0) + squared_means_i.mean(0) - jnp.square(expected_mean_star)
    
    # Assert the values of mean_star and variance_star
    assert jnp.allclose(mean_star, expected_mean_star, rtol=1e-5), \
        f"Expected mean_star: {expected_mean_star}, but got: {mean_star}"
    
    assert jnp.allclose(variance_star, expected_variance_star, rtol=1e-5), \
        f"Expected variance_star: {expected_variance_star}, but got:  {variance_star}"
    