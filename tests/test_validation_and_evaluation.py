import jax
import jax.numpy as jnp

from src.metrics.validation_and_evaluation import (calculate_number_of_different_classified_by_particles,
                                                   get_most_common_class_over_particles)


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


def test_compute_confidence_intervals_with_2_neurons():
    # Define a simple mock model
    class MockModel:
        def predict(self, params, dz):
            # Return controlled predictions and precision
            return dz * 2, jnp.ones_like(dz) * 0.5

    # Create mock data
    mock_dz = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # Input dataset
    mock_particles = jnp.array([jnp.ones_like(mock_dz), jnp.ones_like(mock_dz) * 2])  # Two sets of parameters/particles

    # Create a mock output object
    class MockOut:
        def __init__(self, particles):
            self.particles = particles

    mock_out = MockOut(mock_particles)

    # Tree definition is just an identity function for this mock example
    def mock_tree_def(p):
        return p

    # Instantiate the model
    mock_model = MockModel()

    # Compute the confidence intervals
    mean_star, variance_star = compute_confidence_intervals_with_2_neurons(
        mock_model, mock_tree_def, mock_out, mock_dz
    )

    # Manually compute expected results for this simple case
    expected_predictions = jnp.array([[2.0, 4.0], [4.0, 8.0]])  # As per mock_model
    expected_mean_star = expected_predictions.mean(0)  # Should be [3.0, 6.0]
    squared_means_i = jnp.square(expected_predictions)
    variance_i = jnp.array([[0.5, 0.5], [0.5, 0.5]])  # As per mock_model
    expected_variance_star = variance_i.mean(0) + squared_means_i.mean(0) - jnp.square(expected_mean_star)

    # Assertions
    assert jnp.allclose(mean_star, expected_mean_star), f"Expected {expected_mean_star}, but got {mean_star}"
    assert jnp.allclose(variance_star, expected_variance_star), f"Expected {expected_variance_star}, but got {variance_star}"
