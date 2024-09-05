import jax
import jax.numpy as jnp

from BNN_Example_clean_version.validation_and_evaluation import (calculate_number_of_different_classified_by_particles,
                                                                 get_most_common_class_over_particles)


def test_calculate_number_of_different_classified_by_particles():
    # given
    num_particles = 20
    key = jax.random.PRNGKey(0)
    predictions = jax.random.randint(key, (num_particles, 10), minval=1, maxval=4)
    # when
    num_differently_classified = calculate_number_of_different_classified_by_particles(predictions)
    # then
    assert num_differently_classified.sum == num_particles*10


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
