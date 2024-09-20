import pytest
import jax
import jax.numpy as jnp
import tensorflow as tf

from fixtures import stein_vi_regression_example, stein_vi_multiclass_example
from run_stein_vi.data.data_handling import apply_data_settings_keras
from run_stein_vi.data.regression_toy_example import get_regression_toy_example
from stein_vi.algorithm.svgd import initialize_svgd_state
from stein_vi.algorithm.get_posteriori import link_function

from stein_vi.metrics.validation_and_evaluation import (get_evaluation_metrics_over_predictions, calculate_mse,
                                                        calculate_accuracy, calculate_mean_span_over_particles,
                                                        calculate_number_of_different_classified_by_particles,
                                                        get_most_common_class_over_particles, get_most_common_class,
                                                        compute_confidence_intervals_with_2_neurons)


def test_get_evaluation_metrics_over_predictions_regression(stein_vi_regression_example, stein_vi_multiclass_example):
    # given
    regression_toy_example = get_regression_toy_example(num_points=100)
    z_train, y_train, _, _, _, _ = regression_toy_example
    svgd_vi = stein_vi_regression_example
    state, _ = initialize_svgd_state(svgd_vi)
    nnet_model = svgd_vi.nnet
    # when
    mse, averaged_var, prediction = get_evaluation_metrics_over_predictions(state, nnet_model, z_train, y_train,
                                                                            model_regression=True, print_eva=True)
    # then
    expected_predictions, precisions = jax.vmap(lambda p: nnet_model.predict(p, z_train))(state.particles)
    expected_mse = calculate_mse(expected_predictions.squeeze(), y_train)
    scale = jax.vmap(lambda p: link_function(p))(precisions.squeeze())
    expected_averaged_var = jnp.sqrt(scale).mean()
    assert jnp.all(prediction == expected_predictions)
    assert jnp.all(mse == expected_mse)
    assert jnp.all(averaged_var == expected_averaged_var)


def test_get_evaluation_metrics_over_predictions_multiclass(stein_vi_regression_example, stein_vi_multiclass_example):
    # given
    mnist = tf.keras.datasets.mnist
    mnist_dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)
    z_train, y_train, _, _, _, _ = mnist_dataset
    svgd_vi = stein_vi_multiclass_example
    state, _ = initialize_svgd_state(svgd_vi)
    nnet_model = svgd_vi.nnet
    # when
    accuracy, _, prediction = get_evaluation_metrics_over_predictions(state, nnet_model, z_train, y_train,
                                                                      model_regression=False, print_eva=True)
    # then
    predictions, precisions = jax.vmap(lambda p: nnet_model.predict(p, z_train))(state.particles)
    expected_accuracy = calculate_accuracy(precisions, y_train)
    assert expected_accuracy == accuracy


def test_calculate_mse():
    # given
    predictions = jnp.array([[[1, ], [1, ]], [[0, ], [0, ]]])
    true_output = jnp.array([1, 1],)
    # when
    mse = calculate_mse(predictions, true_output)
    # then
    expected_mse = jnp.sqrt(((1+1)/2 - 1)**2 + ((0+0)/2 -1)**2)/4
    assert mse == expected_mse


def test_calculate_accuracy():
    # given
    number_particles = 4
    x_dim = 3
    num_classes = 2
    precisions = jnp.array([[[1, 0], [1, 0], [0, 1]], [[1, 0], [1, 0], [0, 1]], [[1, 0], [1, 0], [0, 1]], [[1, 0], [1, 0], [0, 1]]])
    true_output = jnp.array([0, 0, 0], )
    # when
    accuracy = calculate_accuracy(precisions, true_output)
    averaged_precision = precisions.mean(0)
    predicted_classes = jnp.argmax(averaged_precision, axis=-1)
    # one class is wrongly predicted
    # then
    print(predicted_classes)
    assert precisions.shape == (number_particles, x_dim, num_classes)
    assert accuracy == 2 / 3


def test_calculate_mean_span_over_particles():
    predictions = jnp.array([[[4, ], [4, ]], [[1, ], [1, ]],
                             [[2, ], [2, ]], [[1, ], [1, ]],
                             [[2, ], [2, ]], [[0, ], [0, ]],
                             [[1, ], [1, ]], [[0, ], [0, ]],
                             [[1, ], [1, ]], [[-1, ], [-1, ]]])
    # TODO: make a bigger array that has number not within 5% with chat gpt
    # when
    span = calculate_mean_span_over_particles(predictions.squeeze())
    # then
    assert span == 5


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
    # given
    column = jnp.array([1, 2, 3, 3, 3, 3, 3, 2, 1, 3, 3, 3, 3, 3, 2, 2, 1, 1])
    # when
    most_common_class = get_most_common_class(column)
    # then
    assert most_common_class == 3


def test_compute_confidence_intervals_with_2_neurons(regression_toy_examplestein_vi_regression_example):
    # given
    regression_toy_example = get_regression_toy_example(num_points=100)
    z_train, y_train, _, _, _, _ = regression_toy_example
    nnet_model = stein_vi_regression_example.nnet
    state, _ = initialize_svgd_state(stein_vi_regression_example)
    # when
    mean_star, variance_star = compute_confidence_intervals_with_2_neurons(nnet_model, out=state, dz=z_train)
    # then
    # TODO: ask chatgpt
