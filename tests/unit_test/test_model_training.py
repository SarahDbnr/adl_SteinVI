import pytest
from _pytest.capture import CaptureResult
import jax
import jax.numpy as jnp

from fixtures import stein_vi_regression_example, stein_vi_multiclass_example, get_regression_toy_example
from stein_vi.algorithm.svgd import set_up_svgd

from stein_vi.algorithm.model_training import (create_particle_minibatch_indices,
                                               create_minibatches, shuffle_data,
                                               get_evaluation_and_apply_early_stopping_logic,
                                               early_stopping_fn)


@pytest.fixture
def no_print():
    return CaptureResult(
        out='Training data shape: (7200, 2), Training labels shape: (7200,)\nValidation data shape: (800, 2), Validation labels shape: (800,)\nTest data shape: (2000, 2), Test labels shape: (2000,)\nTraining data shape: (72, 2), Training labels shape: (72,)\nValidation data shape: (8, 2), Validation labels shape: (8,)\nTest data shape: (20, 2), Test labels shape: (20,)\n',
        err='')


def test_create_minibatches(get_regression_toy_example):
    # given
    key = jax.random.PRNGKey(1)
    z_train, y_train, z_val, y_val, _, _ = get_regression_toy_example
    batch_size = 8
    # when
    minibatches = create_minibatches(batch_size, z_train, y_train, key)
    # then
    z_train, y_train = shuffle_data(key, z_train, y_train)
    assert jnp.all(minibatches[0][0] == z_train[0:8])
    assert jnp.all(minibatches[0][8] == z_train[64:72])
    assert len(minibatches[0]) == len(z_train) / 8
    assert jnp.all(minibatches[1][0] == y_train[0:8])
    assert jnp.all(minibatches[1][8] == y_train[64:72])
    assert len(minibatches[1]) == len(y_train) / 8


@pytest.mark.parametrize(
    "batch_size, expected_length",
    [
        (2, 5),
        (11, 1),
    ]
)
def test_create_particle_minibatch_indices(batch_size, expected_length):
    # given
    key = jax.random.PRNGKey(1)
    num_particles = 10
    # when
    indices = create_particle_minibatch_indices(key, num_particles, batch_size)
    # then
    assert len(indices) == expected_length
    flattened_indices = jnp.array(indices).ravel()
    sorted_indices = jnp.sort(flattened_indices)
    expected_indices = jnp.arange(10)
    assert jnp.all(sorted_indices == expected_indices)


# TODO: check
@pytest.mark.parametrize(
    "mode_training_print, iteration, check_no_print_expected",
    [
        ('full', 1, False),
        ('reduced', 9, True),
        ('reduced', 10, False),
        ('none', 1, True),
    ]
)
def test_get_evaluation_and_apply_early_stopping_logic(capsys, stein_vi_regression_example, get_regression_toy_example,
                                                       no_print, mode_training_print, iteration,
                                                       check_no_print_expected):
    # given
    z_train, y_train, z_val, y_val, _, _ = get_regression_toy_example
    patience_counter = 0
    stein_vi_regression_example.parameter.num_iterations = 100
    best_eval_metric = 0
    stein_vi_regression_example.handler.set_training_print_mode(mode_training_print)

    set_up_svgd(stein_vi_regression_example)
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train, y_train)
    keep_len_1 = len(stein_vi_regression_example.evaluation_metrics_1)
    keep_len_2 = len(stein_vi_regression_example.evaluation_metrics_2)

    # when
    best_metric, patience_counter = get_evaluation_and_apply_early_stopping_logic(
        stein_vi_regression_example, z_val, y_val, iteration, best_eval_metric,
        patience_counter)
    # then
    assert best_metric == best_eval_metric
    assert len(stein_vi_regression_example.evaluation_metrics_1) == keep_len_1 + 1
    assert len(stein_vi_regression_example.evaluation_metrics_2) == keep_len_2 + 1
    assert patience_counter == 0
    capture = capsys.readouterr()
    check_no_print = (capture == no_print)
    assert check_no_print == check_no_print_expected


@pytest.mark.parametrize(
    "current_metrics, expected_best_metrics, expected_patience_counter, regression",
    [
        (9.994, 9.994, 0, True),
        (9.995, 9.995, 0, True),
        (9.996, 10, 1, True),
        (10.006, 10.006, 0, False),
        (10.005, 10.005, 0, False),
        (10.004, 10, 1, False),
    ]
)
def test_early_stopping_fn(current_metrics, expected_best_metrics, expected_patience_counter, regression,
                           stein_vi_regression_example, stein_vi_multiclass_example):
    # given
    best_metrics = 10
    patience_counter = 0
    if regression:
        parameter = stein_vi_regression_example.parameter
    else:
        parameter = stein_vi_multiclass_example.parameter
    # when
    patience_counter, best_metrics = early_stopping_fn(current_metrics, best_metrics, patience_counter, parameter,
                                                       regression)
    # then
    assert patience_counter == expected_patience_counter
    assert best_metrics == expected_best_metrics
