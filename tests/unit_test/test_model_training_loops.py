import pytest
import jax

from fixtures import stein_vi_regression_example
from run_stein_vi.data.regression_toy_example import get_regression_toy_example

from stein_vi.algorithm.svgd import set_up_svgd

from stein_vi.algorithm.model_training import (no_minibatch_training_loop, data_minibatch_training_loop,
                                               particle_minibatch_training_loop,
                                               data_and_particle_minibatch_training_loop,
                                               create_particle_minibatch_indices,
                                               create_minibatches)

# TODO: this whole file for multiclass


@pytest.mark.parametrize(
    "patience_early_stopping",
    [
        1,
        2,
    ]
)
def test_no_minibatch_training_loop(stein_vi_regression_example, patience_early_stopping):
    # given
    regression_toy_example = get_regression_toy_example(num_points=100)
    z_train, y_train, z_val, y_val, _, _ = regression_toy_example
    stein_vi_regression_example.parameter.batch_size = 0
    stein_vi_regression_example.parameter.particle_batch_size = 0
    stein_vi_regression_example.parameter.num_iterations = 2
    stein_vi_regression_example.parameter.patience_early_stopping = patience_early_stopping
    stein_vi_regression_example._full_evaluation = True
    stein_vi_regression_example = set_up_svgd(stein_vi_regression_example)

    # when
    trained_stein_vi = no_minibatch_training_loop(stein_vi_regression_example, regression_toy_example)

    # then
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train, y_train)
    if patience_early_stopping > 1:
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train, y_train)
    assert stein_vi_regression_example == trained_stein_vi


@pytest.mark.parametrize(
    "patience_early_stopping",
    [
        1,
        2,
    ]
)
def test_data_minibatch_training_loop(stein_vi_regression_example, patience_early_stopping):
    # given
    key = jax.random.PRNGKey(1)
    regression_toy_example = get_regression_toy_example(num_points=100)
    z_train, y_train, z_val, y_val, _, _ = regression_toy_example
    stein_vi_regression_example.parameter.batch_size = 36
    stein_vi_regression_example.parameter.particle_batch_size = 0
    stein_vi_regression_example.parameter.num_iterations = 2
    stein_vi_regression_example.parameter.patience_early_stopping = patience_early_stopping
    stein_vi_regression_example._full_evaluation = True
    stein_vi_regression_example = set_up_svgd(stein_vi_regression_example)

    z_train_batched, y_train_batched = create_minibatches(36, z_train, y_train, key)

    # when
    trained_stein_vi = data_minibatch_training_loop(stein_vi_regression_example, regression_toy_example, key)

    # then
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train_batched[0], y_train_batched[0])
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train_batched[1], y_train_batched[1])
    if patience_early_stopping > 1:
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train_batched[0],
                                                                                  y_train_batched[0])
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train_batched[1],
                                                                                  y_train_batched[1])
    assert stein_vi_regression_example == trained_stein_vi


@pytest.mark.parametrize(
    "patience_early_stopping",
    [
        1,
        2,
    ]
)
def test_particle_minibatch_training_loop(stein_vi_regression_example, patience_early_stopping):
    # given
    key = jax.random.PRNGKey(1)
    regression_toy_example = get_regression_toy_example(num_points=100)
    z_train, y_train, z_val, y_val, _, _ = regression_toy_example
    stein_vi_regression_example.parameter.num_particles = 10
    stein_vi_regression_example.parameter.batch_size = 0
    stein_vi_regression_example.parameter.particle_batch_size = 5
    stein_vi_regression_example.parameter.num_iterations = 2
    stein_vi_regression_example.parameter.patience_early_stopping = patience_early_stopping
    stein_vi_regression_example._full_evaluation = True
    stein_vi_regression_example = set_up_svgd(stein_vi_regression_example)

    particle_indices_batches = create_particle_minibatch_indices(key,
                                                                 stein_vi_regression_example.state.particles.shape[0],
                                                                 batch_size=5)

    # when
    trained_stein_vi = particle_minibatch_training_loop(stein_vi_regression_example, regression_toy_example, key)

    # then
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train, y_train,
                                                                              particle_indices=particle_indices_batches[0])
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train, y_train,
                                                                              particle_indices=particle_indices_batches[1])
    if patience_early_stopping > 1:
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train, y_train,
                                                                                  particle_indices=particle_indices_batches[0])
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train, y_train,
                                                                                  particle_indices=particle_indices_batches[1])
    assert stein_vi_regression_example == trained_stein_vi


@pytest.mark.parametrize(
    "patience_early_stopping",
    [
        1,
        2,
    ]
)
def test_data_and_particle_minibatch_training_loop(stein_vi_regression_example, patience_early_stopping):
    # given
    key = jax.random.PRNGKey(1)
    regression_toy_example = get_regression_toy_example(num_points=100)
    z_train, y_train, z_val, y_val, _, _ = regression_toy_example
    stein_vi_regression_example.parameter.num_particles = 10
    stein_vi_regression_example.parameter.batch_size = 36
    stein_vi_regression_example.parameter.particle_batch_size = 5
    stein_vi_regression_example.parameter.num_iterations = 2
    stein_vi_regression_example.parameter.patience_early_stopping = patience_early_stopping
    stein_vi_regression_example._full_evaluation = True
    stein_vi_regression_example = set_up_svgd(stein_vi_regression_example)

    particle_indices_batches = create_particle_minibatch_indices(key,
                                                                 stein_vi_regression_example.state.particles.shape[0],
                                                                 batch_size=5)
    z_train_batched, y_train_batched = create_minibatches(36, z_train, y_train, key)

    # when
    trained_stein_vi = data_and_particle_minibatch_training_loop(stein_vi_regression_example, regression_toy_example,
                                                                 key)

    # then
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train_batched[0], y_train_batched[0],
                                                                              particle_indices=particle_indices_batches[0])
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train_batched[0], y_train_batched[0],
                                                                              particle_indices=particle_indices_batches[1])
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train_batched[1], y_train_batched[1],
                                                                              particle_indices=particle_indices_batches[0])
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train_batched[1], y_train_batched[1],
                                                                              particle_indices=particle_indices_batches[1])
    if patience_early_stopping > 1:
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train_batched[0],
                                                                                  y_train_batched[0],
                                                                                  particle_indices=
                                                                                  particle_indices_batches[0])
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train_batched[0],
                                                                                  y_train_batched[0],
                                                                                  particle_indices=
                                                                                  particle_indices_batches[1])
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train_batched[1],
                                                                                  y_train_batched[1],
                                                                                  particle_indices=
                                                                                  particle_indices_batches[0])
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train_batched[1],
                                                                                  y_train_batched[1],
                                                                                  particle_indices=
                                                                                  particle_indices_batches[1])
    assert stein_vi_regression_example == trained_stein_vi
