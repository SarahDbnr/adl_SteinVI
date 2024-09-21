import pytest
import jax
import jax.numpy as jnp
import copy

from fixtures import stein_vi_regression_example, get_regression_toy_example

from stein_vi.algorithm.svgd import set_up_svgd

from stein_vi.algorithm.model_training import (no_minibatch_training_loop, data_minibatch_training_loop,
                                               particle_minibatch_training_loop,
                                               data_and_particle_minibatch_training_loop,
                                               create_particle_minibatch_indices,
                                               create_minibatches)


@pytest.mark.parametrize(
    "patience_early_stopping",
    [
        1,
        2,
    ]
)
def test_no_minibatch_training_loop(stein_vi_regression_example, get_regression_toy_example, patience_early_stopping):
    # given
    regression_toy_example = get_regression_toy_example
    z_train, y_train, z_val, y_val, _, _ = regression_toy_example
    stein_vi_regression_example.parameter.batch_size = 0
    stein_vi_regression_example.parameter.particle_batch_size = 0
    stein_vi_regression_example.parameter.num_iterations = 2
    stein_vi_regression_example.parameter.warm_up_iterations_early_stopping = 0
    stein_vi_regression_example.parameter.early_stopping = True
    stein_vi_regression_example.parameter.patience_early_stopping = patience_early_stopping
    stein_vi_regression_example.parameter.min_delta_early_stopping = jnp.inf
    stein_vi_regression_example._full_evaluation = True
    set_up_svgd(stein_vi_regression_example)
    stein_vi_regression_example_copy = copy.copy(stein_vi_regression_example)

    # when
    no_minibatch_training_loop(stein_vi_regression_example_copy, regression_toy_example)

    # then
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train, y_train)
    if patience_early_stopping > 1:
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train, y_train)
    assert jnp.all(stein_vi_regression_example.state.particles == stein_vi_regression_example_copy.state.particles)


@pytest.mark.parametrize(
    "patience_early_stopping",
    [
        1,
        2,
    ]
)
def test_data_minibatch_training_loop(stein_vi_regression_example, get_regression_toy_example, patience_early_stopping):
    # given
    key = jax.random.PRNGKey(1)
    regression_toy_example = get_regression_toy_example
    z_train, y_train, z_val, y_val, _, _ = regression_toy_example
    stein_vi_regression_example.parameter.batch_size = 36
    stein_vi_regression_example.parameter.particle_batch_size = 0
    stein_vi_regression_example.parameter.num_iterations = 2
    stein_vi_regression_example.parameter.warm_up_iterations_early_stopping = 0
    stein_vi_regression_example.parameter.early_stopping = True
    stein_vi_regression_example.parameter.patience_early_stopping = patience_early_stopping
    stein_vi_regression_example.parameter.min_delta_early_stopping = jnp.inf
    stein_vi_regression_example._full_evaluation = True
    set_up_svgd(stein_vi_regression_example)
    stein_vi_regression_example_copy = copy.copy(stein_vi_regression_example)

    # when
    data_minibatch_training_loop(stein_vi_regression_example_copy, regression_toy_example, key)

    # then
    key_loop, _ = jax.random.split(key)
    z_train_batched, y_train_batched = create_minibatches(stein_vi_regression_example.parameter.batch_size, z_train,
                                                          y_train, key_loop)
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train_batched[0], y_train_batched[0])
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train_batched[1], y_train_batched[1])
    if patience_early_stopping > 1:
        key_loop, _ = jax.random.split(key_loop)
        z_train_batched, y_train_batched = create_minibatches(stein_vi_regression_example.parameter.batch_size, z_train,
                                                              y_train, key_loop)
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train_batched[0],
                                                                                  y_train_batched[0])
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train_batched[1],
                                                                                  y_train_batched[1])

    assert jnp.all(stein_vi_regression_example.state.particles == stein_vi_regression_example_copy.state.particles)


@pytest.mark.parametrize(
    "patience_early_stopping",
    [
        1,
        2,
    ]
)
def test_particle_minibatch_training_loop(stein_vi_regression_example, get_regression_toy_example, patience_early_stopping):
    # given
    key = jax.random.PRNGKey(1)
    regression_toy_example = get_regression_toy_example
    z_train, y_train, z_val, y_val, _, _ = regression_toy_example
    stein_vi_regression_example.parameter.num_particles = 10
    stein_vi_regression_example.parameter.batch_size = 0
    stein_vi_regression_example.parameter.particle_batch_size = 5
    stein_vi_regression_example.parameter.num_iterations = 2
    stein_vi_regression_example.parameter.warm_up_iterations_early_stopping = 0
    stein_vi_regression_example.parameter.early_stopping = True
    stein_vi_regression_example.parameter.patience_early_stopping = patience_early_stopping
    stein_vi_regression_example.parameter.min_delta_early_stopping = jnp.inf
    stein_vi_regression_example._full_evaluation = True
    set_up_svgd(stein_vi_regression_example)
    stein_vi_regression_example_copy = copy.copy(stein_vi_regression_example)

    # when
    particle_minibatch_training_loop(stein_vi_regression_example_copy, regression_toy_example, key)

    # then
    key_loop, _ = jax.random.split(key)
    particle_indices_batches = create_particle_minibatch_indices(key_loop,
                                                                 stein_vi_regression_example.state.particles.shape[0],
                                                                 batch_size=5)
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train, y_train,
                                                                              particle_indices=particle_indices_batches[0])
    stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                              z_train, y_train,
                                                                              particle_indices=particle_indices_batches[1])
    if patience_early_stopping > 1:
        key_loop, _ = jax.random.split(key_loop)
        particle_indices_batches = create_particle_minibatch_indices(key_loop,
                                                                     stein_vi_regression_example.state.particles.shape[
                                                                         0],
                                                                     batch_size=5)
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train, y_train,
                                                                                  particle_indices=particle_indices_batches[0])
        stein_vi_regression_example.state = stein_vi_regression_example.update_fn(stein_vi_regression_example.state,
                                                                                  z_train, y_train,
                                                                                  particle_indices=particle_indices_batches[1])

    assert jnp.all(stein_vi_regression_example.state.particles == stein_vi_regression_example_copy.state.particles)


@pytest.mark.parametrize(
    "patience_early_stopping",
    [
        1,
        2,
    ]
)
def test_data_and_particle_minibatch_training_loop(stein_vi_regression_example, get_regression_toy_example,
                                                   patience_early_stopping):
    # given
    key = jax.random.PRNGKey(1)
    regression_toy_example = get_regression_toy_example
    z_train, y_train, z_val, y_val, _, _ = regression_toy_example
    stein_vi_regression_example.parameter.num_particles = 10
    stein_vi_regression_example.parameter.batch_size = 36
    stein_vi_regression_example.parameter.particle_batch_size = 5
    stein_vi_regression_example.parameter.num_iterations = 2
    stein_vi_regression_example.parameter.warm_up_iterations_early_stopping = 0
    stein_vi_regression_example.parameter.early_stopping = True
    stein_vi_regression_example.parameter.patience_early_stopping = patience_early_stopping
    stein_vi_regression_example.parameter.min_delta_early_stopping = jnp.inf
    stein_vi_regression_example._full_evaluation = True
    set_up_svgd(stein_vi_regression_example)
    stein_vi_regression_example_copy = copy.copy(stein_vi_regression_example)

    # when
    data_and_particle_minibatch_training_loop(stein_vi_regression_example_copy,
                                                                                 regression_toy_example, key)

    # then
    key_loop, _ = jax.random.split(key)
    z_train_batched, y_train_batched = create_minibatches(36, z_train, y_train, key_loop)
    particle_indices_batches = create_particle_minibatch_indices(key_loop,
                                                                 stein_vi_regression_example.state.particles.shape[0],
                                                                 batch_size=5)
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
        key_loop, _ = jax.random.split(key_loop)
        z_train_batched, y_train_batched = create_minibatches(36, z_train, y_train, key_loop)
        particle_indices_batches = create_particle_minibatch_indices(key_loop,
                                                                     stein_vi_regression_example.state.particles.shape[
                                                                         0],
                                                                     batch_size=5)
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

    assert jnp.all(stein_vi_regression_example.state.particles == stein_vi_regression_example_copy.state.particles)
