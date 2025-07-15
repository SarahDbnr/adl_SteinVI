import pytest
import jax.numpy as jnp

from fixtures import stein_vi_regression_example, get_regression_toy_example
from stein_vi.metrics.validation_and_evaluation import get_evaluation_metrics_over_predictions
from stein_vi.algorithm.svgd import (set_up_svgd, particle_minibatching, initialize_svgd_state,
                                     get_batched_optimizer_state, update_optimizer_state)


@pytest.mark.parametrize(
    "do_particle_minibatching",
    [
        False,
        True,
    ]
)
def test_set_up_svgd(stein_vi_regression_example, get_regression_toy_example, do_particle_minibatching):
    # given
    z_train, y_train, z_val, y_val, _, _ = get_regression_toy_example
    if do_particle_minibatching:
        particle_indices = jnp.array([9, 4])
    else:
        particle_indices = None

    # when
    set_up_svgd(stein_vi_regression_example)

    # then
    old_state = stein_vi_regression_example.state
    _, svgd = initialize_svgd_state(stein_vi_regression_example)
    updated_state_stein_vi = stein_vi_regression_example.update_fn(old_state, z_train, y_train,
                                                                   particle_indices=particle_indices)
    assert jnp.all(old_state.particles == stein_vi_regression_example.state.particles)
    if do_particle_minibatching:
        expected_update = particle_minibatching(old_state, z_train, y_train, svgd.step,
                                                particle_indices=particle_indices)
        assert jnp.allclose(updated_state_stein_vi.particles, expected_update.particles)
    else:
        expected_update = svgd.step(old_state, dz=z_train, dy=y_train)
        assert jnp.allclose(updated_state_stein_vi.particles, expected_update.particles)
    mse, averaged_var, predictions = stein_vi_regression_example.evaluate_fn(stein_vi_regression_example.state, z_val,
                                                                             y_val,
                                                                             print_out=True)
    expected_mse, expected_averaged_var, expected_predictions = get_evaluation_metrics_over_predictions(
        stein_vi_regression_example.state,
        stein_vi_regression_example.nnet,
        z_val, y_val, model_regression=True, print_eva=True)
    assert jnp.allclose(jnp.array(mse), jnp.array(expected_mse))
    assert jnp.allclose(jnp.array(averaged_var), jnp.array(expected_averaged_var))
    assert jnp.allclose(jnp.array(predictions), jnp.array(expected_predictions))


def test_particle_minibatching(stein_vi_regression_example, get_regression_toy_example):
    # given
    z_train, y_train, z_val, y_val, _, _ = get_regression_toy_example
    state, svgd = initialize_svgd_state(stein_vi_regression_example)
    step_fn = svgd.step
    particle_indices = jnp.array([9, 4])
    # when
    updated_state = particle_minibatching(state, z_train, y_train, step_fn, particle_indices)
    # then
    updated_particles_within_batch = updated_state.particles[[9, 4], :]
    old_particles_within_batch = state.particles[[9, 4], :]
    assert jnp.all(updated_particles_within_batch != old_particles_within_batch)
    updated_particles_out_of_batch = updated_state.particles[[0, 1, 2, 3, 5, 6, 7, 8], :]
    old_particles_out_of_batch = state.particles[[0, 1, 2, 3, 5, 6, 7, 8], :]
    assert jnp.allclose(updated_particles_out_of_batch, old_particles_out_of_batch)


def test_get_batched_optimizer_state(stein_vi_regression_example):
    # given
    set_up_svgd(stein_vi_regression_example)
    optimizer_state = stein_vi_regression_example.state.opt_state
    indices = jnp.array([1])
    # when
    batched_optimizer_state = get_batched_optimizer_state(optimizer_state, indices)
    # then
    assert optimizer_state[0].mu.shape == (10, 252)
    assert optimizer_state[0].nu.shape == (10, 252)
    assert batched_optimizer_state[0].mu.shape == (1,252)
    assert batched_optimizer_state[0].nu.shape == (1, 252)


def test_update_optimizer_state(stein_vi_regression_example, get_regression_toy_example):
    # given
    z_train, y_train, _, _, _, _ = get_regression_toy_example

    indices = jnp.array([4, 9])
    state, svgd = initialize_svgd_state(stein_vi_regression_example)
    batch_particles = jnp.take(state.particles, indices, axis=0)
    batch_optimizer_state = get_batched_optimizer_state(state.opt_state, indices)
    batch_state = state._replace(particles=batch_particles, opt_state=batch_optimizer_state)
    optimizer_state = state.opt_state
    updated_batch_state = svgd.step(batch_state, dz=z_train, dy=y_train)
    # when
    updated_optimizer = update_optimizer_state(optimizer_state, updated_batch_state, indices)
    # then
    assert jnp.all(updated_optimizer[0].mu[indices] == updated_batch_state.opt_state[0].mu)
