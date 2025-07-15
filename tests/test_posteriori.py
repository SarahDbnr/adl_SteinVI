import jax.numpy as jnp
from jax.scipy.stats import norm

from fixtures import stein_vi_regression_example, stein_vi_multiclass_example, get_regression_toy_example, get_MNIST

from stein_vi.algorithm.get_posteriori import (logp_unnormalized_posterior_regression, link_function, get_scale,
                                               logp_unnormalized_posterior_multinomial)


def test_logp_unnormalized_posterior_regression(stein_vi_regression_example, get_regression_toy_example):
    # given
    z_train, y_train, _, _, _, _ = get_regression_toy_example

    params = stein_vi_regression_example.initial_particle_vector[0]
    nnet_model = stein_vi_regression_example.nnet
    # when
    likelihood = logp_unnormalized_posterior_regression(params, z_train, y_train, nnet_model)
    # then
    assert likelihood.dtype in (jnp.float32, jnp.float64)
    assert likelihood.shape == ()


def test_logp_unnormalized_posterior_multiclass(stein_vi_multiclass_example, get_MNIST):
    # given
    z_train, y_train, _, _, _, _ = get_MNIST

    params = stein_vi_multiclass_example.initial_particle_vector[0]
    nnet_model = stein_vi_multiclass_example.nnet
    # when
    likelihood = logp_unnormalized_posterior_multinomial(params, z_train, y_train, nnet_model)
    # then
    assert likelihood.dtype in (jnp.float32, jnp.float64)
    assert likelihood.shape == ()


def test_log_prior_likelihood():
    params = jnp.array([0.0])
    log_prior_likelihood = jnp.sum(norm.logpdf(params, loc=0, scale=1))
    assert jnp.isclose(log_prior_likelihood, -0.9189385)


def test_get_scale():
    # given
    precision = jnp.array([[1], [2], [3.0]])
    precision_already_squeezed = jnp.array([1, 2, 3.0])
    expected_output = jnp.array([link_function(1), link_function(2), link_function(3.0)])

    # when
    scale = get_scale(precision)
    scale_as = get_scale(precision_already_squeezed)

    # then
    assert all(scale == scale_as)
    assert all(scale == expected_output)


def test_link_function():
    # given
    x = 0
    # when
    scale = link_function(x)
    # then
    assert scale == 0
