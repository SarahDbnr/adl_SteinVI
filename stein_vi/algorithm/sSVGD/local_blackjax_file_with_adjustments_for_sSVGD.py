import functools
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax

from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree
from stein_vi.algorithm.sSVGD.matrices_for_noise_matrix import compute_stochastic_correction

LEARNING_RATE = 0.01

__all__ = [
    "as_top_level_api",
    "init",
    "build_kernel",
    "rbf_kernel",
    "update_median_heuristic"
]


class sSVGDState(NamedTuple):
    particles: ArrayTree
    kernel_parameters: dict[str, Any]
    opt_state: Any


def init(
    initial_particles: ArrayLikeTree,
    kernel_parameters: dict[str, Any],
    optimizer: optax.GradientTransformation,
) -> sSVGDState:
    """
    Initializes the stochastic Stein Variational Gradient Descent Algorithm.

    Parameters
    ----------
    initial_particles
        Initial set of particles to start the optimization
    kernel_paremeters
        Arguments to the kernel function
    """
    opt_state = optimizer.init(initial_particles)
    return sSVGDState(initial_particles, kernel_parameters, opt_state)


def build_kernel(optimizer: optax.GradientTransformation):
    def kernel(
        state: sSVGDState,
        grad_logdensity_fn: Callable,
        kernel: Callable,
        **grad_params,
    ) -> sSVGDState:
        """
        Performs one step of stochastic Stein Variational Gradient Descent.

        See Algorithm 2 of "A STOCHASTIC STEIN VARIATIONAL NEWTON METHOD" by Alex Leviyev, Joshua Chen, Yifei Wang, Omar Ghattas, and Aaron Zimmerman


        Parameters
        ----------
        state
            sSVGDState object containing information about previous iteration
        grad_logdensity_fn
            gradient, or an estimate, of the target log density function to samples approximately from
        kernel
            positive semi definite kernel
        **grad_params
            additional parameters for `grad_logdensity_fn` function, for instance a minibatch parameter
            on a gradient estimator.

        Returns
        -------
        sSVGDState containing new particles and kernel parameters.
        """
        particles, kernel_params, opt_state = state
        kernel = functools.partial(kernel, **kernel_params)

        def phi_star_summand(particle, particle_):
            gradient = grad_logdensity_fn(particle, **grad_params)
            k, grad_k = jax.value_and_grad(kernel, argnums=0)(particle, particle_)
            return jax.tree_util.tree_map(lambda g, gk: -(k * g) - gk, gradient, grad_k)

        functional_gradient = jax.vmap(
            lambda p_: jax.tree_util.tree_map(
                lambda phi_star: phi_star.mean(axis=0),
                jax.vmap(lambda p: phi_star_summand(p, p_))(particles),
            )
        )(particles)

        particle_array = jax.vmap(lambda p: ravel_pytree(p)[0])(particles)
        rng_key = jax.random.PRNGKey(1)
        rng_key, rng_subkey = jax.random.split(rng_key)
        Nd = particles.shape[0] * particle_array.shape[1]
        random_normal_samples = jax.random.normal(rng_subkey, (Nd,))
        noise = compute_stochastic_correction(particle_array, kernel, kernel_params, random_normal_samples)

        updates, opt_state = optimizer.update(functional_gradient, opt_state, particles)
        updates = updates + jnp.sqrt(opt_state.hyperparams['learning_rate']) * noise
        particles = optax.apply_updates(particles, updates)
            
        # Normal sSVGD without any optax optimizer.
        # particles = jax.tree_util.tree_map(lambda p, u, n: p + (LEARNING_RATE) * u + jnp.sqrt(LEARNING_RATE) * n, particles, functional_gradient, noise)

        return sSVGDState(particles, kernel_params, opt_state)

    return kernel


def rbf_kernel(x, y, length_scale=1):
    arg = ravel_pytree(jax.tree_util.tree_map(lambda x, y: (x - y) ** 2, x, y))[0]
    return jnp.exp(-(1 / length_scale) * arg.sum())


def median_heuristic(kernel_parameters, particles):
    particle_array = jax.vmap(lambda p: ravel_pytree(p)[0])(particles)

    def distance(x, y):
        return jnp.linalg.norm(jnp.atleast_1d(x - y))

    vmapped_distance = jax.vmap(jax.vmap(distance, (None, 0)), (0, None))
    A = vmapped_distance(particle_array, particle_array)
    pairwise_distances = A[
        jnp.tril_indices(A.shape[0], k=-1)
    ]
    median = jnp.median(pairwise_distances)
    kernel_parameters["length_scale"] = (median**2) / jnp.log(particle_array.shape[0])
    return kernel_parameters


def update_median_heuristic(state: sSVGDState) -> sSVGDState:
    """Median heuristic for setting the bandwidth of RBF kernels.

    A reasonable middle-ground for choosing the `length_scale` of the RBF kernel
    is to pick the empirical median of the squared distance between particles.
    This strategy is called the median heuristic.
    """

    position, kernel_parameters,opt_state = state
    return sSVGDState(position, median_heuristic(kernel_parameters, position), opt_state)


def as_top_level_api(
    grad_logdensity_fn: Callable,
    optimizer,
    kernel: Callable = rbf_kernel,
    update_kernel_parameters: Callable = update_median_heuristic,
):
    """Implements the (basic) user interface for the ssvgd algorithm "A STOCHASTIC STEIN VARIATIONAL NEWTON METHOD" by Alex Leviyev, Joshua Chen, Yifei Wang, Omar Ghattas, and Aaron Zimmerman.

    Parameters
    ----------
    grad_logdensity_fn
        gradient, or an estimate, of the target log density function to samples approximately from
    kernel
        positive semi definite kernel
    update_kernel_parameters
        function that updates the kernel parameters given the current state of the particles

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    kernel_ = build_kernel(optimizer)

    def init_fn(
        initial_position: ArrayLikeTree,
        kernel_parameters: dict[str, Any] = {"length_scale": 1.0},
    ):
        return init(initial_position, kernel_parameters, optimizer)

    def step_fn(state, **grad_params):
        state = kernel_(state, grad_logdensity_fn, kernel, **grad_params)
        return update_kernel_parameters(state)

    return SamplingAlgorithm(init_fn, step_fn)