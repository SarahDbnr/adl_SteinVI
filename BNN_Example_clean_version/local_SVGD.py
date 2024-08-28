import functools
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayLikeTree, ArrayTree

__all__ = [
    "as_top_level_api",
    "init",
    "build_kernel",
    "rbf_kernel",
    "update_median_heuristic",
]


class SVGDState(NamedTuple):
    particles: ArrayTree
    kernel_parameters: dict[str, ArrayTree]
    opt_state: Any


def init(
    initial_particles: ArrayLikeTree,
    kernel_parameters: dict[str, Any],
    optimizer: optax.GradientTransformation,
) -> SVGDState:
    """
    Initializes Stein Variational Gradient Descent Algorithm.

    Parameters
    ----------
    initial_particles
        Initial set of particles to start the optimization
    kernel_paremeters
        Arguments to the kernel function
    optimizer
        Optax compatible optimizer, which conforms to the `optax.GradientTransformation` protocol
    """
    opt_state = optimizer.init(initial_particles)
    return SVGDState(initial_particles, kernel_parameters, opt_state)


def build_kernel(optimizer: optax.GradientTransformation):
    def kernel(
        state: SVGDState,
        grad_logdensity_fn: Callable,
        kernel: Callable,
        **grad_params,
    ) -> SVGDState:
        """
        Performs one step of Stein Variational Gradient Descent.

        See Algorithm 1 of :cite:p:`liu2016stein`.

        Parameters
        ----------
        state
            SVGDState object containing information about previous iteration
        grad_logdensity_fn
            gradient, or an estimate, of the target log density function to samples approximately from
        kernel
            positive semi definite kernel
        **grad_params
            additional parameters for `grad_logdensity_fn` function, for instance a minibatch parameter
            on a gradient estimator.

        Returns
        -------
        SVGDState containing new particles, optimizer state and kernel parameters.
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

        updates, opt_state = optimizer.update(functional_gradient, opt_state, particles)
        particles = optax.apply_updates(particles, updates)

        return SVGDState(particles, kernel_params, opt_state)

    return kernel


def rbf_kernel(x, y, length_scale=1):
    arg = ravel_pytree(jax.tree_util.tree_map(lambda x, y: (x - y) ** 2, x, y))[0]
    return jnp.exp(-(1 / length_scale) * arg.sum())


def median_heuristic(kernel_parameters, particles):
    particle_array = jax.vmap(lambda p: ravel_pytree(p)[0])(particles)

    def distance(x, y):
        return jnp.linalg.norm(jnp.atleast_1d(x - y))

    vmapped_distance = jax.vmap(jax.vmap(distance, (None, 0)), (0, None))
    A = vmapped_distance(particle_array, particle_array)  # Calculate distance matrix
    pairwise_distances = A[
        jnp.tril_indices(A.shape[0], k=-1)
    ]  # Take values below the main diagonal into a vector
    median = jnp.median(pairwise_distances)
    kernel_parameters["length_scale"] = (median**2) / jnp.log(particle_array.shape[0])
    return kernel_parameters


def update_median_heuristic(state: SVGDState) -> SVGDState:
    """Median heuristic for setting the bandwidth of RBF kernels.

    A reasonable middle-ground for choosing the `length_scale` of the RBF kernel
    is to pick the empirical median of the squared distance between particles.
    This strategy is called the median heuristic.
    """

    position, kernel_parameters, opt_state = state
    return SVGDState(position, median_heuristic(kernel_parameters, position), opt_state)


def as_top_level_api(
    grad_logdensity_fn: Callable,
    optimizer,
    kernel: Callable = rbf_kernel,
    update_kernel_parameters: Callable = update_median_heuristic,
):
    """Implements the (basic) user interface for the svgd algorithm :cite:p:`liu2016stein`.

    Parameters
    ----------
    grad_logdensity_fn
        gradient, or an estimate, of the target log density function to samples approximately from
    optimizer
        Optax compatible optimizer, which conforms to the `optax.GradientTransformation` protocol
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

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
def stochastic_build_kernel(optimizer: optax.GradientTransformation, batch_size: int):
    def kernel(
        state: SVGDState,
        grad_logdensity_fn: Callable,
        kernel: Callable,
        data: ArrayLikeTree,  # Add data input
        **grad_params,
    ) -> SVGDState:
        """
        Performs one step of Stochastic Stein Variational Gradient Descent.
        
        Parameters
        ----------
        state
            SVGDState object containing information about previous iteration
        grad_logdensity_fn
            gradient, or an estimate, of the target log density function to samples approximately from
        kernel
            positive semi-definite kernel
        data
            full dataset
        batch_size
            size of the mini-batch for stochastic gradient estimation
        **grad_params
            additional parameters for `grad_logdensity_fn` function.

        Returns
        -------
        SVGDState containing new particles, optimizer state and kernel parameters.
        """
        particles, kernel_params, opt_state = state
        kernel = functools.partial(kernel, **kernel_params)

        # Sample a mini-batch from the data
        indices = jax.random.choice(jax.random.PRNGKey(0), len(data), shape=(batch_size,), replace=False)
        batch_data = jax.tree_map(lambda x: x[indices], data)

        def phi_star_summand(particle, particle_):
            gradient = grad_logdensity_fn(particle, batch_data, **grad_params)  # Pass batch data
            k, grad_k = jax.value_and_grad(kernel, argnums=0)(particle, particle_)
            return jax.tree_util.tree_map(lambda g, gk: -(k * g) - gk, gradient, grad_k)

        functional_gradient = jax.vmap(
            lambda p_: jax.tree_util.tree_map(
                lambda phi_star: phi_star.mean(axis=0),
                jax.vmap(lambda p: phi_star_summand(p, p_))(particles),
            )
        )(particles)

        updates, opt_state = optimizer.update(functional_gradient, opt_state, particles)
        particles = optax.apply_updates(particles, updates)

        return SVGDState(particles, kernel_params, opt_state)

    return kernel

def stochastic_as_top_level_api(
    grad_logdensity_fn: Callable,
    optimizer,
    kernel: Callable = rbf_kernel,
    update_kernel_parameters: Callable = update_median_heuristic,
    batch_size: int = 32,
):
    """Implements the user interface for the stochastic svgd algorithm.

    Parameters
    ----------
    grad_logdensity_fn
        gradient, or an estimate, of the target log density function to samples approximately from
    optimizer
        Optax compatible optimizer, which conforms to the `optax.GradientTransformation` protocol
    kernel
        positive semi-definite kernel
    update_kernel_parameters
        function that updates the kernel parameters given the current state of the particles
    batch_size
        mini-batch size for stochastic gradient estimation

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    kernel_ = stochastic_build_kernel(optimizer, batch_size)

    def init_fn(
        initial_position: ArrayLikeTree,
        kernel_parameters: dict[str, Any] = {"length_scale": 1.0},
    ):
        return init(initial_position, kernel_parameters, optimizer)

    def step_fn(state, data, **grad_params):  # Pass data here
        state = kernel_(state, grad_logdensity_fn, kernel, data, **grad_params)
        return update_kernel_parameters(state)

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]



import jax.scipy.linalg as linalg

def stochastic_variational_newton_kernel(optimizer: optax.GradientTransformation, batch_size: int):
    def kernel(
        state: SVGDState,
        grad_logdensity_fn: Callable,
        hessian_logdensity_fn: Callable,
        kernel: Callable,
        data: ArrayLikeTree,  # Add data input
        **grad_params,
    ) -> SVGDState:
        """
        Performs one step of Stochastic Variational Newton (SVN).
        
        Parameters
        ----------
        state
            SVGDState object containing information about previous iteration
        grad_logdensity_fn
            Gradient of the target log density function.
        hessian_logdensity_fn
            Hessian (second-order derivative) of the target log density function.
        kernel
            Positive semi-definite kernel.
        data
            Full dataset.
        batch_size
            Size of the mini-batch for stochastic gradient and Hessian estimation.
        **grad_params
            Additional parameters for `grad_logdensity_fn` and `hessian_logdensity_fn` functions.

        Returns
        -------
        SVGDState containing new particles, optimizer state, and kernel parameters.
        """
        particles, kernel_params, opt_state = state
        kernel = functools.partial(kernel, **kernel_params)

        # Sample a mini-batch from the data
        indices = jax.random.choice(jax.random.PRNGKey(0), len(data), shape=(batch_size,), replace=False)
        batch_data = jax.tree_map(lambda x: x[indices], data)

        def phi_star_summand(particle, particle_):
            gradient = grad_logdensity_fn(particle, batch_data, **grad_params)  # Pass batch data
            hessian = hessian_logdensity_fn(particle, batch_data, **grad_params)  # Pass batch data
            hessian_inv = linalg.inv(hessian + jnp.eye(hessian.shape[0]) * 1e-6)  # Adding small regularization

            preconditioned_grad = jax.tree_util.tree_map(
                lambda g: jnp.dot(hessian_inv, g),
                gradient
            )

            k, grad_k = jax.value_and_grad(kernel, argnums=0)(particle, particle_)
            return jax.tree_util.tree_map(lambda g, gk: -(k * g) - gk, preconditioned_grad, grad_k)

        functional_gradient = jax.vmap(
            lambda p_: jax.tree_util.tree_map(
                lambda phi_star: phi_star.mean(axis=0),
                jax.vmap(lambda p: phi_star_summand(p, p_))(particles),
            )
        )(particles)

        updates, opt_state = optimizer.update(functional_gradient, opt_state, particles)
        particles = optax.apply_updates(particles, updates)

        return SVGDState(particles, kernel_params, opt_state)

    return kernel

def stochastic_variational_newton_api(
    grad_logdensity_fn: Callable,
    hessian_logdensity_fn: Callable,
    optimizer,
    kernel: Callable = rbf_kernel,
    update_kernel_parameters: Callable = update_median_heuristic,
    batch_size: int = 32,
):
    """Implements the user interface for the stochastic variational newton (SVN) algorithm.

    Parameters
    ----------
    grad_logdensity_fn
        Gradient of the target log density function to sample approximately from.
    hessian_logdensity_fn
        Hessian (second-order derivative) of the target log density function.
    optimizer
        Optax compatible optimizer, which conforms to the `optax.GradientTransformation` protocol.
    kernel
        Positive semi-definite kernel.
    update_kernel_parameters
        Function that updates the kernel parameters given the current state of the particles.
    batch_size
        Mini-batch size for stochastic gradient and Hessian estimation.

    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    kernel_ = stochastic_variational_newton_kernel(optimizer, batch_size)

    def init_fn(
        initial_position: ArrayLikeTree,
        kernel_parameters: dict[str, Any] = {"length_scale": 1.0},
    ):
        return init(initial_position, kernel_parameters, optimizer)

    def step_fn(state, data, **grad_params):  # Pass data here
        state = kernel_(state, grad_logdensity_fn, hessian_logdensity_fn, kernel, data, **grad_params)
        return update_kernel_parameters(state)

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]