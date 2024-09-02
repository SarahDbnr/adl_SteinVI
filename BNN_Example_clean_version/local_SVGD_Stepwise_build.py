import functools
from typing import Any, Callable, NamedTuple
from jax import grad, jacfwd, random
from jax.scipy.linalg import cho_solve, cho_factor
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




def stochastic_svgd_kernel(optimizer: optax.GradientTransformation ):
    def kernel(
        state: SVGDState,
        tau:float,
        grad_logdensity_fn: Callable,
        kernel_fn: Callable,
        **grad_params,
    ) -> SVGDState:
        particles, kernel_params, opt_state = state
        kernel = functools.partial(kernel_fn, **kernel_params)
        #Computes functional gradient
        # def phi_star_summand(particle, particle_):
        #     gradient = grad_logdensity_fn(particle, **grad_params)
        #     k, grad_k = jax.value_and_grad(kernel, argnums=0)(particle, particle_)
        #     return jax.tree_util.tree_map(lambda g, gk: -(k * g) - gk, gradient, grad_k)

        # functional_gradient = jax.vmap(
        #     lambda p_: jax.tree_util.tree_map(
        #         lambda phi_star: phi_star.mean(axis=0),
        #         jax.vmap(lambda p: phi_star_summand(p, p_))(particles),
        #     )
        # )(particles)

        # # Update particles using deterministic part (SVGD)
        # updates, opt_state = optimizer.update(functional_gradient, opt_state, particles)
        def compute_velocity_field(particle):
            def phi_star_summand(particle_, particle):
                gradient = grad_logdensity_fn(particle_, **grad_params)
                k, grad_k = jax.value_and_grad(kernel, argnums=0)(particle_, particle)
                return jax.tree_util.tree_map(lambda g, gk: k * g + gk, gradient, grad_k)
            
            # Calculate the average of phi_star_summand over all particles
            return jax.tree_util.tree_map(
                lambda phi_star: phi_star.mean(axis=0),
                jax.vmap(lambda p_: phi_star_summand(p_, particle))(particles)
            )
        #Until here same as before
        #particles = optax.apply_updates(particles, updates) we do this later now
        velocity_field = jax.vmap(compute_velocity_field)(particles)
        # Add stochastic correction
        particle_array = jax.vmap(lambda p: ravel_pytree(p)[0])(particles)

        noise = compute_stochastic_correction(particle_array,kernel,kernel_params)  # Ensure noise shape matches particle_array
        #K=kernel_matrix(particles,kernel,kernel_params)
        #noise = jax.random.normal(jax.random.PRNGKey(0), K)
        particles = jax.tree_util.tree_map(lambda p, u, n: p + (tau+1) * u + jnp.sqrt(tau+1) * n, particles, velocity_field, noise)
        #particles = optax.apply_updates(particles, updates)# we do this later now
        return SVGDState(particles, kernel_params, opt_state)

    return kernel


def kernel_matrix(particles, kernel_fn, kernel_params):
    #only needed when not using theory
    n_particles = particles.shape[0]
    d = particles.shape[1]  # Dimension of each particle
    K = jnp.zeros((n_particles * d, n_particles * d))
    for i in range(n_particles):
        for j in range(n_particles):
            # Compute the kernel value k(z_i, z_j)
            k_ij = 2*kernel_fn(particles[i], particles[j], **kernel_params)
            # Create a block for K_ij = k(z_i, z_j) * I_{dxd}
            block = k_ij * jnp.eye(d)
            # Assign the block to the corresponding position in the matrix K
            K = K.at[i*d:(i+1)*d, j*d:(j+1)*d].set(block)
    
    # Scale the matrix by 1/N
    K = K / n_particles
    
    return K

def compute_stochastic_correction(particles, kernel_fn, kernel_params):
    n_particles, d = particles.shape
    Nd = n_particles * d
    
    # Step 1: Calculate the kernel Gram matrix k_bar
    k_bar = jnp.zeros((n_particles, n_particles))
    for i in range(n_particles):
        for j in range(n_particles):
            k_bar = k_bar.at[i, j].set(kernel_fn(particles[i], particles[j], **kernel_params))
    
    # Step 2: Construct the block-diagonal matrix D_K
    D_K = jnp.kron(jnp.eye(d), k_bar / n_particles)
    # Step 3: Compute the Cholesky decomposition of D_K
    L_DK = jnp.linalg.cholesky(D_K)
    
    # Step 4: Generate random normal samples from N(0, I_{Nd x Nd})
    random_normal_samples = jax.random.normal(jax.random.PRNGKey(0), (Nd,))
    
    # Step 5: Create the permutation matrix P
    P = compute_permutation_matrix(n_particles, d)
    #P_test = compute_permutation_matrix_test(n_particles, d)
    #Create test, that checks that the permutation is correct
    # Step 6: Compute the stochastic correction v^STC
    v_stc = jnp.sqrt(2) * jnp.dot(P.T, jnp.dot(L_DK, random_normal_samples))
    return v_stc.reshape(n_particles, d)

def compute_permutation_matrix_test(N, d):
    Nd = N * d
    P = jnp.zeros((Nd, Nd))
    for i in range(d):
        for j in range(N):
            P = P.at[i * N + j, j * d + i].set(1)
    return P

def compute_permutation_matrix(N, d):
    #should return the same result as the test matrix but i am not sure if this is the case
    Nd = N * d
    # Create the identity matrix of size Nd x Nd
    identity = jnp.eye(Nd)
    
    # Reshape and transpose to achieve the desired permutation
    P = identity.reshape(N, d, N, d).transpose(1, 0, 2, 3).reshape(Nd, Nd)
    
    return P
# The as_top_level_api function would remain largely the same, except it would now use stochastic_svgd_kernel instead of build_kernel.
def as_top_level_api_stochastic(
    grad_logdensity_fn: Callable,
    optimizer,
    kernel: Callable = rbf_kernel,
    update_kernel_parameters: Callable = update_median_heuristic,
):
    kernel_ = stochastic_svgd_kernel(optimizer)

    def init_fn(
        initial_position: ArrayLikeTree,
        kernel_parameters: dict[str, Any] = {"length_scale": 1.0},
    ):
        return init(initial_position, kernel_parameters, optimizer)

    def step_fn(state,tau, **grad_params):
        state = kernel_(state,tau, grad_logdensity_fn, kernel, **grad_params)
        return update_kernel_parameters(state)

    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
















































# #Calculate Damped Hassian
# def compute_hmn(grad_logdensity_fn, kernel, zp, zm, zn):
#     """Compute the Hmn matrix component for SVN direction."""
#     # Gradient and Hessian of log density
#     hessian_logdensity = jacfwd(grad(grad_logdensity_fn))(zp)
    
#     # Kernel gradients
#     grad_k_zp_zn = jax.grad(lambda x: kernel(x, zn))(zp)
#     grad_k_zp_zm = jax.grad(lambda x: kernel(x, zm))(zp)
    
#     # Hmn calculation
#     hmn = - kernel(zp, zm) * kernel(zp, zn) * hessian_logdensity \
#           + jnp.outer(grad_k_zp_zn, grad_k_zp_zm)
    
#     return hmn
