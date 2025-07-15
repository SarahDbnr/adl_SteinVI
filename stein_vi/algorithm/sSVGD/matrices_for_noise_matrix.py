import jax.numpy as jnp
import jax


def compute_stochastic_correction(particles, kernel_fn, kernel_params, random_normal_samples):
    """Efficiently computes the stochastic correction term used in sSVGD.

    Args:
        particles (ndarray): The particle set, shape (n_particles, d), representing the current state.
        kernel_fn (Callable): The kernel function to compute pairwise distances between particles.
        kernel_params (dict): Parameters for the kernel function (e.g., length scale).
        random_normal_samples (ndarray): Random normal samples used to introduce stochasticity, shape (Nd,).

    Returns:
        ndarray: The stochastic correction term `v_stc` used to adjust particle updates, shape (n_particles, d).
    """

    n_particles, d = particles.shape

    def compute_k_bar(particles):
        pairwise_kernels = jax.vmap(
            lambda x: jax.vmap(lambda y: kernel_fn(x, y, **kernel_params))(particles)
        )(particles)
        return pairwise_kernels / n_particles

    k_bar = compute_k_bar(particles)  
    L_k_bar = jnp.linalg.cholesky(k_bar)  
    epsilon = random_normal_samples.reshape(d, n_particles)  
    V = L_k_bar @ epsilon.T 
    v_stc = jnp.sqrt(2) * V  

    return v_stc
