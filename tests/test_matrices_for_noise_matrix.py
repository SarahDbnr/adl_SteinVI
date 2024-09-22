import jax
import jax.numpy as jnp
from stein_vi.algorithm.sSVGD.matrices_for_noise_matrix import compute_stochastic_correction, compute_stochastic_correction_old
from stein_vi.algorithm.sSVGD.local_blackjax_file_with_adjustments_for_sSVGD import rbf_kernel


def test_compute_stochastic_correction():
    n_particles = 5
    d = 5
    particles = jax.random.normal(jax.random.PRNGKey(42), (n_particles, d))

    kernel_params = {'length_scale': 1.0}

    # Generate random normal samples
    Nd = n_particles * d
    rng_key = jax.random.PRNGKey(0)
    random_normal_samples = jax.random.normal(rng_key, (Nd,))

    # Compute the stochastic correction using the old function
    v_stc_old = compute_stochastic_correction_old(particles, rbf_kernel, kernel_params, random_normal_samples)

    # Compute the stochastic correction using the new function
    v_stc_new = compute_stochastic_correction(particles, rbf_kernel, kernel_params, random_normal_samples)

    # Compare the outputs
    print("v_stc_old:\n", v_stc_old)
    print("v_stc_new:\n", v_stc_new)

    # Compute the difference
    diff = jnp.linalg.norm(v_stc_old - v_stc_new)
    print("Difference between old and new v_stc:", diff)

    # Check if they are close
    assert jnp.allclose(v_stc_old, v_stc_new, atol=1e-6), "The outputs are not the same!"
    print("ha")

test_compute_stochastic_correction()


def compute_stochastic_correction_old(particles, kernel_fn, kernel_params, random_normal_samples):
    """Computes the stochastic correction term used in sSVGD. Based on the paper "A STOCHASTIC STEIN VARIATIONAL NEWTON METHOD" by Alex Leviyev, Joshua Chen, Yifei Wang, Omar Ghattas, and Aaron Zimmerman.

    Args:
        particles (ndarray): The particle set, shape (n_particles, d), representing the current state.
        kernel_fn (Callable): The kernel function to compute pairwise distances between particles.
        kernel_params (dict): Parameters for the kernel function (e.g., length scale).
        random_normal_samples (ndarray): Random normal samples used to introduce stochasticity, shape (Nd,).

    Returns:
        ndarray: The stochastic correction term `v_stc` used to adjust particle updates, shape (n_particles, d).
    """
    
    n_particles, d = particles.shape

    k_bar = jnp.zeros((n_particles, n_particles))
    for i in range(n_particles):
        for j in range(n_particles):
            k_bar = k_bar.at[i, j].set(kernel_fn(particles[i], particles[j], **kernel_params))
    
    D_K = jnp.kron(jnp.eye(d), k_bar / n_particles)
    L_DK = jnp.linalg.cholesky(D_K)
    P = compute_permutation_matrix(n_particles, d)
    v_stc = jnp.sqrt(2) * jnp.dot(P.T, jnp.dot(L_DK, random_normal_samples))

    return v_stc.reshape(n_particles, d)


def compute_permutation_matrix_test(N, d):
    """
    Computes the permutation matrix to reshape elements between particle dimensions for testing purposes.

    Args:
        N (int): The number of particles.
        d (int): The dimension of each particle.

    Returns:
        ndarray: The permutation matrix of shape (Nd, Nd), where Nd = N * d.
    """

    Nd = N * d
    P = jnp.zeros((Nd, Nd))
    for i in range(d):
        for j in range(N):
            P = P.at[i * N + j, j * d + i].set(1)
    return P

def compute_permutation_matrix(N, d):
    """
    Efficiently computes the permutation matrix to reshape elements between particle dimensions.

    Args:
        N (int): The number of particles.
        d (int): The dimension of each particle.

    Returns:
        ndarray: The permutation matrix of shape (Nd, Nd), where Nd = N * d.
    """

    Nd = N * d
    identity = jnp.eye(Nd)
    P = identity.reshape(N, d, N, d).transpose(1, 0, 2, 3).reshape(Nd, Nd)
    
    return P