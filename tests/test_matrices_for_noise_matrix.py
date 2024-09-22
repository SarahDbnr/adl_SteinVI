import jax
import jax.numpy as jnp
from stein_vi.algorithm.sSVGD.matrices_for_noise_matrix import compute_stochastic_correction
from stein_vi.algorithm.sSVGD.local_blackjax_file_with_adjustments_for_sSVGD import rbf_kernel


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
def compute_stochastic_correction_old(particles, kernel_fn, kernel_params, random_normal_samples):
    """Computes the stochastic correction term used in sSVGD. Based on the paper "A STOCHASTIC STEIN VARIATIONAL NEWTON METHOD" by Alex Leviyev, Joshua Chen, Yifei Wang, Omar Ghattas, and Aaron Zimmerman.
    Without clever tricks, more like in the paper.

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

def test_compute_stochastic_correction():
    """This test compares the old and new implementations of the stochastic correction term used in sSVGD, since compute_stochastic_correction_old is more like the implementation in the paper.
    """
    n_particles = 5
    d = 5
    particles = jax.random.normal(jax.random.PRNGKey(42), (n_particles, d))

    kernel_params = {'length_scale': 1.0}

    Nd = n_particles * d
    rng_key = jax.random.PRNGKey(0)
    random_normal_samples = jax.random.normal(rng_key, (Nd,))

    v_stc_old = compute_stochastic_correction_old(particles, rbf_kernel, kernel_params, random_normal_samples)

    v_stc_new = compute_stochastic_correction(particles, rbf_kernel, kernel_params, random_normal_samples)


    diff = jnp.linalg.norm(v_stc_old - v_stc_new)

    assert jnp.allclose(v_stc_old, v_stc_new, atol=1e-6), "The outputs are not the same!"
