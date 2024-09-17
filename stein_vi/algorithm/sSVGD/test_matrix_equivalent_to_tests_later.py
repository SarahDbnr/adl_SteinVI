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



#Todo: write test for permutation matrices