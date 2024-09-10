import jax.numpy as jnp
import jax

from src.algorithm.svgd import create_particle_minibatch_indices

def test_create_particle_minibatch_indices():
    # given
    key = jax.random.PRNGKey(0)
    num_particles = 100
    batch_size = 30
    # when
    result = create_particle_minibatch_indices(key, num_particles, batch_size)
    # then
    all_indices = jnp.concatenate(result)
    assert jnp.array_equal(jnp.sort(all_indices), jnp.arange(num_particles))
    assert all(len(batch) in [34, 33, 33] for batch in result)
