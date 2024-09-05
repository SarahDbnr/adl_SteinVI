import jax.numpy as jnp
import jax

from BNN_Example_clean_version.svgd import create_particle_minibatch_indices, update_optimizer_state, update_optimizer_iteration, get_batched_optimizer_state, initialize_particles
from BNN_Example_clean_version.BNN_Model import build_model
import optax

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


 def test_update_optimizer_state():
    # Set up test data
    key = jax.random.PRNGKey(0)
    z_train = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    output_size = 1
    hidden_layers = [32, 32]
    regression = True

    nnet_model, tree_def, param_vec = build_model(key, z_train, output_size=output_size,
                                                  hidden_layers=hidden_layers,
                                                  use_for_regression=regression)

    num_particles = 3
    initial_particles_vector = initialize_particles(param_vec, key, num_particles)

    optimizer = optax.adam(0.001)
    opt_state = optimizer.init(initial_particles_vector)

    # Create a batched state
    indices = jnp.array([1])
    batched_particles = jnp.take(initial_particles_vector, indices, axis=0)
    batched_opt_state = get_batched_optimizer_state(opt_state, indices)

    # Call the function
    updated_opt_state = update_optimizer_state(opt_state, batched_opt_state, indices)

    # Assertions
    assert jax.tree_map(lambda x, y: jnp.allclose(x, y), opt_state, updated_opt_state)
    assert jnp.allclose(jax.tree_util.tree_leaves(updated_opt_state)[0][indices], 
                        jax.tree_util.tree_leaves(batched_opt_state)[0])

def test_update_optimizer_iteration():
    # Set up test data
    key = jax.random.PRNGKey(0)
    z_train = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    output_size = 1
    hidden_layers = [32, 32]
    regression = True

    nnet_model, tree_def, param_vec = build_model(key, z_train, output_size=output_size,
                                                  hidden_layers=hidden_layers,
                                                  use_for_regression=regression)

    num_particles = 3
    initial_particles_vector = initialize_particles(param_vec, key, num_particles)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(initial_particles_vector)

    class State:
        def __init__(self, particles, opt_state):
            self.particles = particles
            self.opt_state = opt_state
        
        def _replace(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            return self

    state = State(initial_particles_vector, opt_state)

    # Call the function
    updated_state = update_optimizer_iteration(state)

    # Assertions
    assert jnp.all(updated_state.opt_state.count == state.opt_state.count + 1)
    assert jnp.allclose(updated_state.particles, state.particles)

def test_get_batched_optimizer_state():
    # Set up test data
    key = jax.random.PRNGKey(0)
    z_train = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    output_size = 1
    hidden_layers = [32, 32]
    regression = True

    nnet_model, tree_def, param_vec = build_model(key, z_train, output_size=output_size,
                                                  hidden_layers=hidden_layers,
                                                  use_for_regression=regression)

    num_particles = 3
    initial_particles_vector = initialize_particles(param_vec, key, num_particles)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(initial_particles_vector)

    # Call the function
    indices = jnp.array([0, 2])
    batched_opt_state = get_batched_optimizer_state(opt_state, indices)

    # Assertions
    assert jax.tree_map(lambda x, y: x.shape[0] == len(indices) if x.ndim > 0 else True, 
                    batched_opt_state, opt_state)
    assert jnp.allclose(jax.tree_util.tree_leaves(batched_opt_state)[0], 
                        jnp.take(jax.tree_util.tree_leaves(opt_state)[0], indices, axis=0))   