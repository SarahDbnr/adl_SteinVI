import jax
import jax.numpy as jnp
import blackjax
from blackjax.vi.svgd import rbf_kernel, update_median_heuristic
from tqdm import tqdm

from src.metrics.validation_and_evaluation import get_evaluation_metrics_over_predictions
from src.algorithm.get_posteriori import get_posteriori
from src.Parameter_Class import Parameter
from src.algorithm.training_loop import train_general_algorithm


def train_with_svgd(dataset, nnet_model, tree_def, param_vec, parameter, key):
    """
    Trains a neural network using the Stein Variational Gradient Descent (SVGD) algorithm.
    Args are similar to the original function with Parameter objects and key.
    """
    # Initialize particles for SVGD
    rng_key_observed, rng_key_init = jax.random.split(key, 2)
    initial_particles_vector = initialize_particles(param_vec, rng_key_init, parameter.num_particles)
    
    # Define posterior function
    logp_model = get_posteriori(nnet_model, tree_def, parameter.use_for_regression)
    
    # Define the kernel function
    kernel_fn = rbf_kernel
    
    # SVGD-specific update function
    def svgd_update_fn(state, z_batch, y_batch, particle_indices=None):
        return update_svgd(state, logp_model, z_batch, y_batch, particle_indices, kernel_fn, parameter)

    # Run the training loop
    best_state, eval_metrics_1, eval_metrics_2 = train_general_algorithm(
        dataset=dataset,
        nnet_model=nnet_model,
        tree_def=tree_def,
        param_vec=param_vec,
        parameter=parameter,
        key=key,
        initialize_particles_fn=initialize_particles,
        update_fn=svgd_update_fn,
        evaluate_fn=evaluate_model_fn,
        early_stopping_fn=early_stopping_fn
    )

    return best_state, eval_metrics_1, eval_metrics_2


def update_svgd(state, logp_model, z_batch, y_batch, particle_indices, kernel_fn, parameter):
    """
    Performs an update step for SVGD. Supports both particle and data minibatching.
    """
    grad_log_posterior = jax.grad(logp_model)
    svgd = blackjax.svgd(grad_log_posterior, parameter.optimizer, kernel_fn, update_median_heuristic)
    step_fn = jax.jit(svgd.step)
    
    if particle_indices is not None:
        # Minibatch update for particles
        batch_particles = jnp.take(state.particles, particle_indices, axis=0)
        batch_optimizer_state = get_batched_optimizer_state(state.opt_state, particle_indices)
        
        batch_state = state._replace(particles=batch_particles, opt_state=batch_optimizer_state)
        updated_batch_state = step_fn(batch_state, dz=z_batch, dy=y_batch)
        
        new_particles = state.particles.at[particle_indices].set(updated_batch_state.particles)
        new_optimizer_state = update_optimizer_state(state.opt_state, updated_batch_state, particle_indices)
        
        state = state._replace(particles=new_particles, opt_state=new_optimizer_state)
    else:
        # Full update
        state = step_fn(state, dz=z_batch, dy=y_batch)

    return state


def initialize_particles(param_vec, rng_key_init, num_particles):
    """
    Initializes particles for SVGD.
    """
    return jax.random.normal(rng_key_init, shape=(num_particles,) + param_vec.shape)


def evaluate_model_fn(state, nnet_model, tree_def, z_val, y_val, parameter):
    """
    Evaluates the model on validation data, returning the metrics.
    """
    return get_evaluation_metrics_over_predictions(state, nnet_model, tree_def, z_val, y_val, parameter.use_for_regression)


def early_stopping_fn(current_metrics, best_metrics, patience_counter, state, best_state, parameter):
    """
    Implements early stopping based on validation metrics.
    """
    if current_metrics > best_metrics + parameter.min_delta_early_stopping:
        return state, current_metrics, 0
    return best_state, best_metrics, patience_counter + 1


def get_batched_optimizer_state(optimizer_state, indices):
    """
    Extracts the optimizer state for a minibatch of particles.
    """
    def batch_fn(x):
        if hasattr(x, 'ndim') and x.ndim > 0:
            return jnp.take(x, indices, axis=0)
        return x

    return jax.tree_map(batch_fn, optimizer_state)


def update_optimizer_state(optimizer_state, batched_state, indices):
    """
    Updates the global optimizer state with the minibatched state.
    """
    def update_fn(orig, batched):
        if hasattr(orig, 'ndim') and orig.ndim > 0:
            return orig.at[indices].set(batched)
        return orig
    
    return jax.tree_map(update_fn, optimizer_state, batched_state.opt_state)
