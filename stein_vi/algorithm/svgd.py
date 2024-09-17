import jax
import jax.numpy as jnp
import blackjax
from blackjax.vi.svgd import rbf_kernel, update_median_heuristic

from stein_vi.metrics.validation_and_evaluation import get_evaluation_metrics_over_predictions


def set_up_svgd(steinvi_svdg):

    steinvi_svdg.state, svgd = initialize_svgd_state(steinvi_svdg)

    def svgd_update_fn(state, z_batch, y_batch, step_fn=jax.jit(svgd.step), particle_indices=None):
        if particle_indices is not None:
            # TODO: why is this necessary: and particle_indices != 0
            return particle_minibatching(state, z_batch, y_batch, step_fn, particle_indices)
        else:
            return step_fn(state, dz=z_batch, dy=y_batch)

    steinvi_svdg.update_fn = svgd_update_fn

    def evaluate_model_fn(state, z_val, y_val, print):
        return get_evaluation_metrics_over_predictions(state, steinvi_svdg.nnet, steinvi_svdg.tree_def, z_val, y_val,
                                                       steinvi_svdg.use_for_regression, print)

    steinvi_svdg.evaluate_fn = evaluate_model_fn

    return steinvi_svdg


def initialize_svgd_state(svi):
    """
    Initializes the state for SVGD including the log posterior function, particles, and kernel function.

    Args:
        logp_model (callable): A function representing the log posterior of the model given the data.
        initial_particles_vector (jax.numpy.ndarray): Initial particle vectors for SVGD.
        kernel_fn (callable): The kernel function used in SVGD to measure distances between particles.
        parameter (Parameter): A Parameter object containing the SVGD hyperparameters like kernel length scale and optimizer.

    Returns:
        tuple: The initialized SVGD state and the JIT-compiled update function for SVGD.
    """
    grad_log_posterior = jax.grad(svi.log_posteriori)
    svgd = blackjax.svgd(grad_log_posterior, svi.parameter.optimizer, rbf_kernel, update_median_heuristic)
    initial_kernel_params = {"length_scale": svi.parameter.kernel_length}
    return svgd.init(svi.initial_particle_vector, initial_kernel_params), svgd


def particle_minibatching(state, z_batch, y_batch, step_fn, particle_indices):
    """
    Updates the SVGD particles and optimizer state, with support for minibatching.

    Args:
        state (object): The current SVGD state containing particles and optimizer state.
        z_batch (jax.numpy.ndarray): The current batch of input data for training.
        y_batch (jax.numpy.ndarray): The corresponding true output labels for the current batch.
        step_fn (callable): The update function that performs the SVGD step.
        particle_indices (jax.numpy.ndarray or None): Indices of the particles to be updated (if minibatching particles).

    Returns:
        object: The updated SVGD state after the current step.
    """

    # Minibatch update for particles
    batch_particles = jnp.take(state.particles, particle_indices, axis=0)
    batch_optimizer_state = get_batched_optimizer_state(state.opt_state, particle_indices)

    batch_state = state._replace(particles=batch_particles, opt_state=batch_optimizer_state)
    updated_batch_state = step_fn(batch_state, dz=z_batch, dy=y_batch)

    new_particles = state.particles.at[particle_indices].set(updated_batch_state.particles)
    new_optimizer_state = update_optimizer_state(state.opt_state, updated_batch_state, particle_indices)

    state = state._replace(particles=new_particles, opt_state=new_optimizer_state)

    return state


def get_batched_optimizer_state(optimizer_state, indices):
    """
    Extracts a minibatch of the optimizer state for the selected particles.

    Args:
        optimizer_state (object): The full optimizer state across all particles.
        indices (jax.numpy.ndarray): Indices of the particles to extract optimizer states for.

    Returns:
        object: The optimizer state for the selected particles.
    """

    def batch_fn(x):
        if hasattr(x, 'ndim') and x.ndim > 0:
            return jnp.take(x, indices, axis=0)
        return x

    return jax.tree.map(batch_fn, optimizer_state)


def update_optimizer_state(optimizer_state, batched_state, indices):
    """
    Updates the global optimizer state with the minibatched state after an update step.

    Args:
        optimizer_state (object): The full optimizer state across all particles.
        batched_state (object): The optimizer state after a minibatch update.
        indices (jax.numpy.ndarray): Indices of the particles that were updated.

    Returns:
        object: The updated optimizer state.
    """

    def update_fn(orig, batched):
        if hasattr(orig, 'ndim') and orig.ndim > 0:
            return orig.at[indices].set(batched)
        return orig

    return jax.tree_map(update_fn, optimizer_state, batched_state.opt_state)