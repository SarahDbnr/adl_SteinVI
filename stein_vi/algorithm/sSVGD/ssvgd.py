import jax
import jax.numpy as jnp

import stein_vi.algorithm.sSVGD.local_blackjax_file_with_adjustments_for_sSVGD
from stein_vi.algorithm.sSVGD.local_blackjax_file_with_adjustments_for_sSVGD import rbf_kernel, update_median_heuristic
from stein_vi.metrics.validation_and_evaluation import get_evaluation_metrics_over_predictions


def set_up_ssvgd(steinvi_ssvdg):
    """Sets up the stochastic Stein Variational Gradient Descent (sSVGD) process for training a Bayesian Neural Network (BNN).

    This function initializes the state for SVGD, including defining the update function for sSVGD and the evaluation function
    for assessing the model's performance.
    This function performs sSVGD. Here we often get the same results as in normal SVGD this is because the particles weights are very big values e.g. between -40000 and 40000
    and the the variance of the noise is based of the Kernel which only can take on values between 0 and 1 so the noise is very small. By adding a constant factor to the noise in the local blackjax file e.g. noise * 1000000, instead of jnp.sqrt(learning_rate), one can get different results to SVGD.

    Args:
        steinvi_ssvdg (SteinVI_BNN): An instance of the `SteinVI_BNN` class, which contains the Bayesian Neural Network, 
        training parameters, particles, and the log posterior function.

    Returns:
        SteinVI_BNN: The updated `SteinVI_BNN` instance with initialized SVGD state, update function, and evaluation function.
    """

    steinvi_ssvdg.state, svgd = initialize_ssvgd_state(steinvi_ssvdg)

    def ssvgd_update_fn(state, z_batch, y_batch, step_fn=jax.jit(svgd.step), particle_indices=None):
        if particle_indices is not None:
            return particle_minibatching(state, z_batch, y_batch, step_fn, particle_indices)
        else:
            return step_fn(state, dz=z_batch, dy=y_batch)

    steinvi_ssvdg.update_fn = ssvgd_update_fn

    def evaluate_model_fn(state, z_val, y_val, print_out):
        return get_evaluation_metrics_over_predictions(state, steinvi_ssvdg.nnet, z_val, y_val,
                                                       steinvi_ssvdg.use_for_regression, print_out)

    steinvi_ssvdg.evaluate_fn = evaluate_model_fn


def initialize_ssvgd_state(steinvi_ssvdg):
    """
    Initializes the state for sSVGD including the log posterior function, particles, and kernel function.

    Args:
        logp_model (callable): A function representing the log posterior of the model given the data.
        initial_particles_vector (jax.numpy.ndarray): Initial particle vectors for sSVGD.
        kernel_fn (callable): The kernel function used in sSVGD to measure distances between particles.
        parameter (Parameter): A Parameter object containing the sSVGD hyperparameters like kernel length and scale.

    Returns:
        tuple: The initialized sSVGD state and the JIT-compiled update function for sSVGD.
    """
    
    grad_log_posterior = jax.grad(steinvi_ssvdg.log_posteriori)
    ssvgd = stein_vi.algorithm.sSVGD.local_blackjax_file_with_adjustments_for_sSVGD.as_top_level_api(grad_log_posterior, steinvi_ssvdg.parameter.optimizer, rbf_kernel, update_median_heuristic)
    initial_kernel_params = {"length_scale": steinvi_ssvdg.parameter.kernel_length}
    return ssvgd.init(steinvi_ssvdg.initial_particle_vector, initial_kernel_params), ssvgd


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
        object: The updated sSVGD state after the current step.
    """

    batch_particles = jnp.take(state.particles, particle_indices, axis=0)
    batch_optimizer_state = get_batched_optimizer_state(state.opt_state, particle_indices)

    batch_state = state._replace(particles=batch_particles, opt_state=batch_optimizer_state)
    updated_batch_state = step_fn(batch_state, dz=z_batch, dy=y_batch)

    new_particles = state.particles.at[particle_indices].set(updated_batch_state.particles)
    new_optimizer_state = update_optimizer_state(state.opt_state, updated_batch_state, particle_indices)

    return state._replace(particles=new_particles, opt_state=new_optimizer_state)

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

    return jax.tree.map(update_fn, optimizer_state, batched_state.opt_state)
