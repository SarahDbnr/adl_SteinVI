import jax
import jax.numpy as jnp
import blackjax
from blackjax.vi.svgd import rbf_kernel, update_median_heuristic
from tqdm import tqdm

from src.metrics.validation_and_evaluation import get_evaluation_metrics_over_predictions
from src.algorithm.get_posteriori import get_posteriori
from src.Parameter_Class import Parameter
from src.algorithm.model_training import train_general_algorithm


def train_with_svgd(dataset, nnet_model, tree_def, param_vec, parameter, key):
    """
    Trains a neural network using the Stein Variational Gradient Descent (SVGD) algorithm.

    Args:
        dataset (tuple): A tuple containing the training data (features and labels).
        nnet_model (object): The neural network model used for making predictions.
        tree_def (object): Tree structure used for parameter transformation in JAX.
        param_vec (jax.numpy.ndarray): Initial parameter vector for the neural network.
        parameter (Parameter): A Parameter object containing hyperparameters such as number of particles and optimizer settings.
        key (jax.random.PRNGKey): JAX random key for initializing particles and managing randomness.

    Returns:
        tuple: The state of the model after training, and two evaluation metric values.
    """
    _, rng_key_init = jax.random.split(key, 2)
    initial_particles_vector = initialize_particles(param_vec, rng_key_init, parameter.num_particles)
    
    logp_model = get_posteriori(nnet_model, tree_def, parameter.use_for_regression)
    
    kernel_fn = rbf_kernel
    
    svgd_state, svgd= initialize_svgd_state(logp_model, initial_particles_vector, kernel_fn, parameter)

    def svgd_update_fn(state, z_batch, y_batch, step_fn=jax.jit(svgd.step), particle_indices=None):
        if particle_indices is not None:
            # TODO: why is this necessary: and particle_indices != 0
            return particle_minibatching(state, z_batch, y_batch, step_fn, particle_indices)
        else:
            return step_fn(state, dz=z_batch, dy=y_batch)

    state, eval_metrics_1, eval_metrics_2 = train_general_algorithm(
        dataset=dataset,
        nnet_model=nnet_model,
        tree_def=tree_def,
        parameter=parameter,
        key=key,
        state=svgd_state,
        update_fn=svgd_update_fn,
        evaluate_fn=evaluate_model_fn,
        early_stopping_fn=early_stopping_fn,
    )

    return state, eval_metrics_1, eval_metrics_2

def initialize_svgd_state(logp_model, initial_particles_vector, kernel_fn, parameter):
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
    grad_log_posterior = jax.grad(logp_model)
    svgd = blackjax.svgd(grad_log_posterior, parameter.optimizer, kernel_fn, update_median_heuristic)
    initial_kernel_params = {"length_scale": parameter.kernel_length}
    return svgd.init(initial_particles_vector, initial_kernel_params), svgd

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


def initialize_particles(param_vec, rng_key_init, num_particles):
    """
    Initializes particles for the SVGD algorithm using a normal distribution.

    Args:
        param_vec (jax.numpy.ndarray): The initial parameter vector for the model.
        rng_key_init (jax.random.PRNGKey): JAX random key for initializing particles.
        num_particles (int): The number of particles to initialize.

    Returns:
        jax.numpy.ndarray: A set of initialized particles.
    """
    return jax.random.normal(rng_key_init, shape=(num_particles,) + param_vec.shape)


def evaluate_model_fn(state, nnet_model, tree_def, z_val, y_val, parameter):
    """
    Evaluates the model using the current SVGD state on validation data.

    Args:
        state (object): The current SVGD state containing particles and optimizer state.
        nnet_model (object): The neural network model used for making predictions.
        tree_def (object): Tree structure used for parameter transformation in JAX.
        z_val (jax.numpy.ndarray): Validation input features.
        y_val (jax.numpy.ndarray): True output labels for the validation set.
        parameter (Parameter): A Parameter object that defines if the task is regression or classification.

    Returns:
        tuple: Evaluation metrics such as MSE or accuracy, depending on the task.
    """
    return get_evaluation_metrics_over_predictions(state, nnet_model, tree_def, z_val, y_val, parameter.use_for_regression)


def early_stopping_fn(current_metrics, best_metrics, patience_counter, parameter):
    """
    Implements early stopping by comparing validation metrics over training iterations.

    Args:
        current_metrics (float): The evaluation metric for the current iteration.
        best_metrics (float): The best evaluation metric seen so far.
        patience_counter (int): A counter tracking how many iterations the model has been non-improving.
        parameter (Parameter): A Parameter object defining early stopping criteria like minimum delta.

    Returns:
        tuple: Updated patience counter and best metric value.
    """
    if current_metrics < best_metrics + parameter.min_delta_early_stopping:
        patience_counter = patience_counter + 1
    else:
        patience_counter = 0
        best_metrics = current_metrics
    return patience_counter, best_metrics


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

