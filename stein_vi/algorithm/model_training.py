import jax
import jax.numpy as jnp
from tqdm import tqdm

DEFAULT_NUM_BATCHES = 10


def train_general_algorithm(steinvi, dataset, key):
    """
    General training function for neural networks that supports different mini-batching modes for both data and particles.

    Args:
        dataset (tuple): A tuple containing training and validation datasets (z_train, y_train, z_val, y_val, etc.).
        key (jax.random.PRNGKey): JAX random key for managing randomness.

    Returns:
        tuple: Best model state and two lists of evaluation metrics (e.g., accuracy or MSE) during training.
    """

    if steinvi.parameter.batch_size == 0 and steinvi.parameter.particle_batch_size == 0:
        training_loop_fn = no_minibatch_training_loop
    elif steinvi.parameter.batch_size != 0 and steinvi.parameter.particle_batch_size == 0:
        training_loop_fn = data_minibatch_training_loop
    elif steinvi.parameter.batch_size == 0 and steinvi.parameter.particle_batch_size != 0:
        training_loop_fn = particle_minibatch_training_loop
    else:
        training_loop_fn = data_and_particle_minibatch_training_loop

    steinvi = training_loop_fn(steinvi, dataset, key)

    return steinvi


def no_minibatch_training_loop(steinvi, dataset, key):
    """
    Training loop without mini-batching, using full data and particles.

    Args:
        steinvi : ...
        dataset (tuple): The dataset containing training and validation data.
        key (jax.random.PRNGKey): JAX random key for randomness.

    Returns:
        tuple: Updated model state and two lists of evaluation metrics (e.g., accuracy or MSE).
    """
    z_train, y_train, z_val, y_val, _, _ = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0

    for iteration in tqdm(range(steinvi.parameter.num_iterations), desc="Training"):

        steinvi.state = steinvi.update_fn(steinvi.state, z_train, y_train)  # Full data and full particles

        if steinvi.handler._full_evaluation:
            steinvi, best_eval_metric, patience_counter = get_evaluation_and_apply_early_stopping_logic(
                steinvi, z_val, y_val, iteration, best_eval_metric, patience_counter)

        if patience_counter >= steinvi.parameter.patience_early_stopping:
            break
    return steinvi


def data_minibatch_training_loop(steinvi, dataset, key):
    """
    Training loop with mini-batching on the data while using the full particle set.

    Args:
        steinvi : ...
        dataset (tuple): The dataset containing training and validation data.
        key (jax.random.PRNGKey): JAX random key for randomness.
        early_stopping_fn (callable): Function to apply early stopping criteria.

    Returns:
        tuple: Updated model state and two lists of evaluation metrics.
    """
    z_train, y_train, z_val, y_val, _, _ = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0
    key_loop = key
    for iteration in tqdm(range(steinvi.parameter.num_iterations), desc="Training"):
        key_loop, _ = jax.random.split(key_loop)
        z_train_batched, y_train_batched = create_minibatches(steinvi.parameter.batch_size, z_train, y_train, key_loop)

        for z_batch, y_batch in zip(z_train_batched, y_train_batched):
            steinvi.state = steinvi.update_fn(steinvi.state, z_batch, y_batch)

        if steinvi.handler._full_evaluation:
            steinvi, best_eval_metric, patience_counter = get_evaluation_and_apply_early_stopping_logic(
                steinvi, z_val, y_val, iteration, best_eval_metric, patience_counter)

        if patience_counter >= steinvi.parameter.patience_early_stopping:
            break

    return steinvi


# Particle minibatching: Use the full dataset but split particles into minibatches
def particle_minibatch_training_loop(steinvi, dataset, key):
    """
    Training loop with mini-batching on the particles while using the full dataset.

    Args:
        steinvi : ...
        dataset (tuple): The dataset containing training and validation data.
        key (jax.random.PRNGKey): JAX random key for randomness.
        early_stopping_fn (callable): Function to apply early stopping criteria.

    Returns:
        tuple: Updated model state and two lists of evaluation metrics.
    """
    z_train, y_train, z_val, y_val, _, _ = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0
    key_loop = key

    for iteration in tqdm(range(steinvi.parameter.num_iterations), desc="Training"):
        key_loop, _ = jax.random.split(key_loop)
        particle_indices_batches = create_particle_minibatch_indices(key_loop, steinvi.state.particles.shape[0],
                                                                     steinvi.parameter.particle_batch_size)

        for particle_indices in particle_indices_batches:
            steinvi.state = steinvi.update_fn(steinvi.state, z_train, y_train, particle_indices=particle_indices)

        if steinvi.handler._full_evaluation:
            steinvi, best_eval_metric, patience_counter = get_evaluation_and_apply_early_stopping_logic(
                steinvi, z_val, y_val, iteration, best_eval_metric, patience_counter)

        if patience_counter >= steinvi.parameter.patience_early_stopping:
            break

    return steinvi


# Both data and particle minibatching
def data_and_particle_minibatch_training_loop(steinvi, dataset, key):
    """
    Training loop with mini-batching on both data and particles.

    Args:
        steinvi : ...
        dataset (tuple): The dataset containing training and validation data.
        key (jax.random.PRNGKey): JAX random key for randomness.
        early_stopping_fn (callable): Function to apply early stopping criteria.

    Returns:
        tuple: Updated model state and two lists of evaluation metrics.
    """
    z_train, y_train, z_val, y_val, _, _ = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0
    key_loop = key

    for iteration in tqdm(range(steinvi.parameter.num_iterations), desc="Training"):
        key_loop, _ = jax.random.split(key_loop)
        z_train_batched, y_train_batched = create_minibatches(steinvi.parameter.batch_size, z_train, y_train, key_loop)
        particle_indices_batches = create_particle_minibatch_indices(key_loop, steinvi.state.particles.shape[0],
                                                                     steinvi.parameter.particle_batch_size)

        for z_batch, y_batch in zip(z_train_batched, y_train_batched):
            for particle_indices in particle_indices_batches:
                steinvi.state = steinvi.update_fn(steinvi.state, z_batch, y_batch, particle_indices=particle_indices)

        if steinvi.handler._full_evaluation:
            steinvi, best_eval_metric, patience_counter = get_evaluation_and_apply_early_stopping_logic(
                steinvi, z_val, y_val, iteration, best_eval_metric, patience_counter)

        if patience_counter >= steinvi.parameter.patience_early_stopping:
            break

    return steinvi


# Utility function to create minibatches

def create_minibatches(batch_size, input_data, output_data, key):
    """
    Creates minibatches of data for training.

    Args:
        batch_size (int): The size of each minibatch.
        input_data (jax.numpy.ndarray): Input features of the dataset.
        output_data (jax.numpy.ndarray): Output labels or target values of the dataset.
        key (jax.random.PRNGKey): JAX random key for shuffling the data.

    Returns:
        tuple: Minibatched input and output data.
    """
    # TODO: this is unused, can we delete it
    if batch_size != 0:
        if batch_size is None:
            num_batches = DEFAULT_NUM_BATCHES
        elif len(input_data) < batch_size:
            num_batches = DEFAULT_NUM_BATCHES
            print("\n WARNING: Batch size to large default batch size will be used!")
        else:
            num_batches = len(input_data) // batch_size

    num_batches = len(input_data) // batch_size
    input_data, output_data = shuffle_data(key, input_data, output_data)
    input_data = jnp.array_split(input_data, num_batches)
    output_data = jnp.array_split(output_data, num_batches)
    return input_data, output_data


# Utility function to create particle minibatch indices
def create_particle_minibatch_indices(key, num_particles, batch_size):
    """
    Creates minibatches of particle indices for particle minibatching.

    Args:
        key (jax.random.PRNGKey): JAX random key for shuffling the particles.
        num_particles (int): Total number of particles.
        batch_size (int): The size of each minibatch for particles.

    Returns:
        list: A list of minibatched particle indices.
    """
    indices = jax.random.permutation(key, num_particles)
    num_batches = max(1, num_particles // batch_size)
    return jnp.array_split(indices, num_batches)


# Utility function to shuffle data
@jax.jit
def shuffle_data(key, input_data, output_data):
    """
    Shuffles the input and output data based on the given random key.

    Args:
        key (jax.random.PRNGKey): JAX random key for generating the permutation.
        input_data (jax.numpy.ndarray): Input features to shuffle.
        output_data (jax.numpy.ndarray): Output labels to shuffle.

    Returns:
        tuple: Shuffled input and output data.
    """
    permutation = jax.random.permutation(key, input_data.shape[0])
    return jnp.take(input_data, permutation, axis=0), jnp.take(output_data, permutation, axis=0)


def get_evaluation_and_apply_early_stopping_logic(stein_vi, z_val, y_val, iteration, best_eval_metric,
                                                  patience_counter):
    """
    Handles the printing and evaluation during training based on the training print mode.

    Args:
        steinvi : ...
        z_val (ndarray): Validation data inputs.
        y_val (ndarray): Validation data targets.
        iteration (int): Current training iteration.
        best_eval_metric (float): Current best evaluation metric.
        patience_counter (int): Counter for early stopping patience.

    Returns:
        tuple: Updated evaluation metrics, best_eval_metric, and patience_counter.
    """
    if stein_vi.handler._full_training_print:
        current_eval_1, current_eval_2, _ = stein_vi.evaluate_fn(stein_vi.state, z_val, y_val, print_out=True)
    # TODO: reduced prints every 10th iteration is 10 the right choice?
    elif stein_vi.handler._reduced_training_print and iteration % 10 == 0:
        current_eval_1, current_eval_2, _ = stein_vi.evaluate_fn(stein_vi.state, z_val, y_val, print_out=True)
    else:
        current_eval_1, current_eval_2, _ = stein_vi.evaluate_fn(stein_vi.state, z_val, y_val, print_out=False)
    stein_vi.evaluation_metrics_1.append(current_eval_1)
    stein_vi.evaluation_metrics_2.append(current_eval_2)

    if stein_vi.parameter.early_stopping:
        patience_counter, best_eval_metric = early_stopping_fn(
            current_eval_1, best_eval_metric, patience_counter, stein_vi.parameter
        )

    return stein_vi, best_eval_metric, patience_counter


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