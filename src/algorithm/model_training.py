import jax
import jax.numpy as jnp
from tqdm import tqdm

DEFAULT_NUM_BATCHES = 10

def train_general_algorithm(dataset, nnet_model, tree_def, parameter, key, state, update_fn, evaluate_fn, early_stopping_fn, init_update_fn):
    """
    General training function for neural networks that supports different mini-batching modes for both data and particles.

    Args:
        dataset (tuple): A tuple containing training and validation datasets (z_train, y_train, z_val, y_val, etc.).
        nnet_model (object): The neural network model used for predictions.
        tree_def (object): Tree structure for parameter transformations in JAX.
        parameter (Parameter): A Parameter object defining training hyperparameters (e.g., batch size, number of iterations).
        key (jax.random.PRNGKey): JAX random key for managing randomness.
        state (object): The initial state of the model, including particles and optimizer state.
        update_fn (callable): Function to update the model parameters during training.
        evaluate_fn (callable): Function to evaluate the model during training.
        early_stopping_fn (callable): Function to apply early stopping criteria.
        init_update_fn (callable): The function to initialize updates during training.

    Returns:
        tuple: Best model state and two lists of evaluation metrics (e.g., accuracy or MSE) during training.
    """

    evaluation_metrics_1, evaluation_metrics_2 = [], []
    best_state = None

    if parameter.batch_size == 0 and parameter.particle_batch_size == 0:
        training_loop_fn = no_minibatch_training_loop
    elif parameter.batch_size != 0 and parameter.particle_batch_size == 0:
        training_loop_fn = data_minibatch_training_loop
    elif parameter.batch_size == 0 and parameter.particle_batch_size != 0:
        training_loop_fn = particle_minibatch_training_loop
    else:
        training_loop_fn = data_and_particle_minibatch_training_loop

    best_state, evaluation_metrics_1, evaluation_metrics_2 = training_loop_fn(
        state, dataset, nnet_model, tree_def, parameter, key, update_fn, evaluate_fn, early_stopping_fn, init_update_fn
    )

    return best_state, evaluation_metrics_1, evaluation_metrics_2


def no_minibatch_training_loop(state, dataset, nnet_model, tree_def, parameter, key, update_fn, evaluate_fn, early_stopping_fn, init_update_fn):
    """
    Training loop without mini-batching, using full data and particles.

    Args:
        state (object): The current state of the model.
        dataset (tuple): The dataset containing training and validation data.
        nnet_model (object): The neural network model for predictions.
        tree_def (object): Tree structure for parameter transformations in JAX.
        parameter (Parameter): A Parameter object containing training hyperparameters.
        key (jax.random.PRNGKey): JAX random key for randomness.
        update_fn (callable): Function to update model parameters.
        evaluate_fn (callable): Function to evaluate the model.
        early_stopping_fn (callable): Function to apply early stopping criteria.
        init_update_fn (callable): Initialization function for training updates.

    Returns:
        tuple: Updated model state and two lists of evaluation metrics (e.g., accuracy or MSE).
    """
    z_train, y_train, z_val, y_val, _, _  = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0
    evaluation_metrics_1, evaluation_metrics_2 = [], []

    for _ in tqdm(range(parameter.num_iterations), desc="Training"):
        state = update_fn(state, z_train, y_train, init_update_fn)  # Full data and full particles
        current_eval_1, current_eval_2, _ = evaluate_fn(state, nnet_model, tree_def, z_val, y_val, parameter)
        evaluation_metrics_1.append(current_eval_1)
        evaluation_metrics_2.append(current_eval_2)

        patience_counter, best_eval_metric = early_stopping_fn(
            current_eval_1, best_eval_metric, patience_counter, parameter
        )

        if patience_counter >= parameter.patience_early_stopping:
            break

    return state, evaluation_metrics_1, evaluation_metrics_2

def data_minibatch_training_loop(state, dataset, nnet_model, tree_def, parameter, key, update_fn, evaluate_fn, early_stopping_fn, init_update_fn):
    """
    Training loop with mini-batching on the data while using the full particle set.

    Args:
        state (object): The current state of the model.
        dataset (tuple): The dataset containing training and validation data.
        nnet_model (object): The neural network model for predictions.
        tree_def (object): Tree structure for parameter transformations in JAX.
        parameter (Parameter): A Parameter object containing training hyperparameters.
        key (jax.random.PRNGKey): JAX random key for randomness.
        update_fn (callable): Function to update model parameters.
        evaluate_fn (callable): Function to evaluate the model.
        early_stopping_fn (callable): Function to apply early stopping criteria.
        init_update_fn (callable): Initialization function for training updates.

    Returns:
        tuple: Updated model state and two lists of evaluation metrics.
    """
    z_train, y_train, z_val, y_val, _, _ = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0
    evaluation_metrics_1, evaluation_metrics_2 = [], []
    key_loop = key
    for _ in tqdm(range(parameter.num_iterations), desc="Training"):
        key_loop, _ = jax.random.split(key_loop)
        z_train_batched, y_train_batched = create_minibatches(parameter.batch_size, z_train, y_train, key_loop)

        for z_batch, y_batch in zip(z_train_batched, y_train_batched):
            state = update_fn(state, z_batch, y_batch, init_update_fn)

        current_eval_1, current_eval_2, _ = evaluate_fn(state, nnet_model, tree_def, z_val, y_val, parameter)
        evaluation_metrics_1.append(current_eval_1)
        evaluation_metrics_2.append(current_eval_2)
        # Check for early stopping
        patience_counter, best_eval_metric = early_stopping_fn(
            current_eval_1, best_eval_metric, patience_counter, parameter
        )

        if patience_counter >= parameter.patience_early_stopping:
            break

    return state, evaluation_metrics_1, evaluation_metrics_2


# Particle minibatching: Use the full dataset but split particles into minibatches
def particle_minibatch_training_loop(state, dataset, nnet_model, tree_def, parameter, key, update_fn, evaluate_fn, early_stopping_fn, init_update_fn):
    """
    Training loop with mini-batching on the particles while using the full dataset.

    Args:
        state (object): The current state of the model.
        dataset (tuple): The dataset containing training and validation data.
        nnet_model (object): The neural network model for predictions.
        tree_def (object): Tree structure for parameter transformations in JAX.
        parameter (Parameter): A Parameter object containing training hyperparameters.
        key (jax.random.PRNGKey): JAX random key for randomness.
        update_fn (callable): Function to update model parameters.
        evaluate_fn (callable): Function to evaluate the model.
        early_stopping_fn (callable): Function to apply early stopping criteria.
        init_update_fn (callable): Initialization function for training updates.

    Returns:
        tuple: Updated model state and two lists of evaluation metrics.
    """
    z_train, y_train, z_val, y_val, _, _  = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0
    evaluation_metrics_1, evaluation_metrics_2 = [], []
    key_loop = key

    for _ in tqdm(range(parameter.num_iterations), desc="Training"):
        key_loop, _ = jax.random.split(key_loop)
        particle_indices_batches = create_particle_minibatch_indices(key_loop, state.particles.shape[0], parameter.particle_batch_size)

        for particle_indices in particle_indices_batches:
            state = update_fn(state, z_train, y_train, init_update_fn, particle_indices)

        current_eval_1, current_eval_2, _ = evaluate_fn(state, nnet_model, tree_def, z_val, y_val, parameter)
        evaluation_metrics_1.append(current_eval_1)
        evaluation_metrics_2.append(current_eval_2)

        # Check for early stopping
        patience_counter, best_eval_metric = early_stopping_fn(
            current_eval_1, best_eval_metric, patience_counter, parameter
        )

        if patience_counter >= parameter.patience_early_stopping:
            break

    return state, evaluation_metrics_1, evaluation_metrics_2


# Both data and particle minibatching
def data_and_particle_minibatch_training_loop(state, dataset, nnet_model, tree_def, parameter, key, update_fn, evaluate_fn, early_stopping_fn, init_update_fn):
    """
    Training loop with mini-batching on both data and particles.

    Args:
        state (object): The current state of the model.
        dataset (tuple): The dataset containing training and validation data.
        nnet_model (object): The neural network model for predictions.
        tree_def (object): Tree structure for parameter transformations in JAX.
        parameter (Parameter): A Parameter object containing training hyperparameters.
        key (jax.random.PRNGKey): JAX random key for randomness.
        update_fn (callable): Function to update model parameters.
        evaluate_fn (callable): Function to evaluate the model.
        early_stopping_fn (callable): Function to apply early stopping criteria.
        init_update_fn (callable): Initialization function for training updates.

    Returns:
        tuple: Updated model state and two lists of evaluation metrics.
    """
    z_train, y_train, z_val, y_val, _, _ = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0
    evaluation_metrics_1, evaluation_metrics_2 = [], []
    key_loop = key

    for _ in tqdm(range(parameter.num_iterations), desc="Training"):
        key_loop, _ = jax.random.split(key_loop)
        z_train_batched, y_train_batched = create_minibatches(parameter.batch_size, z_train, y_train, key_loop)
        particle_indices_batches = create_particle_minibatch_indices(key_loop, state.particles.shape[0], parameter.particle_batch_size)

        for z_batch, y_batch in zip(z_train_batched, y_train_batched):
            for particle_indices in particle_indices_batches:
                state = update_fn(state, z_batch, y_batch, init_update_fn, particle_indices)

        current_eval_1, current_eval_2, _ = evaluate_fn(state, nnet_model, tree_def, z_val, y_val, parameter)
        evaluation_metrics_1.append(current_eval_1)
        evaluation_metrics_2.append(current_eval_2)

        # Check for early stopping
        patience_counter, best_eval_metric = early_stopping_fn(
            current_eval_1, best_eval_metric, patience_counter, parameter
        )

        if patience_counter >= parameter.patience_early_stopping:
            break

    return state, evaluation_metrics_1, evaluation_metrics_2


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
