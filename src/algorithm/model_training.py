import jax
import jax.numpy as jnp
from tqdm import tqdm

DEFAULT_NUM_BATCHES = 10

def train_general_algorithm(dataset, nnet_model, tree_def, parameter, key, state, update_fn, evaluate_fn, early_stopping_fn, init_update_fn):
    """
    General training function that supports different mini-batching modes for both data and particles.
    
    Args are the same as before, but with additional support for different minibatching strategies.
    """

    evaluation_metrics_1, evaluation_metrics_2 = [], []
    best_state = None

    # Determine loop type based on minibatching requirements
    if parameter.batch_size == 0 and parameter.particle_batch_size == 0:
        training_loop_fn = no_minibatch_training_loop
    elif parameter.batch_size != 0 and parameter.particle_batch_size == 0:
        training_loop_fn = data_minibatch_training_loop
    elif parameter.batch_size == 0 and parameter.particle_batch_size != 0:
        training_loop_fn = particle_minibatch_training_loop
    else:
        training_loop_fn = data_and_particle_minibatch_training_loop

    # Execute the appropriate training loop
    best_state, evaluation_metrics_1, evaluation_metrics_2 = training_loop_fn(
        state, dataset, nnet_model, tree_def, parameter, key, update_fn, evaluate_fn, early_stopping_fn, init_update_fn
    )

    return best_state, evaluation_metrics_1, evaluation_metrics_2


# No minibatching: Use the entire dataset and all particles in each iteration
def no_minibatch_training_loop(state, dataset, nnet_model, tree_def, parameter, key, update_fn, evaluate_fn, early_stopping_fn, init_update_fn):
    z_train, y_train, z_val, y_val, _, _  = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0
    evaluation_metrics_1, evaluation_metrics_2, evaluation_metrics_3 = [], [], []
    best_state = None

    for _ in tqdm(range(parameter.num_iterations), desc="Training"):
        state = update_fn(state, z_train, y_train, init_update_fn)  # Full data and full particles
        current_eval_1, current_eval_2, current_eval_3 = evaluate_fn(state, nnet_model, tree_def, z_val, y_val, parameter)
        evaluation_metrics_1.append(current_eval_1)
        evaluation_metrics_2.append(current_eval_2)
        evaluation_metrics_3.append(current_eval_3)

        # Check for early stopping
        best_state, best_eval_metric, patience_counter = early_stopping_fn(
            current_eval_1, best_eval_metric, patience_counter, state, best_state, parameter
        )

        if patience_counter >= parameter.patience_early_stopping:
            break

    return best_state or state, evaluation_metrics_1, evaluation_metrics_2


# Data minibatching: Split the dataset into batches but use all particles
def data_minibatch_training_loop(state, dataset, nnet_model, tree_def, parameter, key, update_fn, evaluate_fn, early_stopping_fn, init_update_fn):
    z_train, y_train, z_val, y_val, _, _ = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0
    evaluation_metrics_1, evaluation_metrics_2, evaluation_metrics_3 = [], [], []
    best_state = None
    key_loop = key
    for _ in tqdm(range(parameter.num_iterations), desc="Training"):
        key_loop, _ = jax.random.split(key_loop)
        z_train_batched, y_train_batched = create_minibatches(parameter.batch_size, z_train, y_train, key_loop)

        for z_batch, y_batch in zip(z_train_batched, y_train_batched):
            state = update_fn(state, z_batch, y_batch, init_update_fn)

        current_eval_1, current_eval_2, current_eval_3 = evaluate_fn(state, nnet_model, tree_def, z_val, y_val, parameter)
        evaluation_metrics_1.append(current_eval_1)
        evaluation_metrics_2.append(current_eval_2)
        evaluation_metrics_3.append(current_eval_3)
        # Check for early stopping
        best_state, best_eval_metric, patience_counter = early_stopping_fn(
            current_eval_1, best_eval_metric, patience_counter, state, best_state, parameter
        )

        if patience_counter >= parameter.patience_early_stopping:
            break

    return best_state or state, evaluation_metrics_1, evaluation_metrics_2


# Particle minibatching: Use the full dataset but split particles into minibatches
def particle_minibatch_training_loop(state, dataset, nnet_model, tree_def, parameter, key, update_fn, evaluate_fn, early_stopping_fn, init_update_fn):
    z_train, y_train, z_val, y_val, _, _  = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0
    evaluation_metrics_1, evaluation_metrics_2, evaluation_metrics_3 = [], [], []
    best_state = None
    key_loop = key

    for _ in tqdm(range(parameter.num_iterations), desc="Training"):
        key_loop, _ = jax.random.split(key_loop)
        particle_indices_batches = create_particle_minibatch_indices(key_loop, state.particles.shape[0], parameter.particle_batch_size)

        for particle_indices in particle_indices_batches:
            state = update_fn(state, z_train, y_train, init_update_fn, particle_indices)

        current_eval_1, current_eval_2, current_eval_3 = evaluate_fn(state, nnet_model, tree_def, z_val, y_val, parameter)
        evaluation_metrics_1.append(current_eval_1)
        evaluation_metrics_2.append(current_eval_2)
        evaluation_metrics_3.append(current_eval_3)

        # Check for early stopping
        best_state, best_eval_metric, patience_counter = early_stopping_fn(
            current_eval_1, best_eval_metric, patience_counter, state, best_state, parameter
        )

        if patience_counter >= parameter.patience_early_stopping:
            break

    return best_state or state, evaluation_metrics_1, evaluation_metrics_2


# Both data and particle minibatching
def data_and_particle_minibatch_training_loop(state, dataset, nnet_model, tree_def, parameter, key, update_fn, evaluate_fn, early_stopping_fn, init_update_fn):
    z_train, y_train, z_val, y_val = dataset
    best_eval_metric = float('-inf')
    patience_counter = 0
    evaluation_metrics_1, evaluation_metrics_2, evaluation_metrics_3 = [], [], []
    best_state = None
    key_loop = key
    for _ in tqdm(range(parameter.num_iterations), desc="Training"):
        key_loop, _ = jax.random.split(key_loop)
        z_train_batched, y_train_batched = create_minibatches(parameter.batch_size, z_train, y_train, key_loop)
        particle_indices_batches = create_particle_minibatch_indices(key_loop, state.particles.shape[0], parameter.particle_batch_size)

        for z_batch, y_batch in zip(z_train_batched, y_train_batched):
            for particle_indices in particle_indices_batches:
                state = update_fn(state, z_batch, y_batch, init_update_fn, particle_indices)

        current_eval_1, current_eval_2, current_eval_3 = evaluate_fn(state, nnet_model, tree_def, z_val, y_val, parameter)
        evaluation_metrics_1.append(current_eval_1)
        evaluation_metrics_2.append(current_eval_2)
        evaluation_metrics_3.append(current_eval_3)

        # Check for early stopping
        best_state, best_eval_metric, patience_counter = early_stopping_fn(
            current_eval_1, best_eval_metric, patience_counter, state, best_state, parameter
        )

        if patience_counter >= parameter.patience_early_stopping:
            break

    return best_state or state, evaluation_metrics_1, evaluation_metrics_2


# Utility function to create minibatches
def create_minibatches(batch_size, input_data, output_data, key):
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
    indices = jax.random.permutation(key, num_particles)
    num_batches = max(1, num_particles // batch_size)
    return jnp.array_split(indices, num_batches)


# Utility function to shuffle data
@jax.jit
def shuffle_data(key, input_data, output_data):
    permutation = jax.random.permutation(key, input_data.shape[0])
    return jnp.take(input_data, permutation, axis=0), jnp.take(output_data, permutation, axis=0)
