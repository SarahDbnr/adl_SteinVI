import blackjax
from blackjax.vi.svgd import rbf_kernel, update_median_heuristic
from tqdm import tqdm
import jax
import jax.numpy as jnp

from src.metrics.validation_and_evaluation import get_evaluation_metrics_over_predictions
from src.algorithm.get_posteriori import get_posteriori

DEFAULT_NUM_BATCHES = 10


def train_with_svgd(dataset, nnet_model, tree_def, param_vec, parameter, key):
    """
    Function for training a neural network with Stein Variational Gradient Descent (SVGD).

    Args:
        dataset (tuple of ndarray): Tuple containing three ndarrays for training, validation, and testing data, 
                                    each split into covariates and targets (e.g., (X_train, y_train), (X_val, y_val), (X_test, y_test)).
        nnet_model (flax.linen.Module): Neural network model to be trained, should be a Flax model instance.
        tree_def (jax.tree_util.PyTreeDef): Automatically generated structure used to flatten and unflatten the 
                                           model parameters. This helps in handling parameters as a single vector.
        param_vec (jax.numpy.ndarray): Flattened array of model parameters initialized for the SVGD algorithm.
        parameter (src.Parameter_Class): Parameter Class object containing all necessary configurations and 
                                         hyperparameters for the SVGD algorithm, such as learning rate, batch size, etc.
        key (jax.random.PRNGKey): PRNG key used for random number generation in JAX, necessary for stochastic 
                                  elements of the SVGD algorithm like shuffling data or initializing parameters.

    Returns:
        tuple: Output of the src.algorithm.svgd.svgd_training_loop is passd through
    """
    z_train, y_train, z_val, y_val, z_test, y_test = dataset

    # Initialize particles for the SVGD algorithm
    rng_key_observed, rng_key_init = jax.random.split(key, 2)
    initial_particles_vector = initialize_particles(param_vec, rng_key_init, parameter.num_particles)

    logp_model = get_posteriori(nnet_model, tree_def, parameter.use_for_regression)

    # Run SVGD training loop with Adam optimizer and validation accuracy tracking
    out, evaluation_metrics_1, evaluation_metrics_2 = svgd_training_loop(
        log_p=logp_model,
        initial_position=initial_particles_vector,
        initial_kernel_parameters={"length_scale": parameter.kernel_length},
        kernel=rbf_kernel,
        optimizer=parameter.optimizer,
        nnet_model=nnet_model,
        tree_def=tree_def,
        z_train=z_train,
        y_train=y_train,
        z_val=z_val,
        y_val=y_val,
        svgd_parameter=parameter,
    )

    return out, evaluation_metrics_1, evaluation_metrics_2

# SVGD training loop with early stopping
def svgd_training_loop(
        log_p,
        initial_position,
        initial_kernel_parameters,
        kernel,
        optimizer,
        nnet_model,
        tree_def,
        z_train,
        y_train,
        z_val,
        y_val,
        svgd_parameter,
):
    """
    Executes the training loop for Stein Variational Gradient Descent (SVGD) on a neural network model.

    Args:
        log_p (Callable): The log posterior function to be optimized.
        initial_position (jax.numpy.ndarray): Initial position of the particles in the parameter space.
        initial_kernel_parameters (dict): Parameters for the RBF kernel used in SVGD.
        kernel (Callable): Kernel function used in SVGD to compute pairwise interactions between particles.
        optimizer (Optimizer): Optimizer used to update particle positions based on the gradients.
        nnet_model (flax.linen.Module): Neural network model for which parameters are optimized.
        tree_def (jax.tree_util.PyTreeDef): Tree structure used for parameter vectorization.
        z_train (jax.numpy.ndarray): Training input data.
        y_train (jax.numpy.ndarray): Training target data.
        z_val (jax.numpy.ndarray): Validation input data.
        y_val (jax.numpy.ndarray): Validation target data.
        svgd_parameter (src.Parameter_Class): Configuration class containing SVGD-specific settings such as batch sizes and number of iterations.

    Returns:
        tuple: The final state of the SVGD optimizer, and lists containing evaluation metrics over training iterations.
    """
    grad_log_posterior = jax.grad(log_p)
    svgd = blackjax.svgd(grad_log_posterior, optimizer, kernel, update_median_heuristic)
    step = jax.jit(svgd.step)
    best_evaluation_metrics_1 = float('-inf')
    patience_counter = 0
    best_state = None
    evaluation_metrics_1 = []  # mse and accuracy
    evaluation_metrics_2 = []  # val_accuracies

    state = svgd.init(initial_position, initial_kernel_parameters)

    @jax.jit
    def training_step(state, dz, dy):
        """
        Performs a single training step for SVGD using the entire given data.

        Args:
            state (object): Current state of the SVGD optimizer, containing particles and optimizer state.
            dz (jax.numpy.ndarray): Input data for training.
            dy (jax.numpy.ndarray): Output data for training, corresponding to dz.

        Returns:
            object: Updated state of the SVGD optimizer after applying one step of gradient updates.
        """
        return step(state, dz=dz, dy=dy)
    
    @jax.jit
    def training_minibatched_step(state, particle_indices, dz, dy):
        """
        Performs a single training step for a minibatch of particles in SVGD, updating only the selected particles.

        Args:
            state (object): Current state of the SVGD optimizer, containing all particles and optimizer state.
            particle_indices (jax.numpy.ndarray): Indices of the particles to update in this step.
            dz (jax.numpy.ndarray): Batch of input data for training.
            dy (jax.numpy.ndarray): Batch of output data for training, corresponding to dz.

        Returns:
            object: Updated state of the SVGD optimizer with only the selected particles updated.
        """     
        # Get the current state of the optimizer and particles
        batch_particles = jnp.take(state.particles, particle_indices, axis=0)
        optimizer_state = state.opt_state
        batch_optimizer_state = get_batched_optimizer_state(optimizer_state, particle_indices)
        # Set the new minibatch state
        batch_state = state._replace(particles=batch_particles, opt_state=batch_optimizer_state)
        
        # Perform a SVGD step with the minibatch
        updated_batch_state = step(batch_state, dz=dz, dy=dy)
            
        # Update the particles and optimizer state
        new_particles = state.particles.at[particle_indices].set(updated_batch_state.particles)
        new_opt_state = update_optimizer_state(optimizer_state, updated_batch_state, particle_indices)
            
        return state._replace(particles=new_particles, opt_state=new_opt_state)

    for iteration in tqdm(range(svgd_parameter.num_iterations), desc="SVGD Training"):
        if svgd_parameter.batch_size != 0:
            key = jax.random.PRNGKey(iteration)
            if svgd_parameter.particle_batch_size != 0:
                z_train_batched, y_train_batched = create_minibatches(svgd_parameter.batch_size, z_train, y_train, key)
                for training_batch_input, training_batch_output in zip(z_train_batched, y_train_batched):
                    particle_indices = create_particle_minibatch_indices(key, state.particles.shape[0], svgd_parameter.particle_batch_size)
                    for indices in particle_indices:
                        state = training_minibatched_step(state, indices, training_batch_input, training_batch_output)
                    state = update_optimizer_iteration(state)
            else:
                z_train_batched, y_train_batched = create_minibatches(svgd_parameter.batch_size, z_train, y_train, key)
                for training_batch_input, training_batch_output in zip(z_train_batched, y_train_batched):
                    state = training_step(state, training_batch_input, training_batch_output)
        elif svgd_parameter.particle_batch_size != 0: 
            particle_indices = create_particle_minibatch_indices(key, state.particles.shape[0], svgd_parameter.particle_batch_size)
            for indices in particle_indices:
                state = training_minibatched_step(state, indices, z_train, y_train)
            state = update_optimizer_iteration(state)
        else:
            state = training_step(state, z_train, y_train)

        current_evaluation_metrics_1, current_evaluation_metrics_2, _ = get_evaluation_metrics_over_predictions(state,
                                                                                                             nnet_model,
                                                                                                             tree_def,
                                                                                                             z_val,
                                                                                                             y_val,
                                                                                                             svgd_parameter.use_for_regression)
        evaluation_metrics_2.append(current_evaluation_metrics_2)
        evaluation_metrics_1.append(current_evaluation_metrics_1)
        if svgd_parameter.use_for_regression:
            print(f"\nMSE_val: {current_evaluation_metrics_1}")
            print(f"Precision_val: {current_evaluation_metrics_2}")
        else:
            print(f"\nAccuracy: {current_evaluation_metrics_1}")
        best_state, best_evaluation_metrics_1, patience_counter = check_for_early_stopping(current_evaluation_metrics_1,
                                                                                           best_evaluation_metrics_1,
                                                                                           iteration, state, best_state,
                                                                                           patience_counter,
                                                                                           svgd_parameter)
        if patience_counter >= svgd_parameter.patience_early_stopping:
            print(f"Early stopping triggered at iteration {iteration + 1}")
            svgd_parameter.stopped_at_iteration = iteration + 1
            break

    if best_state is None:
        best_state = state

    return best_state, evaluation_metrics_1, evaluation_metrics_2

def initialize_particles(param_vec, rng_key_init, num_particles):
    """
    Initializes particles for the SVGD algorithm by sampling from a normal distribution. Each particle is initialized
    to have the same shape as the provided parameter vector but with values drawn from a normal distribution.

    Args:
        param_vec (jax.numpy.ndarray): The initial parameter vector which dictates the shape of each particle.
        rng_key_init (jax.random.PRNGKey): JAX random key used to ensure reproducible results in particle initialization.
        num_particles (int): Number of particles to initialize, which determines the number of separate sets of parameters to optimize.

    Returns:
        jax.numpy.ndarray: An array of initialized particles, where each particle is a perturbed version of the initial parameter vector.
    """
    initial_particles_vector = jax.random.normal(
        rng_key_init,
        shape=(num_particles,) + param_vec.shape
    )
    return initial_particles_vector

    
def create_minibatches(batch_size, input_data, output_data, key):
    """
    Divides the provided input and output data into minibatches of specified size. If the batch size is zero,
    the entire dataset is returned as one batch.

    Args:
        batch_size (int): The number of samples per batch. If set to zero, the entire dataset is treated as a single batch.
        input_data (jax.numpy.ndarray): The complete set of input data.
        output_data (jax.numpy.ndarray): The complete set of output data corresponding to the input data.
        key (jax.random.PRNGKey): JAX random key used for shuffling the data to prevent ordered data from influencing the training.

    Returns:
        list of tuples: Each tuple contains two ndarrays, with the first being a batch of input data and the second being the corresponding batch of output data.
    """
    if batch_size != 0:
        if batch_size is None:
            num_batches = DEFAULT_NUM_BATCHES
        elif len(input_data) < batch_size:
            num_batches = DEFAULT_NUM_BATCHES
            print("\n WARNING: Batch size to large default batch size will be used!")
        else:
            num_batches = len(input_data) // batch_size
        input_data, output_data = shuffle_paired_data(key, input_data, output_data)
        input_data = jnp.array_split(input_data, num_batches)
        output_data = jnp.array_split(output_data, num_batches)
        return input_data, output_data
    return input_data, output_data

def create_particle_minibatch_indices(key, num_particles, batch_size):
    """
    Generates indices for minibatching particles in the SVGD algorithm.

    Args:
        key (jax.random.PRNGKey): Random key used for generating indices.
        num_particles (int): Total number of particles.
        batch_size (int): Number of particles in each minibatch.

    Returns:
        list of jnp.ndarray: List of arrays where each array contains indices for a minibatch of particles.
    """
    if num_particles < batch_size:
        num_batches = 2
        print("\n WARNING: Batch size to large, particle batching with 2 batches will be used!")
    indices = jax.random.permutation(key, num_particles)
    num_batches = max(1, num_particles // batch_size)
    batched_indices = jnp.array_split(indices, num_batches)
    return batched_indices

def get_batched_optimizer_state(optimizer_state, indices):
    """
    Extracts the optimizer state for a specific batch of particles.

    Args:
        optimizer_state (object): The global optimizer state.
        indices (jnp.ndarray): Indices of the particles for which to retrieve the optimizer state.

    Returns:
        object: Optimizer state corresponding to the batched particles.
    """
    def batch_fn(x):
        if hasattr(x, 'ndim') and x.ndim > 0:
            return jnp.take(x, indices, axis=0)
        return x

    batched_optimizer_state = jax.tree.map(batch_fn, optimizer_state)
    return batched_optimizer_state

def update_optimizer_state(optimizer_state, batched_state, indices):
    """
    Updates the global optimizer state with changes from a minibatched optimizer state.

    Args:
        optimizer_state (object): Current global optimizer state.
        batched_state (object): Updated optimizer state from a minibatch step.
        indices (jnp.ndarray): Indices of the particles that were updated.

    Returns:
        object: Updated global optimizer state.
    """
    def update_fn(orig, batched):
        if hasattr(orig, 'ndim') and orig.ndim > 0:
            return orig.at[indices].set(batched)
        return orig
    
    updated_optimizer_state = jax.tree.map(update_fn, optimizer_state, batched_state.opt_state)
    return updated_optimizer_state

# TODO: Ref better for variable count
def update_optimizer_iteration(state):
    """
    Increments optimization-related counters or state properties, such as the number of iterations completed.

    Args:
        state (object): Current state of the SVGD optimizer.

    Returns:
        object: Updated state with incremented properties.
    """
    def increment_count_fn(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.integer):
            return x + 1
        return x

    new_optimizer_state = jax.tree.map(increment_count_fn, state.opt_state)
    return state._replace(opt_state=new_optimizer_state)

@jax.jit
def shuffle_paired_data(key, input_data, output_data):
    """
    Shuffles paired input and output data arrays using a JAX random key.

    Args:
        key (jax.random.PRNGKey): Random key for shuffling data.
        input_data (jax.numpy.ndarray): Input data to shuffle.
        output_data (jax.numpy.ndarray): Output data to shuffle, aligned with input data.

    Returns:
        tuple of jax.numpy.ndarray: Shuffled input and output data.
    """
    num_samples = input_data.shape[0]
    permutation = jax.random.permutation(key, num_samples)
    shuffled_input = jnp.take(input_data, permutation, axis=0)
    shuffled_output = jnp.take(output_data, permutation, axis=0)
    return shuffled_input, shuffled_output


def check_for_early_stopping(val_accuracy, best_evaluation_metrics_1, iteration, state, best_state, patience_counter,
                             parameter):
    """
    Evaluates if early stopping criteria are met based on validation performance metrics after a warmup period.

    Args:
        val_accuracy (float): Validation accuracy of the current iteration.
        best_evaluation_metrics_1 (float): Best validation accuracy observed so far.
        iteration (int): Current iteration count.
        state (object): Current state of the SVGD optimizer.
        best_state (object): Best state of the SVGD optimizer observed so far.
        patience_counter (int): Counter for the number of consecutive iterations without improvement.
        parameter (src.Parameter_Class): Contains settings for early stopping, including the number of iterations to wait without improvement (patience) and the minimum change in validation accuracy required to reset the patience counter (min_delta).

    Returns:
        tuple: Updated best state of the optimizer, best evaluation metric, and patience counter. If stopping criteria are met, also provides a command to halt training.
    """
    if iteration >= parameter.warm_up_iterations_early_stopping:
        if val_accuracy > best_evaluation_metrics_1 + parameter.min_delta_early_stopping:
            best_evaluation_metrics_1 = val_accuracy
            patience_counter = 0
            best_state = state
        else:
            patience_counter += 1
            return None, best_evaluation_metrics_1, patience_counter
    return best_state, best_evaluation_metrics_1, patience_counter