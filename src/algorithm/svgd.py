import blackjax
from blackjax.vi.svgd import rbf_kernel, update_median_heuristic
from tqdm import tqdm
import jax
import jax.numpy as jnp

from src.metrics.validation_and_evaluation import get_evaluation_metrics_over_predictions
from src.algorithm.get_posteriori import get_posteriori

DEFAULT_NUM_BATCHES = 10


def train_with_svgd(dataset, nnet_model, tree_def, param_vec, parameter, key):
    """_summary_

    Args:
        dataset (_type_): _description_
        nnet_model (_type_): _description_
        tree_def (_type_): _description_
        param_vec (_type_): _description_
        parameter (_type_): _description_
        key (_type_): _description_

    Returns:
        _type_: _description_
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
    """_summary_

    Args:
        log_p (_type_): _description_
        initial_position (_type_): _description_
        initial_kernel_parameters (_type_): _description_
        kernel (_type_): _description_
        optimizer (_type_): _description_
        nnet_model (_type_): _description_
        tree_def (_type_): _description_
        z_train (_type_): _description_
        y_train (_type_): _description_
        z_val (_type_): _description_
        y_val (_type_): _description_
        svgd_parameter (_type_): _description_

    Returns:
        _type_: _description_
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

    # Define a training step function that JIT compiles the SVGD step
    @jax.jit
    def training_step(state, dz, dy):
        """_summary_

        Args:
            state (_type_): _description_
            dz (_type_): _description_
            dy (_type_): _description_

        Returns:
            _type_: _description_
        """        
        return step(state, dz=dz, dy=dy)
    
    # Define a training step function that JIT compiles the SVGD step with minibatched particles
    @jax.jit
    def training_minibatched_step(state, particle_indices, dz, dy):
        """_summary_

        Args:
            state (_type_): _description_
            particle_indices (_type_): _description_
            dz (_type_): _description_
            dy (_type_): _description_

        Returns:
            _type_: _description_
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
    """_summary_

    Args:
        param_vec (_type_): _description_
        rng_key_init (_type_): _description_
        num_particles (_type_): _description_

    Returns:
        _type_: _description_
    """    
    initial_particles_vector = jax.random.normal(
        rng_key_init,
        shape=(num_particles,) + param_vec.shape
    )
    return initial_particles_vector

    
def create_minibatches(batch_size, input_data, output_data, key):
    """_summary_

    Args:
        batch_size (_type_): _description_
        input_data (_type_): _description_
        output_data (_type_): _description_
        key (_type_): _description_

    Returns:
        _type_: _description_
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
    """_summary_

    Args:
        key (_type_): _description_
        num_particles (_type_): _description_
        batch_size (_type_): _description_

    Returns:
        _type_: _description_
    """    
    if num_particles < batch_size:
        num_batches = 2
        print("\n WARNING: Batch size to large, particle batching with 2 batches will be used!")
    indices = jax.random.permutation(key, num_particles)
    num_batches = max(1, num_particles // batch_size)
    batched_indices = jnp.array_split(indices, num_batches)
    return batched_indices

def get_batched_optimizer_state(optimizer_state, indices):
    """_summary_

    Args:
        optimizer_state (_type_): _description_
        indices (_type_): _description_
    """    

    def batch_fn(x):
        if hasattr(x, 'ndim') and x.ndim > 0:
            return jnp.take(x, indices, axis=0)
        return x

    batched_optimizer_state = jax.tree.map(batch_fn, optimizer_state)
    return batched_optimizer_state

def update_optimizer_state(optimizer_state, batched_state, indices):
    """_summary_

    Args:
        optimizer_state (_type_): _description_
        batched_state (_type_): _description_
        indices (_type_): _description_
    """    
    
    def update_fn(orig, batched):
        if hasattr(orig, 'ndim') and orig.ndim > 0:
            return orig.at[indices].set(batched)
        return orig
    
    updated_optimizer_state = jax.tree.map(update_fn, optimizer_state, batched_state.opt_state)
    return updated_optimizer_state

# TODO: Ref better for variable count
def update_optimizer_iteration(state):
    """_summary_

    Args:
        state (_type_): _description_
    """    

    def increment_count_fn(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.integer):
            return x + 1
        return x

    new_optimizer_state = jax.tree.map(increment_count_fn, state.opt_state)
    return state._replace(opt_state=new_optimizer_state)

@jax.jit
def shuffle_paired_data(key, input_data, output_data):
    """_summary_

    Args:
        key (_type_): _description_
        input_data (_type_): _description_
        output_data (_type_): _description_

    Returns:
        _type_: _description_
    """    
    num_samples = input_data.shape[0]
    permutation = jax.random.permutation(key, num_samples)
    shuffled_input = jnp.take(input_data, permutation, axis=0)
    shuffled_output = jnp.take(output_data, permutation, axis=0)
    return shuffled_input, shuffled_output


def check_for_early_stopping(val_accuracy, best_evaluation_metrics_1, iteration, state, best_state, patience_counter,
                             parameter):
    """check_for_early_stopping

    Args:
        val_accuracy (_type_): _description_
        best_evaluation_metrics_1 (_type_): _description_
        iteration (_type_): _description_
        state (_type_): _description_
        best_state (_type_): _description_
        patience_counter (_type_): _description_
        parameter (_type_): _description_

    Returns:
        best_state: test
    """    
    # Apply early stopping logic only after warm-up period
    if iteration >= parameter.warm_up_iterations_early_stopping:
        if val_accuracy > best_evaluation_metrics_1 + parameter.min_delta_early_stopping:
            best_evaluation_metrics_1 = val_accuracy
            patience_counter = 0
            best_state = state
        else:
            patience_counter += 1
            return None, best_evaluation_metrics_1, patience_counter
    return best_state, best_evaluation_metrics_1, patience_counter