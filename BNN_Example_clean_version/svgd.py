import blackjax
from blackjax.vi.svgd import rbf_kernel, update_median_heuristic
from tqdm import tqdm
import jax
import jax.numpy as jnp

from BNN_Example_clean_version.validation_and_evaluation import get_evaluation_metrics_over_predictions
from BNN_Example_clean_version.get_posteriori import get_posteriori

DEFAULT_NUM_BATCHES = 10


def train_with_svgd(dataset, nnet_model, tree_def, param_vec, parameter, key):
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
    grad_log_posterior = jax.grad(log_p)
    svgd = blackjax.svgd(grad_log_posterior, optimizer, kernel, update_median_heuristic)
    state = svgd.init(initial_position, initial_kernel_parameters)
    step = jax.jit(svgd.step)

    best_evaluation_metrics_1 = float('-inf')
    patience_counter = 0
    best_state = None
    evaluation_metrics_1 = []  # mse and accuracy
    evaluation_metrics_2 = []  # val_accuracies

    # Define a training step function that JIT compiles the SVGD step
    @jax.jit
    def training_step(state, dz, dy):
        return step(state, dz=dz, dy=dy)

    for iteration in tqdm(range(svgd_parameter.num_iterations), desc="SVGD Training"):
        if svgd_parameter.batch_size != 0:
            key = jax.random.PRNGKey(iteration)
            z_train_batched, y_train_batched = create_minibatches(svgd_parameter.batch_size, z_train, y_train, key)
            for training_batch_input, training_batch_output in zip(z_train_batched, y_train_batched):
                state = training_step(state, training_batch_input, training_batch_output)
        else:
            state = training_step(state, z_train, y_train)

        # TODO: Check time effort for mse and accuracy calc and use as option only
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
            print(f"\nPrecision_val: {current_evaluation_metrics_2}")
        else:
            print(f"\nAccuracy: {current_evaluation_metrics_1}")
        best_state, best_evaluation_metrics_1, patience_counter = check_for_early_stopping(current_evaluation_metrics_1,
                                                                                           best_evaluation_metrics_1,
                                                                                           iteration, state, best_state,
                                                                                           patience_counter,
                                                                                           svgd_parameter)
        if patience_counter >= svgd_parameter.patience_early_stopping:
            print(f"Early stopping triggered at iteration {iteration + 1}")
            break

    # If we didn't trigger early stopping, make sure we return the best state
    if best_state is None:
        best_state = state

    return best_state, evaluation_metrics_1, evaluation_metrics_2


def initialize_particles(param_vec, rng_key_init, num_particles):
    initial_particles_vector = jax.random.normal(
        rng_key_init,
        shape=(num_particles,) + param_vec.shape
    )
    return initial_particles_vector


def create_minibatches(batch_size, input_data, output_data, key):
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


@jax.jit
def shuffle_paired_data(key, input_data, output_data):
    num_samples = input_data.shape[0]
    permutation = jax.random.permutation(key, num_samples)
    shuffled_input = jnp.take(input_data, permutation, axis=0)
    shuffled_output = jnp.take(output_data, permutation, axis=0)
    return shuffled_input, shuffled_output


def check_for_early_stopping(val_accuracy, best_evaluation_metrics_1, iteration, state, best_state, patience_counter,
                             parameter):
    # Apply early stopping logic only after warm-up period
    if iteration >= parameter.warm_up_iterations_early_stopping:
        if val_accuracy > best_evaluation_metrics_1 + parameter.min_delta_early_stopping:
            best_evaluation_metrics_1 = val_accuracy
            patience_counter = 0
            best_state = state
        else:
            patience_counter += 1
            return None, float('-inf'), patience_counter
    return best_state, best_evaluation_metrics_1, patience_counter
