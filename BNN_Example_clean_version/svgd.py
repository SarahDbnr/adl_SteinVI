from optax import adam, exponential_decay
#import blackjax
#from blackjax.vi.svgd import rbf_kernel, update_median_heuristic
from local_SVGD import rbf_kernel, update_median_heuristic
import local_SVGD
from tqdm import tqdm
import jax
import jax.numpy as jnp

from validation_and_evaluation import get_mse_and_accuracy_over_predictions
from BNN_Model import build_model
from get_posteriori import get_posteriori

NUM_ITERATIONS = 30
# Early stopping parameters
WARM_UP_ITERATIONS = 150
PATIENCE = 100
MIN_DELTA = 0.01
KERNEL_LENGTH = 0.05
# Learning rate schedule with exponential_decay for the optimizer if needed
INITIAL_LEARNING_RATE = 0.05
DECAY_RATE = 0.95  # Learning rate decay rate
DECAY_STEPS = 100  # Learning rate decay steps


def train_with_svgd(dataset, output_size, network_structure, batch_size, num_particles, key, regression, pen_lambda=0):
    z_train, y_train, z_val, y_val, z_test, y_test = dataset
    # TODO: Change batch size to number of batches
    nnet_model, tree_def, param_vec = build_model(key, z_train, output_size=output_size,
                                                  hidden_layers=network_structure,
                                                  use_for_regression=regression)

    # Initialize particles for the SVGD algorithm
    rng_key_observed, rng_key_init = jax.random.split(key, 2)
    initial_particles_vector = initialize_particles(param_vec, rng_key_init, num_particles)

    logp_model = get_posteriori(nnet_model, tree_def, regression, pen_lambda)

    if batch_size != "Full":
        # Create minibatches
        num_batches = len(z_train) // batch_size
        z_train = jnp.array_split(z_train, num_batches)
        y_train = jnp.array_split(y_train, num_batches)

    # Run SVGD training loop with Adam optimizer and validation accuracy tracking
    out, mse, val_accuracies = svgd_training_loop(
        log_p=logp_model,
        initial_position=initial_particles_vector,
        initial_kernel_parameters={"length_scale": KERNEL_LENGTH},
        kernel=rbf_kernel,
        optimizer=get_adam_optimizer(),
        num_iterations=NUM_ITERATIONS,
        nnet_model=nnet_model,
        tree_def=tree_def,
        z_train=z_train,
        y_train=y_train,
        z_val=z_val,
        y_val=y_val,
        batch_size=batch_size,
        regression=regression,
    )

    # TODO: plot mse, val_accuracies

    return out, z_test, y_test, nnet_model, tree_def


#  Initialize particles for the SVGD algorithm
def initialize_particles(param_vec, rng_key_init, num_particles):
    initial_particles_vector = jax.random.normal(
        rng_key_init,
        shape=(num_particles,) + param_vec.shape
    )
    return initial_particles_vector


# SVGD training loop with early stopping
def svgd_training_loop(
        log_p,
        initial_position,
        initial_kernel_parameters,
        kernel,
        optimizer,
        num_iterations,
        nnet_model,
        tree_def,
        z_train,
        y_train,
        z_val,
        y_val,
        batch_size="Full",
        regression=False,
):
    grad_log_posterior = jax.grad(log_p)
    svgd = local_SVGD.as_top_level_api(grad_log_posterior, optimizer, kernel, update_median_heuristic)
    state = svgd.init(initial_position, initial_kernel_parameters)
    step = jax.jit(svgd.step)

    best_val_accuracy = float('-inf')
    patience_counter = 0
    best_state = None
    mse = []
    val_accuracies = []

    # Define a training step function that JIT compiles the SVGD step
    @jax.jit
    def training_step(state, dz, dy):
        return step(state, dz=dz, dy=dy)

    for iteration in tqdm(range(num_iterations), desc="SVGD Training"):
        if batch_size != "Full":
            print("\n Batching")
            for batch_idx in range(len(y_train)):
                state = training_step(state, z_train[batch_idx], y_train[batch_idx])
        else:
            print("\n Full")
            state = training_step(state, z_train, y_train)

        # TODO: Check time effort for mse and accuracy calc and use as option only
        current_mse, val_accuracy = get_mse_and_accuracy_over_predictions(state, nnet_model, tree_def, z_val, y_val, regression)
        val_accuracies.append(val_accuracy)
        mse.append(current_mse)

        best_state, best_val_accuracy, patience_counter = check_for_early_stopping(val_accuracy, best_val_accuracy,
                                                                                   iteration, state, best_state,
                                                                                   patience_counter)
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at iteration {iteration + 1}")
            break

    # If we didn't trigger early stopping, make sure we return the best state
    if best_state is None:
        best_state = state

    return best_state, mse, val_accuracies


def get_adam_optimizer(): # TODO: exponential decay nur als option default const or less drastic decay then exponential
    # stepwise decay check how high decay rate should be
    learning_rate_schedule = exponential_decay(
        init_value=INITIAL_LEARNING_RATE,
        transition_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True
    )  # https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html

    # If you don't want to use a learning rate schedule, you can use a fixed learning rate
    # optimizer = adam(0.01)

    return adam(learning_rate_schedule)


def check_for_early_stopping(val_accuracy, best_val_accuracy, iteration, state, best_state, patience_counter):
    # Apply early stopping logic only after warm-up period
    if iteration >= WARM_UP_ITERATIONS:
        if val_accuracy > best_val_accuracy + MIN_DELTA:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_state = state
        else:
            patience_counter += 1
            return None, float('-inf'), patience_counter
    return best_state, best_val_accuracy, patience_counter
