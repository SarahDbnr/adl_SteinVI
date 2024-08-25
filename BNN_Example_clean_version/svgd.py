from optax import adam, exponential_decay
import blackjax
from blackjax.vi.svgd import rbf_kernel, update_median_heuristic
from tqdm import tqdm
import jax
import jax.numpy as jnp

from validation_and_evaluation import evaluate_particles_regression, evaluate_particles

NUM_ITERATIONS = 30
# Number of warm-up iterations before starting early stopping
WARM_UP_ITERATIONS = 150
KERNEL_LENGTH = 0.05
# Learning rate schedule with exponential_decay for the optimizer if needed
INITIAL_LEARNING_RATE = 0.05
DECAY_RATE = 0.95  # Learning rate decay rate
DECAY_STEPS = 100  # Learning rate decay steps


def run_svgd(dataset, batch_size, nn_model, logp_model, num_particles, key, regression):
    z_train, y_train, z_val, y_val, z_test, y_test = dataset
    nnet_model, tree_def, param_vec = nn_model

    # Initialize particles for the SVGD algorithm
    rng_key_observed, rng_key_init = jax.random.split(key, 2)
    prior_mu, prior_prec, initial_particles_vector = initialize_particles(param_vec, rng_key_init, num_particles)

    if batch_size != "Full":
        # Create minibatches
        num_batches = len(z_train) // batch_size
        z_train = jnp.array_split(z_train, num_batches)
        y_train = jnp.array_split(y_train, num_batches)

    # Run SVGD training loop with Adam optimizer and validation accuracy tracking
    out, val_accuracies = svgd_training_loop(
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
        warm_up_iterations=WARM_UP_ITERATIONS,
        batch_size=batch_size,
        regression=regression
    )
    return out, val_accuracies


#  Initialize particles for the SVGD algorithm
def initialize_particles(param_vec, rng_key_init, num_particles):
    inital_param_len = len(param_vec)
    prior_mu = jnp.zeros(inital_param_len)
    prior_prec = jnp.ones(inital_param_len)
    initial_particles_vector = jax.random.normal(
        rng_key_init,
        shape=(num_particles,) + prior_mu.shape
    )
    return prior_mu, prior_prec, initial_particles_vector


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
        patience=100,
        min_delta=0.01,
        warm_up_iterations=300,
        batch_size="Full",
        regression=False,
):
    grad_log_posterior = jax.grad(log_p)
    svgd = blackjax.svgd(grad_log_posterior, optimizer, kernel, update_median_heuristic)
    state = svgd.init(initial_position, initial_kernel_parameters)
    step = jax.jit(svgd.step)

    val_accuracies = []
    best_val_accuracy = float('-inf')
    patience_counter = 0
    best_state = None
    mse = []

    # Define a training step function that JIT compiles the SVGD step
    @jax.jit
    def training_step(state, dz, dy):
        return step(state, dz=dz, dy=dy)

    for iteration in tqdm(range(num_iterations), desc="SVGD Training"):
        if batch_size != "Full":
            print("Batching")
            for batch_idx in range(len(y_train)):
                state = training_step(state, z_train[batch_idx], y_train[batch_idx])
        else:
            print("Full")
            state = training_step(state, z_train, y_train)
        if regression:
            mse.append(evaluate_particles_regression(state, nnet_model=nnet_model, tree_def=tree_def, x_test=z_val,
                                                     y_test=y_val))
        else:
            # Evaluate the particles on the validation set for plotting, early stopping and output during training
            _, val_accuracy = evaluate_particles(state, nnet_model, tree_def, z_val, y_val)
            print(val_accuracy)
            val_accuracies.append(val_accuracy)
            ####TODO: Für MSE anpassen
            # Apply early stopping logic only after warm-up period
            if iteration >= warm_up_iterations:
                if val_accuracy > best_val_accuracy + min_delta:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    best_state = state
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered at iteration {iteration + 1}")
                    break

    # If we didn't trigger early stopping, make sure we return the best state
    if best_state is None:
        best_state = state

    if regression:
        return best_state, mse
    else:
        return best_state, val_accuracies


def get_adam_optimizer():
    learning_rate_schedule = exponential_decay(
        init_value=INITIAL_LEARNING_RATE,
        transition_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True
    )  # https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html

    # If you don't want to use a learning rate schedule, you can use a fixed learning rate
    # optimizer = adam(0.01)

    return adam(learning_rate_schedule)
