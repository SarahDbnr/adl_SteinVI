import functools
import jax
import jax.numpy as jnp
import datetime
from optax import adam, exponential_decay
import matplotlib.pyplot as plt
import blackjax
from blackjax.vi.svgd import rbf_kernel, update_median_heuristic
from tqdm import tqdm

from BNN_Model import build_model
from get_posteriori import logp_unnormalized_posterior_mulitnomial
from Load_Data import load_data
from plot_mse_accuracy import plot_and_save_accuracy

def main():

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(1)
    rng_key_observed, rng_key_init = jax.random.split(key, 2)

    ###################################
    ### Parameters to run the model ###
    ###################################

    num_iterations = 30
    num_particles = 2
    kernel_length = 0.05
    batch_size =  "Full" #Full
    
    # Number of warm-up iterations before starting early stopping
    warm_up_iterations = 150

    # Network structure for the NNet
    network_structure = (200, 75, 40)

    # Learning rate schedule with exponential_decay for the optimizer if needed
    initial_learning_rate = 0.05
    decay_rate = 0.95  # Learning rate decay rate
    decay_steps = 100  # Learning rate decay steps
    learning_rate_schedule = exponential_decay(
        init_value=initial_learning_rate,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )#https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html
    optimizer = adam(learning_rate_schedule)
    
    # If you don't want to use a learning rate schedule, you can use a fixed learning rate
    #optimizer = adam(0.01)

    # Load the dataset for the experiment
    # Available options: "MNIST", "FashionMNIST", "CIFAR10"
    dataset = "MNIST"

    # Load the dataset and split into training, validation, and test sets
    z_train, y_train, z_val, y_val, z_test, y_test = load_data(dataset, reduce_size=False)
    
    # Build the Neural Network model based on set input parameters
    nnet_model, tree_def, param_vec = build_model(key, z_train, output_size=10, hidden_layers=(network_structure))

    # Initialize particles for the SVGD algorithm
    prior_mu, _, initial_particles_vector = initialize_particles(param_vec, rng_key_init, num_particles)

    @jax.jit
    def logp_model(params, dz, dy):
        return logp_unnormalized_posterior_mulitnomial(
            params,
            nnet_model=nnet_model,
            dz=dz,
            dy=dy,
            prior_mu=prior_mu,
            treedef=tree_def,
        )
    if batch_size != "Full":
        # Create minibatches
        num_batches = len(z_train) // batch_size
        z_train = jnp.array_split(z_train, num_batches)
        y_train = jnp.array_split(y_train, num_batches)
    # Run SVGD training loop with Adam optimizer and validation accuracy tracking
    out, val_accuracies = svgd_training_loop(
        log_p=logp_model,
        initial_position=initial_particles_vector,
        initial_kernel_parameters={"length_scale": kernel_length},
        kernel=rbf_kernel,
        optimizer=optimizer,
        num_iterations=num_iterations,
        nnet_model=nnet_model,
        tree_def=tree_def,
        z_train=z_train,
        y_train=y_train,
        z_val=z_val,
        y_val=y_val,
        warm_up_iterations=warm_up_iterations,
        batch_size=batch_size
    )

    _, test_accuracy = evaluate_particles(out, nnet_model, tree_def, z_test, y_test)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    plot_and_save_accuracy(
        val_accuracies,
        num_particles=num_particles,
        network_structure=network_structure,
        kernel_length=kernel_length,
        adam_learning_rate=initial_learning_rate,
        warm_up_iterations=warm_up_iterations,
        output_folder="svgd_plots"  
    )

# Evaluate the particles by averaging the predictions and calculating the accuracy
def evaluate_particles(out, nnet_model, tree_def, x_test, y_test):
    num_particles = len(out.particles)
    all_predictions = []
    
    for i in range(num_particles):
        particle_params = tree_def(out.particles[i])
        logits = nnet_model.apply(particle_params, x_test)
        probabilities = jax.nn.softmax(logits, axis=-1)
        all_predictions.append(probabilities)
    
    all_predictions = jnp.stack(all_predictions, axis=0)
    averaged_predictions = jnp.mean(all_predictions, axis=0)
    predicted_classes = jnp.argmax(averaged_predictions, axis=-1)
    accuracy = jnp.mean(predicted_classes == y_test)
    
    return averaged_predictions, accuracy

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
        batch_size = "Full"
):
    grad_log_posterior = jax.grad(log_p)
    svgd = blackjax.svgd(grad_log_posterior, optimizer, kernel, update_median_heuristic)
    state = svgd.init(initial_position, initial_kernel_parameters)
    step = jax.jit(svgd.step)

    val_accuracies = []
    best_val_accuracy = float('-inf')
    patience_counter = 0
    best_state = None

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
            state = training_step(state,z_train,y_train)
        # Evaluate the particles on the validation set for plotting, early stopping and output during training
        _, val_accuracy = evaluate_particles(state, nnet_model, tree_def, z_val, y_val)
        print(val_accuracy)
        val_accuracies.append(val_accuracy)

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

    return best_state, val_accuracies

if __name__ == "__main__":
    main()
