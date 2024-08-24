import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import os
import datetime


def calculate_mse(nnet_model, treedef, random_input_1, random_input_2, state, mean_squared_errors):
    # Calculate predictions from the particles
    prediction = jax.vmap(lambda p: nnet_model.apply(treedef(p), random_input_1, random_input_2))(state.particles).mean(
        0)

    # Calculate Mean Squared Error (MSE)
    true_output_sample = get_true_y(random_input_1, random_input_2)
    mse = jnp.mean((prediction.squeeze() - true_output_sample.squeeze()) ** 2)
    if mean_squared_errors is None:
        mean_squared_errors = jnp.array([mse])
    else:
        mean_squared_errors = jnp.append(mean_squared_errors, mse)
    return mean_squared_errors


def plot_mse(mse):
    num_points = len(mse[::1])
    plt.plot(range(0, num_points * 1, 1), mse[::1])
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE over Iterations')
    plt.figure(figsize=(50, 20))
    plt.show()


def create_mse_calc_data(num_mse_calc_points=10, minval=-1, maxval=1, num_iteration=10):
    key = jax.random.PRNGKey(1)
    subkey1, subkey2 = jax.random.split(key, 2)
    random_input_1 = jax.random.uniform(subkey1, (num_mse_calc_points, 1), minval=minval, maxval=maxval)
    random_input_2 = jax.random.uniform(subkey2, (num_mse_calc_points, 1), minval=minval, maxval=maxval)
    mean_squared_errors = jnp.array([])
    return random_input_1, random_input_2, mean_squared_errors


def plot_and_save_accuracy(val_accuracies, num_particles, network_structure, kernel_length, adam_learning_rate,
                           warm_up_iterations, output_folder="plots"):
    # Create the output folder in the current directory if it doesn't exist
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, output_folder)
    os.makedirs(output_path, exist_ok=True)

    actual_iterations = len(val_accuracies)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, actual_iterations + 1), val_accuracies, label='Validation Accuracy')
    plt.title("Validation Accuracy over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.legend()

    info_text = (
        f"Particles: {num_particles}\n"
        f"Network: {network_structure}\n"
        f"Kernel length: {kernel_length}\n"
        f"Adam learning rate: {adam_learning_rate}\n"
        f"Actual iterations: {actual_iterations}\n"
        f"Warm-up iterations: {warm_up_iterations}"
    )
    plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()

    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"validation_accuracy_{timestamp}.png"

    # Save the plot in the specified folder
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Plot saved as: {filepath}")
