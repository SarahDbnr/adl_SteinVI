import os
import datetime
import jax
import matplotlib.pyplot as plt
from src.algorithm.get_posteriori import link_function
import jax.numpy as jnp
def plot_and_save_evaluation_metric(evaluation_metric_val, eval_metric, num_particles=None, network_structure=None,
                                    kernel_length=None, adam_learning_rate=None,
                                    warm_up_iterations=None, output_folder="svgd_plots"):
    """
    Plots and saves a graph of evaluation metrics over iterations for a neural network model.

    Args:
        evaluation_metric_val (list or array): Values of the evaluation metric over iterations.
        eval_metric (str): Name of the evaluation metric, e.g., 'accuracy', 'loss'.
        num_particles (int, optional): Number of particles used in SVGD.
        network_structure (str, optional): Description of the network's architecture.
        kernel_length (float, optional): Length parameter for the kernel used in SVGD.
        adam_learning_rate (float, optional): Learning rate for the Adam optimizer.
        warm_up_iterations (int, optional): Number of warm-up iterations before actual training starts.
        output_folder (str, optional): The directory where the plot will be saved. Defaults to 'svgd_plots'.

    Prints:
        Path to the saved plot file.
    """
    # Create the output folder in the current directory if it doesn't exist
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, output_folder)
    os.makedirs(output_path, exist_ok=True)

    actual_iterations = len(evaluation_metric_val)
    plt.figure(figsize=(10, 6))
    plt.title(f"Validation {eval_metric} over Iterations")
    plt.plot(range(1, actual_iterations + 1), evaluation_metric_val, label=eval_metric)
    plt.xlabel("Iteration")
    plt.ylabel(f"Validation {eval_metric}")
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
    filename = f"{eval_metric}_{timestamp}.png"

    # Save the plot in the specified folder
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=300)

    # Show the plot
    plt.show()

    # Close the plot
    plt.close()

    print(f"Plot saved as: {filepath}")


def plot_residuals(nnet_model, tree_def, out, z_test, y_true, num_particles=None, network_structure=None,
                   kernel_length=None, adam_learning_rate=None, actual_iterations=None, warm_up_iterations=None,
                   output_folder="svgd_plots"):
    """
    Calculates and plots the residuals of predictions against true values for a regression model.

    Args:
        nnet_model (object): The neural network model used for predictions.
        tree_def (object): Tree structure used for parameter transformation in JAX.
        out (object): Output from the model prediction, containing particles.
        z_test (array): Test input features.
        y_true (array): True output values for the test set.
        num_particles (int, optional): Number of particles used in SVGD.
        network_structure (str, optional): Description of the network's architecture.
        kernel_length (float, optional): Length parameter for the kernel used in SVGD.
        adam_learning_rate (float, optional): Learning rate for the Adam optimizer.
        actual_iterations (int, optional): Number of iterations actually performed.
        warm_up_iterations (int, optional): Number of warm-up iterations before actual training starts.
        output_folder (str, optional): The directory where the plot will be saved. Defaults to 'svgd_plots'.

    Prints:
        Path to the saved residual plot file.
    """
    # Calculate residuals
    predictions, _ = jax.vmap(lambda p: nnet_model.predict(tree_def(p), z_test))(out.particles)
    y_pred = predictions.mean(0)  # Averaging over particles
    y_pred = y_pred.squeeze()
    residuals = y_pred - y_true

    # Create the output folder in the current directory if it doesn't exist
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, output_folder)
    os.makedirs(output_path, exist_ok=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Residual Plot")
    plt.xlabel("True Y Values")
    plt.ylabel("Residuals (Predicted - True)")

    # Add metadata text
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
    filename = f"residual_plot_{timestamp}.png"

    # Save the plot in the specified folder
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=300)

    # Show the plot
    plt.show()

    # Close the plot
    plt.close()

    print(f"Residual plot saved as: {filepath}")



def plot_location_in_relation_to_scale(nnet_model, tree_def, out, z_test, num_particles=None, network_structure=None,
                   kernel_length=None, adam_learning_rate=None, actual_iterations=None, warm_up_iterations=None,
                   output_folder="svgd_plots"):
    """
    Plots and saves the relationship between predicted location values and scale values for a neural network model.

    Args:
        nnet_model (object): The neural network model used for predictions.
        tree_def (object): Tree structure used for parameter transformation in JAX.
        out (object): Output from the model prediction, containing particles.
        z_test (array): Test input features.
        num_particles (int, optional): Number of particles used in SVGD.
        network_structure (str, optional): Description of the network's architecture.
        kernel_length (float, optional): Length parameter for the kernel used in SVGD.
        adam_learning_rate (float, optional): Learning rate for the Adam optimizer.
        actual_iterations (int, optional): Number of iterations actually performed.
        warm_up_iterations (int, optional): Number of warm-up iterations before actual training starts.
        output_folder (str, optional): The directory where the plot will be saved. Defaults to 'svgd_plots'.

    Prints:
        Path to the saved plot file.
    """
    # Calculate predictions
    prediction_location, prediction_var_score = jax.vmap(lambda p: nnet_model.predict(tree_def(p), z_test))(out.particles)
    prediction_location = prediction_location.mean(0)  # Averaging over particles
    prediction_location = prediction_location.squeeze()
    prediction_var_score = prediction_var_score.mean(0)  # Averaging over particles
    prediction_var_score = prediction_var_score.squeeze()
    predicted_scale = jax.vmap(lambda p: link_function(p))(prediction_var_score)
    # set maximal standard deviation

    # Create the output folder in the current directory if it doesn't exist
    current_dir = os.getcwd()
    output_path = os.path.join(current_dir, output_folder)
    os.makedirs(output_path, exist_ok=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(prediction_location, predicted_scale, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Location in relation to Scale Prediction")
    plt.xlabel("Predicted Location values")
    plt.ylabel("Predicted Scale Values")

    # Add metadata text
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
    filename = f"relationship_scale_location{timestamp}.png"

    # Save the plot in the specified folder
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=300)

    # Show the plot
    plt.show()

    # Close the plot
    plt.close()

    print(f"Location Scale plot saved as: {filepath}")
