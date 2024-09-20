import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from stein_vi.metrics.validation_and_evaluation import compute_confidence_intervals_with_2_neurons


def plot_evaluation_metric(evaluation_metric_val, eval_metric, num_particles=None):
    """
    Plots and saves a graph of evaluation metrics over iterations for a neural network model.

    Args:
        evaluation_metric_val (list or array): Values of the evaluation metric over iterations.
        eval_metric (str): Name of the evaluation metric, e.g., 'accuracy', 'loss'.
        num_particles (int, optional): Number of particles used in SVGD.

    """

    actual_iterations = len(evaluation_metric_val)
    plt.figure(figsize=(10, 6))
    plt.title(f"Validation {eval_metric} over Iterations")
    plt.plot(range(1, actual_iterations + 1), evaluation_metric_val, label=eval_metric)
    plt.xlabel("Iteration")
    plt.ylabel(f"Validation {eval_metric}")
    plt.legend()

    info_text = (
        f"Particles: {num_particles}\n"
    )
    plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Close the plot
    plt.close()


def plot_residuals(y_pred, y_true, num_particles=None, network_structure=None,
                   kernel_length=None, adam_learning_rate=None, actual_iterations=None, warm_up_iterations=None,
                   output_folder="svgd_plots"):
    """
    Calculates and plots the residuals of predictions against true values for a regression model.

    Args:
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
    residuals = y_pred.mean(0).squeeze() - y_true

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

    # Show the plot
    plt.show()

    # Close the plot
    plt.close()


def plot_location_in_relation_to_scale(nnet_model, out, z_test, num_particles=None, network_structure=None,
                                       kernel_length=None, adam_learning_rate=None, actual_iterations=None,
                                       warm_up_iterations=None):
    """
    Plots and saves the relationship between predicted location values and scale values for a neural network model.
    Based on the paper: A Deeper Look into Aleatoric and Epistemic Uncertainty Disentanglement by Matias Valdenegro-Toro
    and Daniel Saromo Mori.

    Args:
        nnet_model (flax.linen.Module): The neural network model used for predictions.
        tree_def (jax.tree_util.PyTreeDef): Tree structure used for parameter transformation in JAX.
        out (object???): Output from the model prediction, containing particles.
        z_test (array): Test input features.
        num_particles (int, optional): Number of particles used in SVGD.
        network_structure (str, optional): Description of the network's architecture.
        kernel_length (float, optional): Length parameter for the kernel used in SVGD.
        adam_learning_rate (float, optional): Learning rate for the Adam optimizer.
        actual_iterations (int, optional): Number of iterations actually performed.
        warm_up_iterations (int, optional): Number of warm-up iterations before actual training starts.

    """
    # Calculate predictions
    prediction_location, predicted_scale = compute_confidence_intervals_with_2_neurons(nnet_model, out, z_test)

    # set maximal standard deviation

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(prediction_location, predicted_scale, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Location in relation to Scale Prediction")
    plt.xlabel("Predicted Location values")
    plt.ylabel("Predicted Variance Values")

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

    # Show the plot
    plt.show()

    # Close the plot
    plt.close()


def view_probabilities_classification(precisions_sample, predicted_class, true_class, ax=None):
    """Plots or adds the probabilities for each class with the 95% quantile and .
    Additionally, shows the 25% to 75% quantile range in light blue with alpha=0.5.
    The predicted class is highlighted in green, and the true class is highlighted in dark blue.


    Args:
        precisions_sample (jax.numpy.ndarray): Array of shape (num_particles, num_classes) containing the predicted probabilities
        predicted_class (jax.numpy.ndarray): Array with one element containing the predicted class
        true_class (int): True class of the sample.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): Is the subplot where the probabilities are plotted. Defaults to None.
    """

    # Calculate mean of probabilities across particles
    means = precisions_sample.mean(axis=0)

    # Calculate the 2.5% and 97.5% quantiles for the error bars
    lower_quantiles_25_975 = jnp.quantile(precisions_sample, 0.025, axis=0)
    upper_quantiles_25_975 = jnp.quantile(precisions_sample, 0.975, axis=0)

    # Calculate the 25% and 75% quantiles
    lower_quantiles_25_75 = jnp.quantile(precisions_sample, 0.25, axis=0)
    upper_quantiles_25_75 = jnp.quantile(precisions_sample, 0.75, axis=0)

    # Calculate the error bars as the absolute difference between the quantiles and the mean
    lower_errors_25_975 = jnp.abs(means - lower_quantiles_25_975)
    upper_errors_25_975 = jnp.abs(upper_quantiles_25_975 - means)
    error_bars_25_975 = [lower_errors_25_975, upper_errors_25_975]

    # Calculate the error bars for the 25%-75% range
    lower_errors_25_75 = jnp.abs(means - lower_quantiles_25_75)
    upper_errors_25_75 = jnp.abs(upper_quantiles_25_75 - means)
    error_bars_25_75 = [lower_errors_25_75, upper_errors_25_75]

    # Define the positions for the bars
    x_pos = jnp.arange(len(means))

    # Define the colors for the bars
    colors = ['gray'] * len(means)
    colors[predicted_class] = 'green'
    colors[true_class] = 'darkblue'

    # If no axis is provided, create one
    plot = False
    if ax is None:
        plot = True
        fig, ax = plt.subplots()

    # Plot the 2.5%-97.5% quantile range as the main error bars
    bars = ax.bar(x_pos, means, yerr=error_bars_25_975, align='center', alpha=0.7, ecolor='black', capsize=10,
                  color=colors)

    # Plot the 25%-75% quantile range as a filled area (light blue with alpha=0.5)
    ax.bar(x_pos, means, yerr=error_bars_25_75, align='center', alpha=0.5, ecolor='lightblue', capsize=10, color=colors)

    # Add labels and title
    ax.set_ylabel('Probability')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Class {i}' for i in range(len(means))])
    ax.set_title(f'Predicted Class: {predicted_class}, True Class: {true_class}', pad=20)
    ax.yaxis.grid(True)

    # Highlight predicted and true classes
    bars[predicted_class].set_color('green')
    bars[true_class].set_color('darkblue')

    # Add a legend to describe the colors
    legend_elements = [
        plt.Line2D([0], [0], color='gray', lw=4, label='Other Classes'),
        plt.Line2D([0], [0], color='green', lw=4, label='Predicted Class'),
        plt.Line2D([0], [0], color='darkblue', lw=4, label='True Class'),
        plt.Line2D([0], [0], color='lightblue', lw=4, label='25%-75% Quantile Range'),
        plt.Line2D([0], [0], color='black', lw=4, label='2.5%-97.5% Quantile Range')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # If a figure was created within this function, display it
    if plot:
        plt.tight_layout()
        plt.show()
