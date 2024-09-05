import os
import datetime
import jax
import matplotlib.pyplot as plt

def plot_and_save_evaluation_metric(evaluation_metric_val, eval_metric, num_particles=None, network_structure=None,
                                    kernel_length=None, adam_learning_rate=None,
                                    warm_up_iterations=None, output_folder="svgd_plots"):
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
