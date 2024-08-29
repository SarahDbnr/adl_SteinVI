import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import os
import datetime
import os
import datetime
import matplotlib.pyplot as plt

def plot_and_save_evaluation_metric(evaluation_metric_val,eval_metric, num_particles=None, network_structure=None, kernel_length=None, adam_learning_rate=None,
                           warm_up_iterations=None, output_folder="plots"):
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