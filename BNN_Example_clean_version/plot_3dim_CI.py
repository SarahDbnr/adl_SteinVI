import jax.numpy as jnp
from BNN_Model import get_true_y
import matplotlib.pyplot as plt

from tqdm import tqdm


def get_predictions_from_test_grid(num_particles, nnet_model, out, treedef, x_grid):
    mean_predictions = []
    std_predictions = []
    predictions = []
    true_y = []
    for x in tqdm(x_grid, desc="Predicting"):
        x1, x2 = x
        prediction = []
        for i in range(int(num_particles)):
            pred = nnet_model.apply(treedef(out.particles[i]), x1, x2).squeeze()
            prediction.append(pred)
            predictions.append(pred)
        mean_predictions.append(jnp.mean(jnp.array(prediction), axis=0))
        std_predictions.append(jnp.std(jnp.array(prediction), axis=0))
        true_y.append(get_true_y(x1, x2))
    return mean_predictions, std_predictions, predictions, true_y, x_grid


def get_test_samples_grid(x1_values=jnp.linspace(-2, 2, 5), x2_values=jnp.linspace(-2, 2, 5)):
    # Define the range and number of points in each dimension
    x1_grid, x2_grid = jnp.meshgrid(x1_values.flatten(), x2_values.flatten())
    # Combine the grids into a 2D array of shape (200*200, 2)
    x_grid = jnp.stack([x1_grid, x2_grid], axis=-1).reshape(-1, 2)
    return x_grid


def plot_3d_scatterplot(predictions, x_grid, comparison_value1, comparison_value2, num_particles):
    x1_repeated, x2_repeated, x1, x2 = get_x1_x2_from_grid(x_grid, num_particles)
    plt.figure(figsize=(50, 20))
    ax = plt.axes(projection='3d')
    ax.scatter3D(predictions, x1_repeated, x2_repeated, c=predictions, linewidth=0.5)
    ax.scatter3D(comparison_value1, x1, x2, c=comparison_value1, cmap='Reds')
    ax.scatter3D(comparison_value2, x1, x2, c=comparison_value2, cmap='Greens')
    plt.show()


def get_x1_x2_from_grid(x_grid, num_particles):
    x_grid_repeated = jnp.repeat(x_grid[:, None, :], num_particles, axis=1).reshape(-1, 2)
    x1_repeated = x_grid_repeated[:, 0].squeeze()
    x2_repeated = x_grid_repeated[:, 1].squeeze()
    x1 = x_grid[:, 0].squeeze()
    x2 = x_grid[:, 1].squeeze()
    return x1_repeated, x2_repeated, x1, x2
