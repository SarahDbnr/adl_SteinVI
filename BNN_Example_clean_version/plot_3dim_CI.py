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


def plot_results(num_particles, nnet_model, out, tree_def, input_1, input_2, true_output):
    mean_predictions, std_predictions, predictions, true_y, x_grid = get_predictions_from_test_grid(num_particles, nnet_model, out, tree_def, jnp.column_stack((input_1, input_2)))
    
    # Create a grid for surface plotting
    xi = jnp.linspace(input_1.min(), input_1.max(), 100)
    yi = jnp.linspace(input_2.min(), input_2.max(), 100)
    X, Y = jnp.meshgrid(xi, yi)

    # # Create interpolation function
    # from jax import vmap
    # def rbf_kernel(x1, x2, length_scale=1.0):
    #     return jnp.exp(-jnp.sum((x1 - x2)**2) / (2 * length_scale**2))

    # def interpolate(x_new, x_train, y_train):
    #     dists = vmap(lambda x: rbf_kernel(x, x_train))(x_new)
    #     weights = dists / dists.sum(axis=1, keepdims=True)
    #     return jnp.dot(weights, y_train)

    # # Interpolate mean predictions and std predictions
    # grid_points = jnp.column_stack([X.ravel(), Y.ravel()])
    # Z_mean = interpolate(grid_points, jnp.column_stack([input_1, input_2]), mean_predictions).reshape(X.shape)
    # Z_std = interpolate(grid_points, jnp.column_stack([input_1, input_2]), std_predictions).reshape(X.shape)

    fig = plt.figure(figsize=(20, 15))
    
    # Plot the observed data and predictions
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot real data
    scatter = ax.scatter(input_1, input_2, true_output, c='red', s=20, label="Observed Data")
    
    # Plot mean prediction surface
    surf_mean = ax.plot_surface(X, Y, Z_mean, cmap='viridis', alpha=0.7)
    
    # Plot confidence interval surfaces
    surf_upper = ax.plot_surface(X, Y, Z_mean + 2*Z_std, cmap='autumn', alpha=0.3)
    surf_lower = ax.plot_surface(X, Y, Z_mean - 2*Z_std, cmap='winter', alpha=0.3)

    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_zlabel("Output")
    ax.set_title("Observed Data, Mean Prediction, and Confidence Intervals")

    # Create a custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10),
                    Line2D([0], [0], linestyle='-', color='blue'),
                    Line2D([0], [0], linestyle='-', color='orange'),
                    Line2D([0], [0], linestyle='-', color='green')]
    ax.legend(custom_lines, ['Observed Data', 'Mean Prediction', '+2 Std Dev', '-2 Std Dev'])

    plt.tight_layout()
    plt.show()

    # Additional scatter plot for individual predictions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(num_particles):
        ax.scatter(input_1, input_2, predictions[:, i], alpha=0.1, s=1)
    
    ax.scatter(input_1, input_2, true_y, c='red', s=10, alpha=0.5, label='True y')
    ax.scatter(input_1, input_2, mean_predictions, c='black', s=10, alpha=0.5, label='Mean Predictions')
    
    ax.set_xlabel('Input 1')
    ax.set_ylabel('Input 2')
    ax.set_zlabel('Output')
    ax.set_title(f'Predictions for {num_particles} particles')
    ax.legend()
    plt.show()
