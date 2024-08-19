import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from BNN_Model import get_true_y


def calculate_mse(nnet_model, treedef, random_input_1, random_input_2, state, mean_squared_errors):
    # Calculate predictions from the particles
    prediction = jax.vmap(lambda p: nnet_model.apply(treedef(p), random_input_1, random_input_2))(state.particles).mean(
        0)
    ### ACHTUNG! Batch dimension ist hier nicht berücksichtigt!! Das Net nimmt hier quasi einen Datenpunkt!
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
    plt.figure(figsize=(50, 20))
    plt.plot(range(0, num_points * 1, 1), mse[::1])
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE over Iterations')
    plt.show()


def create_mse_calc_data(num_mse_calc_points=10, minval=-1, maxval=1):
    key = jax.random.PRNGKey(1)
    subkey1, subkey2 = jax.random.split(key, 2)
    random_input_1 = jax.random.uniform(subkey1, (num_mse_calc_points, 1), minval=minval, maxval=maxval)
    random_input_2 = jax.random.uniform(subkey2, (num_mse_calc_points, 1), minval=minval, maxval=maxval)
    mean_squared_errors = jnp.array([])
    return random_input_1, random_input_2, mean_squared_errors
