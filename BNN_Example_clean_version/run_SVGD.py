import jax
import tensorflow as tf

from regression_toy_example import get_regression_toy_example
from svgd import train_with_svgd
from validation_and_evaluation import evaluate_mse_on_test_data
from data_handling import apply_data_settings_keras


def run_svgd_on_regression_toy_example(network_structure=(200, 75, 40), output_size=2, num_particles=100,
                                       batch_size=20):
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(1)

    # Load the dataset and split into training, validation, and test sets
    regression_toy_example_data = get_regression_toy_example(num_points=100, key=key)

    out, z_test, y_test, nnet_model, tree_def = train_with_svgd(regression_toy_example_data, output_size,
                                                                network_structure, batch_size, num_particles,
                                                                key, regression=True)

    evaluate_mse_on_test_data(out.particles, z_test, y_test, nnet_model, tree_def)


def run_svgd_on_multiclass_data(dataset, network_structure=(200, 75, 40), output_size=10, num_particles=100,
                                batch_size=100):
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(1)

    out, z_test, y_test, nnet_model, tree_def = train_with_svgd(dataset, output_size,
                                                                network_structure, batch_size, num_particles,
                                                                key, regression=False)

    # TODO evaluate_mse_on_test_data(out.particles, z_test, y_test, nnet_model, tree_def)


def run_MNIST():
    mnist = tf.keras.datasets.mnist
    dataset = apply_data_settings_keras(mnist.load_data())

    run_svgd_on_multiclass_data(dataset, network_structure=(200, 75, 40), output_size=10, num_particles=100,
                                batch_size=3000)


if __name__ == "__main__":
    run_MNIST()
