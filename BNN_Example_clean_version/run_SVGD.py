import jax
import tensorflow as tf

from regression_toy_example import get_regression_toy_example
from svgd import train_with_svgd
from validation_and_evaluation import get_mse_and_accuracy_over_predictions
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

    print("For Test Data:")
    mse, accuracy = get_mse_and_accuracy_over_predictions(out, nnet_model, tree_def, z_test, y_test,
                                                          model_regression=True)
    # TODO plot_mse(accuracy)
    # TODO plot_mse(mse)


def run_svgd_on_multiclass_data(dataset, network_structure=(200, 75, 40), output_size=10, num_particles=2,
                                batch_size=100):
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(1)

    out, z_test, y_test, nnet_model, tree_def = train_with_svgd(dataset, output_size,
                                                                network_structure, batch_size, num_particles,
                                                                key, regression=False)

    print("For Test Data:")
    _, accuracy = get_mse_and_accuracy_over_predictions(out, nnet_model, tree_def, z_test, y_test,
                                                        model_regression=False)
    # TODO plot_mse(accuracy)


def run_MNIST():
    mnist = tf.keras.datasets.mnist
    dataset = apply_data_settings_keras(mnist.load_data())

    run_svgd_on_multiclass_data(dataset, network_structure=(200, 75, 40), output_size=10, num_particles=100,
                                batch_size=3000)


if __name__ == "__main__":
    run_MNIST()
