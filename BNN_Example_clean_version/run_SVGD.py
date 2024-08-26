import jax
import tensorflow as tf

from regression_toy_example import get_regression_toy_example
from svgd import train_with_svgd
from validation_and_evaluation import get_mse_and_accuracy_over_predictions
from data_handling import apply_data_settings_sklearn, apply_data_settings_keras
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine


def run_svgd_on_regression(dataset, network_structure=(200, 75, 40), output_size=2, num_particles=100, batch_size=20):
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(1)

    out, z_test, y_test, nnet_model, tree_def = train_with_svgd(dataset, output_size,
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
    dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)

    run_svgd_on_multiclass_data(dataset, network_structure=(200, 75, 40), output_size=10, num_particles=2,
                                batch_size=300)


def run_FashionMNIST():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    dataset = apply_data_settings_keras(fashion_mnist.load_data(), with_flattening=False)

    run_svgd_on_multiclass_data(dataset, network_structure=(200, 75, 40), output_size=10, num_particles=2,
                                batch_size=300)


def run_CIFAR10():
    cifar10 = tf.keras.datasets.cifar10
    dataset = apply_data_settings_keras(cifar10.load_data(), with_flattening=True)

    run_svgd_on_multiclass_data(dataset, network_structure=(200, 75, 40), output_size=10, num_particles=2,
                                batch_size=3000)


def run_regression_toy_example():
    regression_toy_example = get_regression_toy_example(num_points=100)

    run_svgd_on_regression(regression_toy_example, network_structure=(200, 75, 40), output_size=2, num_particles=100,
                           batch_size=20)


def run_california_housing(): #TODO: Analyse, dosnt work properly for all stages depending on batch size
    california_housing = fetch_california_housing()
    dataset = apply_data_settings_sklearn(california_housing)

    run_svgd_on_regression(dataset, network_structure=(200, 75, 40), output_size=2, num_particles=100,
                           batch_size=2000)


def run_diabetes():
    california_housing = load_diabetes()
    dataset = apply_data_settings_sklearn(california_housing)

    run_svgd_on_regression(dataset, network_structure=(200, 75, 40), output_size=2, num_particles=100,
                           batch_size=300)


def run_wine_quality():
    california_housing = load_wine()
    dataset = apply_data_settings_sklearn(california_housing)

    run_svgd_on_regression(dataset, network_structure=(200, 75, 40), output_size=2, num_particles=100,
                           batch_size=100)


if __name__ == "__main__":
    run_MNIST()
