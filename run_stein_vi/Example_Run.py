import jax
import tensorflow as tf
from optax import adam, exponential_decay

from sklearn.datasets import load_diabetes, load_iris

from stein_vi.Classes.SteinVI_BNN_Class import SteinVI_BNN
from stein_vi.stein_vi import train_with_stein_vi
from run_stein_vi.model.BNN_Model import build_model
from run_stein_vi.data.regression_toy_example import get_regression_toy_example
import data.datasets_info as datasets_info
from data.data_handling import apply_data_settings_sklearn, apply_data_settings_keras


def run_regression_toy_example():
    """
    Run SVGD on a synthetic regression toy example.
    """

    key = jax.random.PRNGKey(1)

    regression_toy_example = get_regression_toy_example(num_points=10000)
    z_train, _, _, _, z_test, y_test = regression_toy_example

    optimizer = adam(
        exponential_decay(
            init_value=0.1,
            transition_steps=20,
            decay_rate=0.95,
            staircase=True
        )
    )

    nnet_model = build_model(output_size=2, hidden_layers=(200, 70, 40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=True, optimizer=optimizer,
                               num_iterations=1000, num_particles=15)
    train_with_stein_vi(steinvi_svdg, regression_toy_example, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.plot_residuals(z_test, y_test)
    steinvi_svdg.plot_location_in_relation_to_scale(z_test)


def run_MNIST(info=False):
    """
    Run SVGD on the MNIST dataset for classification.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """
    key = jax.random.PRNGKey(3000)

    if info:
        datasets_info.print_mnist_dataset_info()

    mnist = tf.keras.datasets.mnist
    mnist_dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)
    z_train, _, _, _, z_test, y_test = mnist_dataset

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )
    nnet_model = build_model(output_size=10, hidden_layers=(200, 70, 40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, image_data=True, use_for_regression=False, optimizer=optimizer,
                               batch_size=300, num_iterations=30, num_particles=5, rf_comparison=True,
                               mode_training_print="full")

    train_with_stein_vi(steinvi_svdg, mnist_dataset, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, key=key)


def run_MNIST_minibatched_particles(info=False):
    """
    Run SVGD on the MNIST dataset with minibatched particles.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """

    key = jax.random.PRNGKey(1)

    if info:
        datasets_info.print_mnist_dataset_info()
    mnist = tf.keras.datasets.mnist
    mnist_dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)
    z_train, _, _, _, _, _ = mnist_dataset

    optimizer = adam(
        0.01
    )
    nnet_model = build_model(output_size=10, hidden_layers=(150, 50, 20))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, optimizer=optimizer, batch_size=300,
                               particle_batch_size=3, num_particles=9, mode_training_print="full", num_iterations=100,
                               early_stopping=True)
    steinvi_svdg.parameter.set_early_stopping(warm_up_iterations=10, patience=5, min_delta=0.01)
    train_with_stein_vi(steinvi_svdg, mnist_dataset, key, algorithm="svgd")


def run_FashionMNIST(info=False):
    """
    Run SVGD on the MNIST dataset for classification.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """
    key = jax.random.PRNGKey(1)

    if info:
        datasets_info.print_fashion_mnist_dataset_info()

    fashion_mnist = tf.keras.datasets.fashion_mnist
    fashion_mnist = apply_data_settings_keras(fashion_mnist.load_data(), with_flattening=False)
    z_train, _, _, _, z_test, y_test = fashion_mnist

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )
    nnet_model = build_model(output_size=10, hidden_layers=(200, 70, 40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, image_data=True, use_for_regression=False, optimizer=optimizer,
                               batch_size=300, num_iterations=30, num_particles=5)

    train_with_stein_vi(steinvi_svdg, fashion_mnist, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, key=key, num_plots=5)


def run_iris(info=False):
    """
    Run SVGD on the Iris dataset for multiclass classification.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """

    if info:
        datasets_info.print_iris_dataset_info()

    key = jax.random.PRNGKey(1)
    iris_dataset = load_iris()
    iris_dataset = apply_data_settings_sklearn(iris_dataset)
    z_train, _, _, _, z_test, y_test = iris_dataset

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )
    nnet_model = build_model(output_size=3, hidden_layers=(20, 30, 20, 10))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, batch_size=30, optimizer=optimizer,
                               num_iterations=40, num_particles=10)

    train_with_stein_vi(steinvi_svdg, iris_dataset, key, algorithm="svgd")
    steinvi_svdg.view_misclassified(z_test, y_test, key=key)


def run_diabetes(info=False):
    """
    Run SVGD on the Diabetes dataset for regression.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """

    if info:
        datasets_info.print_diabetes_dataset_info()
    diabetes = load_diabetes()
    dataset = apply_data_settings_sklearn(diabetes)
    z_train, _, _, _, z_test, y_test = dataset

    key = jax.random.PRNGKey(1)

    optimizer = adam(
        exponential_decay(
            init_value=0.005,
            transition_steps=50,
            decay_rate=0.95,
            staircase=True
        )
    )
    nnet_model = build_model(output_size=2, hidden_layers=(200, 70, 40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=True, batch_size=30, optimizer=optimizer,
                               num_iterations=1000, num_particles=10, rf_comparison=True)
    train_with_stein_vi(steinvi_svdg, dataset, key, algorithm="svgd")
    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.plot_residuals(z_test, y_test)
    steinvi_svdg.plot_location_in_relation_to_scale(z_test)


if __name__ == "__main__":
    run_regression_toy_example()
