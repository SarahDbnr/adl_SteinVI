from optax import adam, exponential_decay
import jax
import tensorflow as tf

from stein_vi.Classes.SteinVI_BNN import SteinVI_BNN
from stein_vi.stein_vi import train_with_stein_vi
from run_stein_vi.model.BNN_Model import build_model
from stein_vi.algorithm.random_forest import random_forest

import data.datasets_info as datasets_info
from data.data_handling import apply_data_settings_sklearn, apply_data_settings_keras, newsgroup_datahandling, \
    adult_income_datahandling, bike_sharing_datahandling
from run_stein_vi.data.regression_toy_example import get_regression_toy_example
from sklearn.datasets import load_diabetes, load_wine, load_iris


def run_regression_toy_example():
    """
    Run SVGD on a synthetic regression toy example.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
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

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=True, optimizer=optimizer)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, regression_toy_example, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.plot_residuals(z_test, y_test)
    steinvi_svdg.plot_location_in_relation_to_scale(z_test)


def run_MNIST(info=False):
    """
    Run SVGD on the MNIST dataset for classification.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """
    key = jax.random.PRNGKey(1)

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

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, optimizer=optimizer, batch_size=300,
                               num_iterations=30, num_particles=5)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, mnist_dataset, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=True)
    print(random_forest(dataset=mnist_dataset))


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

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, optimizer=optimizer, batch_size=300,
                               particle_batch_size=30, num_particles=9)

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

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, optimizer=optimizer, batch_size=300,
                               num_iterations=30, num_particles=5)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, fashion_mnist, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=True)
    print(random_forest(dataset=fashion_mnist))


def run_CIFAR10(info=False):
    """
    Run SVGD on the CIFAR-10 dataset for classification.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to True.
    """
    if info:
        datasets_info.print_cifar10_dataset_info()
    key = jax.random.PRNGKey(1)

    cifar10 = tf.keras.datasets.cifar10
    cifar10 = apply_data_settings_keras(cifar10.load_data(), with_flattening=True)
    z_train, _, _, _, z_test, y_test = cifar10

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )
    nnet_model = build_model(output_size=10, hidden_layers=(200, 70, 40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, batch_size=3000, optimizer=optimizer,
                               num_iterations=10, num_particles=3)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, cifar10, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=True)
    print(random_forest(dataset=cifar10))


def run_20_newsgroups(info=False):
    """
    Run SVGD on the 20 Newsgroups text classification dataset.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to True.
    """
    if info:
        datasets_info.print_20_newsgroups_dataset_info()
    key = jax.random.PRNGKey(1)
    newsgroup_dataset = newsgroup_datahandling()
    newsgroup_dataset = apply_data_settings_sklearn(newsgroup_dataset)
    z_train, _, _, _, z_test, y_test = newsgroup_dataset

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )
    nnet_model = build_model(output_size=20, hidden_layers=(200, 200, 70, 40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, batch_size=3000, optimizer=optimizer,
                               num_iterations=10, num_particles=3)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, newsgroup_dataset, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=False)
    print(random_forest(dataset=newsgroup_dataset))


def run_adult_income(info=False):
    """
    Run SVGD on the Adult Income dataset for binary classification.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """
    if info:
        datasets_info.print_adult_income_dataset_info()

    key = jax.random.PRNGKey(1)
    adult_income_dataset = adult_income_datahandling()
    adult_income_dataset = apply_data_settings_sklearn(adult_income_dataset)
    z_train, _, _, _, z_test, y_test = adult_income_dataset

    optimizer = adam(
        exponential_decay(
            init_value=0.001,
            transition_steps=1000,
            decay_rate=0.95,
            staircase=True
        )
    )
    nnet_model = build_model(output_size=2, hidden_layers=(200, 70, 40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, batch_size=300, optimizer=optimizer,
                               num_iterations=40, num_particles=3)

    train_with_stein_vi(steinvi_svdg, adult_income_dataset, key, algorithm="svgd")
    print(random_forest(dataset=adult_income_dataset))


def run_iris(info=False):
    """
    Run SVGD on the Iris dataset for multiclass classification.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """
    if info:
        datasets_info.print_iris_dataset_info()
    iris = load_iris()  # Loading the Iris dataset
    dataset = apply_data_settings_sklearn(iris)

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
    nnet_model = build_model(output_size=3,hidden_layers=(20,30,20,10))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, batch_size=30, optimizer=optimizer,
                               num_iterations=40, num_particles=10)

    train_with_stein_vi(steinvi_svdg, iris_dataset, key, algorithm="svgd")
    print(random_forest(dataset=iris_dataset))
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=False)


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
    z_train, _, _, _, _, _ = dataset

    key = jax.random.PRNGKey(1)

    optimizer = adam(
        exponential_decay(
            init_value=0.01,
            transition_steps=50,
            decay_rate=0.95,
            staircase=True
        )
    )
    nnet_model = build_model(output_size=2, hidden_layers=(200, 70, 40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=True, batch_size=30, optimizer=optimizer,
                               num_iterations=100, num_particles=10)
    train_with_stein_vi(steinvi_svdg, dataset, key, algorithm="svgd")
    print(random_forest(dataset=dataset, task_type='regression'))


# auch sehr schlecht
def run_wine_quality(info=False):
    """
    Run SVGD on the Wine Quality dataset for multiclass classification.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """
    if info:
        datasets_info.print_wine_quality_dataset_info()
    wine_quality = load_wine()
    dataset = apply_data_settings_sklearn(wine_quality)
    z_train, _, _, _, _, _ = dataset

    key = jax.random.PRNGKey(1)

    optimizer = adam(
        exponential_decay(
            init_value=0.005,
            transition_steps=10,
            decay_rate=0.95,
            staircase=True
        )
    )
    nnet_model = build_model(output_size=2, hidden_layers=(200, 70, 40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=True, batch_size=30, optimizer=optimizer,
                               num_iterations=100, num_particles=10)
    train_with_stein_vi(steinvi_svdg, dataset, key, algorithm="svgd")
    print(random_forest(dataset=dataset, task_type='regression'))


# TODO: Schauen ob der datensatz richtig geladen wird sehr schlecht bei Random forrest aber fur uns ganz ok
def run_bike_sharing(info=False):
    """Runs stein vi for the bike sharing dataset with specified parameters.

    Args:
        info (bool, optional): Specifies if a small overview of the dataset should be printed to the consol, includes e.g. the number of observations in the dataset. Defaults to False.
    """
    if info:
        datasets_info.print_bike_sharing_dataset_info()
    bike_sharing_dataset = bike_sharing_datahandling()
    dataset = apply_data_settings_sklearn(bike_sharing_dataset)
    z_train, _, _, _, _, _ = dataset

    key = jax.random.PRNGKey(1
                             )
    optimizer = adam(
        exponential_decay(
            init_value=0.5,
            transition_steps=10,
            decay_rate=0.95,
            staircase=True
        )
    )
    nnet_model = build_model(output_size=2, hidden_layers=(200, 70, 40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=True, batch_size=300, optimizer=optimizer,
                               num_iterations=100, num_particles=10)
    train_with_stein_vi(steinvi_svdg, dataset, key, algorithm="svgd")
    print(random_forest(dataset=dataset, task_type='regression'))


if __name__ == "__main__":
    run_regression_toy_example()
