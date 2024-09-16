from optax import adam, exponential_decay
import jax
import tensorflow as tf
from stein_vi.Classes.SteinVI_BNN import SteinVI_BNN
from run_stein_vi.model.BNN_Model import build_model
from run_stein_vi.data.regression_toy_example import get_regression_toy_example
import stein_vi.algorithm
import stein_vi.algorithm.random_forest
from stein_vi.stein_vi import train_with_stein_vi
import data.datasets_info as datasets_info
from data.data_handling import apply_data_settings_sklearn, apply_data_settings_keras, newsgroup_datahandling, \
    adult_income_datahandling, bike_sharing_datahandling


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

    nnet_model = build_model(output_size=2)

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=True, optimizer=optimizer)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, regression_toy_example, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.plot_residuals(z_test,y_test)
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
    nnet_model = build_model(output_size=10,hidden_layers=(200,70,40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, optimizer=optimizer, batch_size=300,num_iterations=30, num_particles=5)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, mnist_dataset, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=True)
    print(stein_vi.algorithm.random_forest.random_forest(dataset=mnist_dataset))


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
    nnet_model = build_model(output_size=10,hidden_layers=(200,70,40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, optimizer=optimizer, batch_size=300,particle_batch_size=30, num_particles=9)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, mnist_dataset, key, algorithm="svgd")



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
    nnet_model = build_model(output_size=10,hidden_layers=(200,70,40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, optimizer=optimizer, batch_size=300,num_iterations=30, num_particles=5)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, fashion_mnist, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=True)
    print(stein_vi.algorithm.random_forest.random_forest(dataset=fashion_mnist))

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
    nnet_model = build_model(output_size=10,hidden_layers=(200,70,40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, batch_size=3000,optimizer=optimizer,num_iterations=10, num_particles=3)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, cifar10, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=True)
    print(stein_vi.algorithm.random_forest.random_forest(dataset=cifar10))


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
    nnet_model = build_model(output_size=20,hidden_layers=(200,200,70,40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, batch_size=3000,optimizer=optimizer,num_iterations=10, num_particles=3)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, newsgroup_dataset, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=False)
    print(stein_vi.algorithm.random_forest.random_forest(dataset=newsgroup_dataset))





if __name__ == "__main__":
    run_20_newsgroups(info=True)
