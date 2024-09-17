from optax import adam, exponential_decay, sgd,lbfgs
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
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_iris

def run_MNIST_GD(info=False):
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

    optimizer = sgd(0.001
    )
    nnet_model = build_model(output_size=10,hidden_layers=(200,70,40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, optimizer=optimizer, batch_size=300,num_iterations=30, num_particles=5)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, mnist_dataset, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=True)

def run_MNIST_plain_svgd(info=False):
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

    nnet_model = build_model(output_size=10,hidden_layers=(200,70,40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model,mode_training_print="full", use_for_regression=False, batch_size=0, particle_batch_size=3,num_iterations=3, num_particles=9, learning_rate=0.001)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, mnist_dataset, key, algorithm="plain_svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=True)

#TODO:Problem everything in the noise matrix gets 0
def run_MNIST_ssvgd(info=False):
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

    nnet_model = build_model(output_size=10,hidden_layers=(200,70,40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model,mode_training_print="full", use_for_regression=False, batch_size=0, particle_batch_size=3,num_iterations=3, num_particles=9, learning_rate=0.001)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, mnist_dataset, key, algorithm="ssvgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=True)
#TODO: fix paricel batching
def run_MNIST_quasi_SVN(info=False):
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

    optimizer = lbfgs(
            learning_rate=0.5,  # Increase learning rate to speed up convergence
            memory_size=100,  # Memory size to store previous gradients
            scale_init_precond=False,  # Apply scaling to initial preconditioning
        )
    nnet_model = build_model(output_size=10,hidden_layers=(200,10))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, optimizer=optimizer, batch_size=30000,num_iterations=30, num_particles=9,mode_training_print="full", particle_batch_size=3)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, mnist_dataset, key, algorithm="quasi_svn")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, image_data=True)
if __name__ == "__main__":
    run_MNIST_quasi_SVN()
