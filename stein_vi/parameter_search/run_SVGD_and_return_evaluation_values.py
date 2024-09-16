import jax
import tensorflow as tf
from optax import adam, exponential_decay

from stein_vi.Classes.Parameter_Class import Parameter
from run_stein_vi.model.BNN_Model import build_model
from run_stein_vi.data import datasets_info
from run_stein_vi.data.regression_toy_example import get_regression_toy_example
from stein_vi.algorithm.svgd import train_with_svgd
from run_stein_vi.data.data_handling import apply_data_settings_keras
from stein_vi.parameter_search.print_evaluation import print_evaluation_regression_to_csv, print_evaluation_multiclass_to_csv


def run_svgd_on_regression(dataset, parameter, output_size, network_structure, name):
    """
    Test the SVGD algorithm on a regression dataset.

    Args:
        dataset (tuple): The dataset tuple containing training, validation, and test sets.
        parameter (Parameter): Object containing parameters for the SVGD algorithm.
        output_size (int): The size of the output layer for the regression task.
        network_structure (tuple): Tuple specifying the number of units in each hidden layer.
        name (str): A string used to name the output CSV file for evaluation metrics.
    """
    # for batch_size: default is 10 minibatches, 0 will induce no batching, else batch_size int will be used
    key = jax.random.PRNGKey(1)
    z_train, y_train, z_val, y_val, z_test, y_test = dataset
    nnet_model, tree_def, param_vec_ini = build_model(key, z_train)

    out, mse_val, averaged_precision_val = train_with_svgd(dataset, nnet_model, tree_def,
                                                           param_vec_ini, parameter, key)

    test_predictions, test_precisions = jax.vmap(lambda p: nnet_model.predict(tree_def(p), z_test))(out.particles)

    print_evaluation_regression_to_csv(name, parameter, y_test, test_predictions, test_precisions)


def run_svgd_on_multiclass_data(dataset, parameter, output_size, network_structure, name):
    """
    Test the SVGD algorithm on a multiclass classification dataset.

    Args:
        dataset (tuple): The dataset tuple containing training, validation, and test sets.
        parameter (Parameter): Object containing parameters for the SVGD algorithm.
        output_size (int): The size of the output layer for the classification task.
        network_structure (tuple): Tuple specifying the number of units in each hidden layer.
        name (str): A string used to name the output CSV file for evaluation metrics.
    """    
    key = jax.random.PRNGKey(1)
    z_train, y_train, z_val, y_val, z_test, y_test = dataset
    nnet_model, tree_def, param_vec_ini = build_model(key, z_train, output_size=output_size,
                                                      hidden_layers=network_structure,
                                                      use_for_regression=parameter.use_for_regression)

    out, accuracy_val, _ = train_with_svgd(dataset, nnet_model, tree_def, param_vec_ini, parameter, key)

    test_predictions, test_precisions = jax.vmap(lambda p: nnet_model.predict(tree_def(p), z_test))(out.particles)

    print_evaluation_multiclass_to_csv(name, parameter, y_test, test_predictions)


def run_MNIST(info=False):
    """
    Run SVGD on the MNIST dataset using different numbers of particles.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """
    if info:
        datasets_info.print_mnist_dataset_info()
    mnist = tf.keras.datasets.mnist
    dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    array_num_particles = [5, 10, 20, 30, 40, 50, 80, 100, 150, 200, 250]
    for num_particles in array_num_particles:
        parameter = Parameter(optimizer, regression=False, num_particles=num_particles, num_iterations=200)
        run_svgd_on_multiclass_data(dataset, parameter=parameter, network_structure=(200, 75, 40), output_size=10,
                                    name="MNIST")


def run_regression_toy_example():
    """
    Run SVGD on a synthetic regression dataset with varying numbers of particles.
    """
    regression_toy_example = get_regression_toy_example(num_points=100)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.75,
            staircase=True
        )
    )

    array_num_particles = [5, 10, 20, 30, 40, 50, 80, 100, 150, 200, 250]
    for num_particles in array_num_particles:
        parameter = Parameter(optimizer, regression=True, num_particles=num_particles)
        run_svgd_on_regression(dataset=regression_toy_example, parameter=parameter, network_structure=(200, 75, 40),
                               output_size=2, name="Regression Toy Example")


if __name__ == "__main__":
    run_MNIST()
