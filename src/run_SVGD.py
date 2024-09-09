import jax
import tensorflow as tf
from optax import adam, exponential_decay

from src.data.regression_toy_example import get_regression_toy_example
from src.algorithm.svgd import train_with_svgd
from src.model.BNN_Model import build_model
from src.metrics.validation_and_evaluation import get_evaluation_metrics_over_predictions, print_summary_over_particles
from src.data.data_handling import apply_data_settings_sklearn, apply_data_settings_keras, newsgroup_datahandling, \
    adult_income_datahandling
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_iris
from src.metrics.plots_validation_metrics import plot_and_save_evaluation_metric, plot_residuals, plot_location_in_relation_to_scale
from src.Parameter_Class import Parameter
import src.data.datasets_info as datasets_info


def run_svgd_on_regression(dataset, parameter, output_size, network_structure):
    """Run the SVGD algorithm on a regression dataset.

    Args:
        dataset (Teat): Teat
        parameter (Teat): Teat
        output_size (Teat): Teat
        network_structure (Teat): Teat
    """    
    # for batch_size: default is 10 minibatches, 0 will induce no batching, else batch_size int will be used
    key = jax.random.PRNGKey(1)
    z_train, y_train, z_val, y_val, z_test, y_test = dataset
    nnet_model, tree_def, param_vec_ini = build_model(key, z_train, output_size=output_size,
                                                      hidden_layers=network_structure,
                                                      use_for_regression=parameter.use_for_regression)

    out, mse_val, averaged_precision_val = train_with_svgd(dataset, nnet_model, tree_def,
                                                           param_vec_ini, parameter, key)

    mse_test, averaged_precision_test, predictions_test = get_evaluation_metrics_over_predictions(out, nnet_model,
                                                                                                  tree_def,
                                                                                                  z_test,
                                                                                                  y_test,
                                                                                                  model_regression=True)
    print("For Test Data: MSE ", mse_test, " Averaged Precision ", averaged_precision_test)

    # plot_mse(averaged_precision)
    plot_and_save_evaluation_metric(evaluation_metric_val=mse_val, num_particles=parameter.num_particles,
                                    network_structure=network_structure, eval_metric="MSE")
    plot_and_save_evaluation_metric(evaluation_metric_val=averaged_precision_val, num_particles=parameter.num_particles,
                                    network_structure=network_structure, eval_metric="averaged_precision")
    # Call the plot_residuals function
    plot_residuals(nnet_model, tree_def, out, z_test, y_test, num_particles=parameter.num_particles,
                   network_structure=network_structure)
    plot_location_in_relation_to_scale(nnet_model, tree_def, out, z_test, num_particles=parameter.num_particles,
                   network_structure=network_structure)
    print_summary_over_particles(predictions_test)


def run_svgd_on_multiclass_data(dataset, parameter, output_size, network_structure):
    # for batch_size: default is 10 minibatches, 0 will induce no batching, else batch_size int will be used
    key = jax.random.PRNGKey(1)
    z_train, y_train, z_val, y_val, z_test, y_test = dataset
    nnet_model, tree_def, param_vec = build_model(key, z_train, output_size=output_size,
                                                  hidden_layers=network_structure,
                                                  use_for_regression=parameter.use_for_regression)

    out, accuracy_val, _ = train_with_svgd(dataset, nnet_model, tree_def, param_vec, parameter, key)

    accuracy_test, _, predictions_test = get_evaluation_metrics_over_predictions(out, nnet_model, tree_def, z_test,
                                                                                 y_test,
                                                                                 model_regression=False)
    print("For Test Data: Accuracy ", accuracy_test)
    plot_and_save_evaluation_metric(evaluation_metric_val=accuracy_val, num_particles=parameter.num_particles,
                                    network_structure=network_structure, eval_metric="Accuracy")
    print_summary_over_particles(predictions_test)


def run_MNIST(info=False):
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

    parameter = Parameter(optimizer, regression=False)
    run_svgd_on_multiclass_data(dataset, parameter=parameter, network_structure=(200, 75, 40), output_size=10)


def run_MNIST_minibatched_particles(info=False):
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

    parameter = Parameter(optimizer, batch_size=300, particle_batch_size=2, num_particles = 4, regression=False)
    run_svgd_on_multiclass_data(dataset, parameter=parameter, network_structure=(200, 75, 40), output_size=10)


def run_FashionMNIST(info=False):
    if info:
        datasets_info.print_fashion_mnist_dataset_info()

    fashion_mnist = tf.keras.datasets.fashion_mnist
    dataset = apply_data_settings_keras(fashion_mnist.load_data(), with_flattening=False)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    parameter = Parameter(optimizer, regression=False)
    run_svgd_on_multiclass_data(dataset, parameter=parameter, network_structure=(200, 75, 40), output_size=10)


def run_CIFAR10(info=True):  
    if info:
        datasets_info.print_cifar10_dataset_info()
    cifar10 = tf.keras.datasets.cifar10
    dataset = apply_data_settings_keras(cifar10.load_data(), with_flattening=True)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    parameter = Parameter(optimizer, regression=False)
    run_svgd_on_multiclass_data(dataset, parameter=parameter, network_structure=(200, 75, 40), output_size=10)


def run_20_newsgroups(info=True):
    if info:
        datasets_info.print_20_newsgroups_dataset_info()
    dataset = newsgroup_datahandling()
    dataset = apply_data_settings_sklearn(dataset)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    parameter = Parameter(optimizer, regression=False, batch_size=100)
    run_svgd_on_multiclass_data(dataset, parameter=parameter, network_structure=(200, 200, 75, 40), output_size=20)


def run_adult_income(info=False):
    if info:
        datasets_info.print_adult_income_dataset_info()
    dataset = adult_income_datahandling()
    dataset = apply_data_settings_sklearn(dataset)

    optimizer = adam(
        exponential_decay(
            init_value=0.001,
            transition_steps=1000,
            decay_rate=0.95,
            staircase=True  
        )
    )

    parameter = Parameter(optimizer, regression=False, batch_size=2000)
    run_svgd_on_multiclass_data(dataset, parameter=parameter, network_structure=(200, 75, 40), output_size=2)



def run_iris(info=False):
    if info:
        datasets_info.print_iris_dataset_info()
    iris = load_iris()  # Loading the Iris dataset
    dataset = apply_data_settings_sklearn(iris)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    parameter = Parameter(optimizer, regression=False)
    run_svgd_on_multiclass_data(dataset, parameter=parameter, network_structure=(200, 75, 40), output_size=3)


def run_regression_toy_example(info=False):
    regression_toy_example = get_regression_toy_example(num_points=10000)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    parameter = Parameter(optimizer, num_iterations=250, regression=True)
    parameter.set_early_stopping(10000, 1000, 3)
    run_svgd_on_regression(regression_toy_example, parameter=parameter, network_structure=(200, 75, 40),
                                output_size=2)


def run_california_housing(info=False):
    if info:
        datasets_info.print_california_housing_dataset_info()
    california_housing = fetch_california_housing()
    dataset = apply_data_settings_sklearn(california_housing)

    optimizer = adam(
        exponential_decay(
            init_value=0.075,
            transition_steps=150,
            decay_rate=0.975,
            staircase=True
        )
    )

    #optimizer = adam(
    #    exponential_decay(
    #        init_value=0.025,
    #        transition_steps=150,
    #        decay_rate=0.995,
    #        staircase=True
    #    )
    #)

    parameter = Parameter(optimizer, regression=True, batch_size=1000, num_iterations=10000)
    parameter.set_early_stopping(10000, 1000, 3)

    run_svgd_on_regression(dataset, parameter=parameter, network_structure=(200, 75, 40),
                                output_size=2)


def run_diabetes(info=False): 
    if info:
        datasets_info.print_diabetes_dataset_info()
    diabetes = load_diabetes()
    dataset = apply_data_settings_sklearn(diabetes)

    optimizer = adam(
        exponential_decay(
            init_value=0.01,
            transition_steps=50,
            decay_rate=0.95,
            staircase=True
        )
    )

    parameter = Parameter(optimizer, regression=True ,num_iterations=1000)
    parameter.set_early_stopping(10000, 1000, 3)
    run_svgd_on_regression(dataset, parameter=parameter, network_structure=(200, 75, 40),
                                output_size=2)


def run_wine_quality(info=False):
    if info:
        datasets_info.print_wine_quality_dataset_info()
    wine_quality = load_wine()
    dataset = apply_data_settings_sklearn(wine_quality)

    optimizer = adam(
        exponential_decay(
            init_value=0.005,
            transition_steps=10,
            decay_rate=0.95,
            staircase=True
        )
    )

    parameter = Parameter(optimizer, regression=False)
    run_svgd_on_multiclass_data(dataset, parameter=parameter, network_structure=(200, 75, 40),
                                output_size=3)


if __name__ == "__main__":
    run_diabetes(info=True)
