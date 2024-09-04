import jax
import tensorflow as tf
from optax import adam, exponential_decay

from BNN_Example_clean_version.Parameter_Class import Parameter
from BNN_Example_clean_version.BNN_Model import build_model
from BNN_Example_clean_version import datasets_info
from BNN_Example_clean_version.regression_toy_example import get_regression_toy_example
from BNN_Example_clean_version.svgd import train_with_svgd
from BNN_Example_clean_version.validation_and_evaluation import get_evaluation_metrics_over_predictions
from BNN_Example_clean_version.data_handling import apply_data_settings_sklearn, apply_data_settings_keras, \
    newsgroup_datahandling, adult_income_datahandling
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_iris
from BNN_Example_clean_version.plots_validation_metrics import plot_and_save_evaluation_metric
from Evalutation_Class import EvaluationRegression, EvaluationMulticlass


def run_svgd_on_regression(dataset, parameter, output_size, network_structure, name):
    # for batch_size: default is 10 minibatches, 0 will induce no batching, else batch_size int will be used
    key = jax.random.PRNGKey(1)
    z_train, y_train, z_val, y_val, z_test, y_test = dataset
    nnet_model, tree_def, param_vec_ini = build_model(key, z_train, output_size=output_size,
                                                      hidden_layers=network_structure,
                                                      use_for_regression=parameter.use_for_regression)

    out, mse_val, averaged_precision_val = train_with_svgd(dataset, nnet_model, tree_def,
                                                           param_vec_ini, parameter, key)

    test_predictions, test_precisions = jax.vmap(lambda p: nnet_model.predict(tree_def(p), z_test))(out.particles)

    evaluation = EvaluationRegression(name, y_test, test_predictions, test_precisions)

    print(evaluation)


def run_svgd_on_multiclass_data(dataset, parameter, output_size, network_structure, name):
    # for batch_size: default is 10 minibatches, 0 will induce no batching, else batch_size int will be used
    key = jax.random.PRNGKey(1)
    z_train, y_train, z_val, y_val, z_test, y_test = dataset
    nnet_model, tree_def, param_vec = build_model(key, z_train, output_size=output_size,
                                                  hidden_layers=network_structure,
                                                  use_for_regression=parameter.use_for_regression)

    out, accuracy_val, _ = train_with_svgd(dataset, nnet_model, tree_def, param_vec, parameter, key)

    accuracy_test, _ = get_evaluation_metrics_over_predictions(out, nnet_model, tree_def, z_test, y_test,
                                                               model_regression=False)
    print(f"\nTest accuracy: {accuracy_test}")
    plot_and_save_evaluation_metric(evaluation_metric_val=accuracy_val, num_particles=parameter.num_particles,
                                    network_structure=network_structure, eval_metric="Accuracy")


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
    run_svgd_on_multiclass_data(dataset, parameter=parameter, network_structure=(200, 75, 40), output_size=10,
                                name="MNIST")


def run_regression_toy_example():
    regression_toy_example = get_regression_toy_example(num_points=100)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.75,
            staircase=True
        )
    )

    parameter = Parameter(optimizer, regression=True)
    run_svgd_on_regression(dataset=regression_toy_example, parameter=parameter, network_structure=(200, 75, 40),
                           output_size=2, name="Regression Toy Example")


if __name__ == "__main__":
    run_regression_toy_example()
