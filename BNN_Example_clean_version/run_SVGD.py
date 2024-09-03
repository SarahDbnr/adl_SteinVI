import jax
import tensorflow as tf

from optax import adam, exponential_decay
from regression_toy_example import get_regression_toy_example
from svgd import train_with_svgd
from validation_and_evaluation import get_evaluation_metrics_over_predictions, print_summary_over_particles
from data_handling import apply_data_settings_sklearn, apply_data_settings_keras
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine
from plots_validation_metrics import plot_and_save_evaluation_metric


def run_svgd_on_regression(dataset, optimizer, network_structure=(200, 75, 40), output_size=2, num_particles=100,
                           batch_size=20):
    # for batch_size: default is 10 minibatches, 0 will induce no batching, else batch_size int will be used
    key = jax.random.PRNGKey(1)

    out, z_test, y_test, nnet_model, tree_def, mse_val, averaged_precision_val = train_with_svgd(dataset, output_size,
                                                                                                 network_structure,
                                                                                                 batch_size,
                                                                                                 num_particles,
                                                                                                 key, regression=True,
                                                                                                 optimizer=optimizer)

    print("For Test Data:")
    mse_test, averaged_precision_test, predictions_test = get_evaluation_metrics_over_predictions(out, nnet_model,
                                                                                                  tree_def,
                                                                                                  z_test,
                                                                                                  y_test,
                                                                                                  model_regression=True)
    print(f"\nAveraged precision: {averaged_precision_test}, mean squared error: {mse_test}")
    # plot_mse(averaged_precision)
    plot_and_save_evaluation_metric(evaluation_metric_val=mse_val, num_particles=num_particles,
                                    network_structure=network_structure, eval_metric="MSE")
    plot_and_save_evaluation_metric(evaluation_metric_val=averaged_precision_val, num_particles=num_particles,
                                    network_structure=network_structure, eval_metric="averaged_precision")
    print_summary_over_particles(predictions_test)


def run_svgd_on_multiclass_data(dataset, optimizer, network_structure=(200, 75, 40), output_size=10, num_particles=2,
                                batch_size=300):
    # for batch_size: default is 10 minibatches, 0 will induce no batching, else batch_size int will be used
    key = jax.random.PRNGKey(1)

    out, z_test, y_test, nnet_model, tree_def, accuracy_val, _ = train_with_svgd(dataset, output_size,
                                                                                 network_structure, batch_size,
                                                                                 num_particles,
                                                                                 key, optimizer=optimizer,
                                                                                 regression=False)

    print("For Test Data:")
    accuracy_test, _, predictions_test = get_evaluation_metrics_over_predictions(out, nnet_model, tree_def, z_test,
                                                                                 y_test,
                                                                                 model_regression=False)
    print(f"\nTest accuracy: {accuracy_test}")
    plot_and_save_evaluation_metric(evaluation_metric_val=accuracy_val, num_particles=num_particles,
                                    network_structure=network_structure, eval_metric="Accuracy")
    print_summary_over_particles(predictions_test)


def run_MNIST():
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

    run_svgd_on_multiclass_data(dataset, optimizer, network_structure=(200, 75, 40), output_size=10, num_particles=2)


def run_FashionMNIST():
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

    run_svgd_on_multiclass_data(dataset, optimizer, network_structure=(200, 75, 40), output_size=10, num_particles=2,
                                batch_size=300)


def run_CIFAR10():
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

    run_svgd_on_multiclass_data(dataset, optimizer, network_structure=(200, 75, 40), output_size=10, num_particles=2,
                                batch_size=3000)


def run_regression_toy_example():
    regression_toy_example = get_regression_toy_example(num_points=100)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    run_svgd_on_regression(regression_toy_example, optimizer, network_structure=(200, 75, 40), output_size=2,
                           num_particles=100)


def run_california_housing():  # TODO: Analyse, dosnt work properly for all stages depending on batch size
    california_housing = fetch_california_housing()
    dataset = apply_data_settings_sklearn(california_housing)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    run_svgd_on_regression(dataset, optimizer, network_structure=(200, 75, 40), output_size=2, num_particles=2,
                           batch_size=2000)


def run_diabetes():
    california_housing = load_diabetes()
    dataset = apply_data_settings_sklearn(california_housing)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    run_svgd_on_regression(dataset, optimizer, network_structure=(200, 75, 40), output_size=2, num_particles=100)


def run_wine_quality():
    california_housing = load_wine()
    dataset = apply_data_settings_sklearn(california_housing)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    run_svgd_on_regression(dataset, optimizer, network_structure=(200, 75, 40), output_size=2, num_particles=100)


if __name__ == "__main__":
    run_wine_quality()
