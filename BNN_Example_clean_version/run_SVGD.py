import jax
import tensorflow as tf
from optax import adam, exponential_decay
from regression_toy_example import get_regression_toy_example
from svgd import train_with_svgd
from validation_and_evaluation import get_evaluation_metrics_over_predictions
from data_handling import apply_data_settings_sklearn, apply_data_settings_keras,newsgroup_datahandling,adult_income_datahandling
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_iris
from plots_validation_metrics import plot_and_save_evaluation_metric
import datasets_info



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
    mse_test, averaged_precision_test = get_evaluation_metrics_over_predictions(out, nnet_model, tree_def, z_test,
                                                                                y_test, model_regression=True)
    print(f"\nAveraged precision: {averaged_precision_test}, mean squared error: {mse_test}")
    # plot_mse(averaged_precision)
    plot_and_save_evaluation_metric(evaluation_metric_val=mse_val, num_particles=num_particles,
                                    network_structure=network_structure, eval_metric="MSE")
    plot_and_save_evaluation_metric(evaluation_metric_val=averaged_precision_val, num_particles=num_particles,
                                    network_structure=network_structure, eval_metric="averaged_precision")
    # TODO plot_mse(mse)


def run_svgd_on_multiclass_data(dataset, optimizer, network_structure=(200, 75, 40), output_size=10, num_particles=2,
                                batch_size=300):
    # for batch_size: default is 10 minibatches, 0 will induce no batching, else batch_size int will be used
    key = jax.random.PRNGKey(1)

    out, z_test, y_test, nnet_model, tree_def, accuracy_val, _ = train_with_svgd(dataset, output_size,
                                                                                 network_structure, batch_size,
                                                                                 num_particles,
                                                                                 key, optimizer=optimizer,
                                                                                 regression=False)

    accuracy_test, _ = get_evaluation_metrics_over_predictions(out, nnet_model, tree_def, z_test, y_test,
                                                               model_regression=False)
    print(f"\nTest accuracy: {accuracy_test}")
    plot_and_save_evaluation_metric(evaluation_metric_val=accuracy_val, num_particles=num_particles,
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

    run_svgd_on_multiclass_data(dataset, optimizer, network_structure=(200, 75, 40), output_size=10, num_particles=2)


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

    run_svgd_on_multiclass_data(dataset, optimizer, network_structure=(200, 75, 40), output_size=10, num_particles=2,
                                batch_size=300)


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

    run_svgd_on_multiclass_data(dataset, optimizer, network_structure=(200, 75, 40), output_size=10, num_particles=2,
                                batch_size=3000)



def run_20_newsgroups(info=True):
    if info:
        datasets_info.print_20_newsgroups_dataset_info()
    dataset = newsgroup_datahandling()
    dataset = apply_data_settings_sklearn(dataset)

    optimizer = adam(
        exponential_decay(
            init_value=0.001,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    run_svgd_on_multiclass_data(dataset, optimizer, network_structure=(500, 300, 100,50), output_size=20, num_particles=12)



def run_adult_income(info=False):
    if info:
        datasets_info.print_adult_income_dataset_info
    dataset = adult_income_datahandling()
    dataset = apply_data_settings_sklearn(dataset)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    run_svgd_on_multiclass_data(dataset, optimizer, network_structure=(200, 75, 40), output_size=2, num_particles=2)

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

    run_svgd_on_multiclass_data(dataset, optimizer, network_structure=(20, 15, 7), output_size=3, num_particles=10, batch_size=25)


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


def run_california_housing(info= False):  # TODO: Analyse, dosnt work properly for all stages depending on batch size
    if info:
        datasets_info.print_california_housing_dataset_info()
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


def run_diabetes(info=False):
    if info:
        datasets_info.print_diabetes_dataset_info()
    diabetes = load_diabetes()
    dataset = apply_data_settings_sklearn(diabetes)

    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )

    run_svgd_on_regression(dataset, optimizer, network_structure=(200, 75, 40), output_size=2, num_particles=100)


def run_wine_quality(info=False):
    if info:
        datasets_info.print_wine_quality_dataset_info()
    wine_quality = load_wine()
    dataset = apply_data_settings_sklearn(wine_quality)

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
    run_wine_quality(info=True)
