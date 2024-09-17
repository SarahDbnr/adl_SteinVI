import jax
import jax.numpy as jnp
import tensorflow as tf
from optax import adam, exponential_decay

from stein_vi.stein_vi import train_with_stein_vi
from stein_vi.Classes.SteinVI_BNN import SteinVI_BNN
from run_stein_vi.model.BNN_Model import build_model

from stein_vi.metrics.validation_and_evaluation import (calculate_mse, calculate_accuracy)

from run_stein_vi.data.regression_toy_example import get_regression_toy_example
from run_stein_vi.data.data_handling import apply_data_settings_keras

from stein_vi.parameter_search.print_evaluation import (print_evaluation_regression_to_csv,
                                                        print_evaluation_multiclass_to_csv)


def parameter_loop_regression(dataset, model, name):
    key = jax.random.PRNGKey(1)
    z_train, _, _, _, z_test, y_test = dataset

    # default set up
    early_stopping = False
    batch_size = 0
    particle_batch_size = 0
    num_iterations = 100
    init_value = 0.1
    decay_rate = 0.95

    array_num_particles = [5, 10, 20, 30, 40, 50, 80, 100, 150, 200, 250]
    array_batch_size = [0, 5, 10, 20, 50]
    array_early_stopping = [False, True]
    array_init_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    array_decay_rate = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]

    for num_particles in array_num_particles:
        lowest_mse = jnp.inf
        final_num_particles = num_particles
        steinvi_svdg = initialize_steinvi(key, z_train, model, num_particles, batch_size, particle_batch_size,
                                          num_iterations, early_stopping, init_value, decay_rate,
                                          use_for_regression=True)
        mse = run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                                  decay_rate)
        if mse < lowest_mse:
            final_num_particles = num_particles
    array_particle_batch_size = [0, round(final_num_particles / 2), round(final_num_particles / 3),
                                 round(final_num_particles / 4), round(final_num_particles / 5)]
    for batch_size in array_batch_size:
        lowest_mse = jnp.inf
        final_batch_size = batch_size
        steinvi_svdg = initialize_steinvi(key, z_train, model, final_num_particles, batch_size, particle_batch_size,
                                          num_iterations, early_stopping, init_value, decay_rate,
                                          use_for_regression=True)
        mse = run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                                  decay_rate)
        if mse < lowest_mse:
            final_batch_size = batch_size
    for particle_batch_size in array_particle_batch_size:
        lowest_mse = jnp.inf
        final_particle_batch_size = particle_batch_size
        steinvi_svdg = initialize_steinvi(key, z_train, model, final_num_particles, final_batch_size,
                                          particle_batch_size,
                                          num_iterations, early_stopping, init_value, decay_rate,
                                          use_for_regression=True)
        mse = run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                                  decay_rate)
        if mse < lowest_mse:
            final_particle_batch_size = particle_batch_size
    for early_stopping in array_early_stopping:
        lowest_mse = jnp.inf
        final_early_stopping = early_stopping
        steinvi_svdg = initialize_steinvi(key, z_train, model, final_num_particles, final_batch_size,
                                          final_particle_batch_size,
                                          num_iterations, early_stopping, init_value, decay_rate,
                                          use_for_regression=True)
        mse = run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                                  decay_rate)
        if mse < lowest_mse:
            final_early_stopping = early_stopping
    for init_value in array_init_value:
        lowest_mse = jnp.inf
        final_init_value = init_value
        steinvi_svdg = initialize_steinvi(key, z_train, model, final_num_particles, final_batch_size,
                                          final_particle_batch_size,
                                          num_iterations, final_early_stopping, init_value, decay_rate,
                                          use_for_regression=True)
        mse = run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                                  decay_rate)
        if mse < lowest_mse:
            final_init_value = init_value
    for decay_rate in array_decay_rate:
        steinvi_svdg = initialize_steinvi(key, z_train, model, final_num_particles, final_batch_size,
                                          final_particle_batch_size,
                                          num_iterations, final_early_stopping, final_init_value, decay_rate,
                                          use_for_regression=True)
        run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                            decay_rate)


def parameter_loop_multiclass(dataset, model, name):
    key = jax.random.PRNGKey(1)
    z_train, _, _, _, z_test, y_test = dataset

    # default set up
    early_stopping = False
    batch_size = 0
    particle_batch_size = 0
    num_iterations = 100
    init_value = 0.1
    decay_rate = 0.95

    array_num_particles = [5, 10, 20, 30, 40, 50, 80, 100, 150, 200, 250]
    array_batch_size = [0, 5, 10, 20, 50]
    array_early_stopping = [False, True]
    array_init_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    array_decay_rate = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]

    for num_particles in array_num_particles:
        highest_accuracy = -jnp.inf
        final_num_particles = num_particles
        steinvi_svdg = initialize_steinvi(key, z_train, model, num_particles, batch_size, particle_batch_size,
                                          num_iterations, early_stopping, init_value, decay_rate,
                                          use_for_regression=False)
        accuracy = run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                                       decay_rate)
        if accuracy > highest_accuracy:
            final_num_particles = num_particles
    array_particle_batch_size = [0, round(final_num_particles / 2), round(final_num_particles / 3),
                                 round(final_num_particles / 4), round(final_num_particles / 5)]
    for batch_size in array_batch_size:
        highest_accuracy = -jnp.inf
        final_batch_size = batch_size
        steinvi_svdg = initialize_steinvi(key, z_train, model, final_num_particles, batch_size, particle_batch_size,
                                          num_iterations, early_stopping, init_value, decay_rate,
                                          use_for_regression=False)
        accuracy = run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                                       decay_rate)
        if accuracy > highest_accuracy:
            final_batch_size = batch_size
    for particle_batch_size in array_particle_batch_size:
        highest_accuracy = -jnp.inf
        final_particle_batch_size = particle_batch_size
        steinvi_svdg = initialize_steinvi(key, z_train, model, final_num_particles, final_batch_size,
                                          particle_batch_size,
                                          num_iterations, early_stopping, init_value, decay_rate,
                                          use_for_regression=False)
        accuracy = run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                                       decay_rate)
        if accuracy > highest_accuracy:
            final_particle_batch_size = particle_batch_size
    for early_stopping in array_early_stopping:
        highest_accuracy = -jnp.inf
        final_early_stopping = early_stopping
        steinvi_svdg = initialize_steinvi(key, z_train, model, final_num_particles, final_batch_size,
                                          final_particle_batch_size,
                                          num_iterations, early_stopping, init_value, decay_rate,
                                          use_for_regression=False)
        accuracy = run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                                       decay_rate)
        if accuracy > highest_accuracy:
            final_early_stopping = early_stopping
    for init_value in array_init_value:
        highest_accuracy = -jnp.inf
        final_init_value = init_value
        steinvi_svdg = initialize_steinvi(key, z_train, model, final_num_particles, final_batch_size,
                                          final_particle_batch_size,
                                          num_iterations, final_early_stopping, init_value, decay_rate,
                                          use_for_regression=False)
        accuracy = run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                                       decay_rate)
        if accuracy > highest_accuracy:
            final_init_value = init_value
    for decay_rate in array_decay_rate:
        steinvi_svdg = initialize_steinvi(key, z_train, model, final_num_particles, final_batch_size,
                                          final_particle_batch_size,
                                          num_iterations, final_early_stopping, final_init_value, decay_rate,
                                          use_for_regression=False)
        run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value,
                                            decay_rate)


def initialize_steinvi(key, z_train, model, num_particles, batch_size, particle_batch_size,
                       num_iterations, early_stopping, init_value, decay_rate, use_for_regression):
    optimizer = adam(
        exponential_decay(
            init_value=init_value,
            transition_steps=20,
            decay_rate=decay_rate,
            staircase=True
        )
    )
    steinvi_svdg = SteinVI_BNN(key, z_train, model, use_for_regression=use_for_regression, optimizer=optimizer,
                               num_particles=num_particles, batch_size=batch_size,
                               particle_batch_size=particle_batch_size, num_iterations=num_iterations,
                               early_stopping=early_stopping)

    return steinvi_svdg


def run_training_and_attach_information(steinvi_svdg, dataset, key, z_test, y_test, name, init_value, decay_rate):
    steinvi_svdg = train_with_stein_vi(steinvi_svdg, dataset, key, algorithm="svgd")

    test_predictions, test_precisions = steinvi_svdg.predict_over_particles(z_test)

    if steinvi_svdg.use_for_regression:
        print_evaluation_regression_to_csv(name, steinvi_svdg.parameter, y_test, test_predictions, test_precisions,
                                           init_value, decay_rate)
        return calculate_mse(test_predictions, y_test)
    else:
        print_evaluation_multiclass_to_csv(name, steinvi_svdg.parameter, y_test, test_predictions, init_value,
                                           decay_rate)
        return calculate_accuracy(test_predictions, y_test)


def run_MNIST():
    """
    Run SVGD on the MNIST dataset for classification.
    """
    mnist = tf.keras.datasets.mnist
    mnist_dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)

    nnet_model = build_model(output_size=10, hidden_layers=(200, 70, 40))

    parameter_loop(mnist_dataset, use_for_regression=False, model=nnet_model, name="MNIST")


def run_regression_toy_example():
    """
    Run SVGD on the MNIST dataset for classification.
    """
    regression_toy_example = get_regression_toy_example(num_points=10000)

    nnet_model = build_model(output_size=2, hidden_layers=(200, 70, 40))

    parameter_loop(regression_toy_example, use_for_regression=True, model=nnet_model, name="Regression_toy_example")


if __name__ == "__main__":
    run_regression_toy_example()
