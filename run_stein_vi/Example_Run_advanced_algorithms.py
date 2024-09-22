import jax
import tensorflow as tf
from optax import adam, sgd, exponential_decay
import optax
from stein_vi.Classes.SteinVI_BNN_Class import SteinVI_BNN
from run_stein_vi.model.BNN_Model import build_model
from stein_vi.stein_vi import train_with_stein_vi
import data.datasets_info as datasets_info
from data.data_handling import apply_data_settings_keras


def run_MNIST_GD(info=False):
    """
    Run SVGD on the MNIST dataset for classification. With just standard gradient decent
    from the optax optimizer package.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """

    key = jax.random.PRNGKey(1)

    if info:
        datasets_info.print_mnist_dataset_info()

    mnist = tf.keras.datasets.mnist
    mnist_dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)
    z_train, _, _, _, z_test, y_test = mnist_dataset

    optimizer = sgd(0.0001)

    nnet_model = build_model(output_size=10, hidden_layers=(200, 70, 40))

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, image_data=True, mode_training_print="full",
                               use_for_regression=False, batch_size=0, particle_batch_size=0, num_iterations=30,
                               num_particles=5, optimizer=optimizer, kernel_length=5)

    train_with_stein_vi(steinvi_svdg, mnist_dataset, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()
    steinvi_svdg.view_misclassified(z_test, y_test, key=key)


def run_MNIST_ssvgd_ADAM(info=False):
    """
    Run sSVGD on the MNIST dataset for classification. 

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """

    key = jax.random.PRNGKey(1)

    if info:
        datasets_info.print_mnist_dataset_info()

    mnist = tf.keras.datasets.mnist
    mnist_dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)
    z_train, _, _, _, z_test, y_test = mnist_dataset

    nnet_model = build_model(output_size=10, hidden_layers=(200, 70, 40))

    optimizer = optax.inject_hyperparams(adam)(exponential_decay(init_value=0.01,
                                                                 transition_steps=50,
                                                                 decay_rate=0.95,
                                                                 staircase=True
                                                                 ))

    steinvi_ssvdg = SteinVI_BNN(key, z_train, nnet_model, image_data=True, mode_training_print="full",
                                use_for_regression=False, batch_size=300, particle_batch_size=0, num_iterations=30,
                                num_particles=5, optimizer=optimizer)

    train_with_stein_vi(steinvi_ssvdg, mnist_dataset, key, algorithm="ssvgd")

    steinvi_ssvdg.plot_val_metric_over_iter()
    steinvi_ssvdg.view_misclassified(z_test, y_test, key=key)


def run_MNIST_ssvgd_GD(info=False):
    """
    Run sSVGD on the MNIST dataset for classification. 

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """

    key = jax.random.PRNGKey(1)

    if info:
        datasets_info.print_mnist_dataset_info()

    mnist = tf.keras.datasets.mnist
    mnist_dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)
    z_train, _, _, _, z_test, y_test = mnist_dataset

    nnet_model = build_model(output_size=10, hidden_layers=(200, 70, 40))

    optimizer = optax.inject_hyperparams(optax.sgd)(exponential_decay(
        init_value=0.01,
        transition_steps=50,
        decay_rate=0.95,
        staircase=True
    ))

    steinvi_ssvdg = SteinVI_BNN(key, z_train, nnet_model, image_data=True, mode_training_print="full",
                                use_for_regression=False, batch_size=0, particle_batch_size=5, num_iterations=30,
                                num_particles=5, optimizer=optimizer)

    train_with_stein_vi(steinvi_ssvdg, mnist_dataset, key, algorithm="ssvgd")

    steinvi_ssvdg.plot_val_metric_over_iter()
    steinvi_ssvdg.view_misclassified(z_test, y_test, key=key)


if __name__ == "__main__":
    run_MNIST_ssvgd_GD(info=False)
