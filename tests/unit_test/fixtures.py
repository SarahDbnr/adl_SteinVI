import pytest
import jax
import tensorflow as tf
from optax import adam, exponential_decay

from run_stein_vi.data.data_handling import apply_data_settings_keras
from run_stein_vi.data.regression_toy_example import get_regression_toy_example

from stein_vi.Classes.SteinVI_BNN_Class import SteinVI_BNN
from run_stein_vi.model.BNN_Model import build_model


@pytest.fixture
def stein_vi_regression_example():
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

    steinvi_regression = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=True, optimizer=optimizer)

    return steinvi_regression


@pytest.fixture
def stein_vi_multiclass_example():
    key = jax.random.PRNGKey(1)

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

    stein_vi_multiclass = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=False, optimizer=optimizer,
                                      batch_size=300, num_iterations=30, num_particles=5)

    return stein_vi_multiclass
