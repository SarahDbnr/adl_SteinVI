import pytest
import jax
import tensorflow as tf
from optax import adam, exponential_decay

import jax.numpy as jnp
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from run_stein_vi.data.data_handling import apply_data_settings_keras
from run_stein_vi.data.regression_toy_example import get_regression_toy_example

from stein_vi.Classes.SteinVI_BNN import SteinVI_BNN
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

@pytest.fixture
def synthetic_classification_dataset():
    """Generates a synthetic dataset for classification using sklearn's make_classification."""
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return (jnp.array(X_train), jnp.array(y_train), jnp.array(X_val), jnp.array(y_val), jnp.array(X_test), jnp.array(y_test))


@pytest.fixture
def synthetic_regression_dataset():
    """Generates a synthetic dataset for regression using sklearn's make_regression."""
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return (jnp.array(X_train), jnp.array(y_train), jnp.array(X_val), jnp.array(y_val), jnp.array(X_test), jnp.array(y_test))
