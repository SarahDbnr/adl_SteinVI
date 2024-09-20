import jax
import tensorflow as tf

from fixtures import stein_vi_multiclass_example, stein_vi_regression_example
from run_stein_vi.data.data_handling import apply_data_settings_keras
from run_stein_vi.data.regression_toy_example import get_regression_toy_example

from stein_vi.algorithm.svgd import set_up_svgd


def test_predict_shape_multiclass(stein_vi_multiclass_example):
    # given
    mnist = tf.keras.datasets.mnist
    mnist_dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)
    z_train, y_train, _, _, _, _ = mnist_dataset
    set_up_svgd(stein_vi_multiclass_example)

    number_of_classes = 10

    nnet = stein_vi_multiclass_example.nnet
    particles = stein_vi_multiclass_example.state.particles

    # when
    predictions, precisions = jax.vmap(lambda p: nnet.predict(p, z_train))(particles)

    # then
    assert predictions.shape == (stein_vi_multiclass_example.parameter.num_particles, len(y_train))
    assert precisions.shape == (stein_vi_multiclass_example.parameter.num_particles, len(y_train), number_of_classes)


def test_predict_shape_regression(stein_vi_regression_example):
    # given
    regression_toy_example = get_regression_toy_example(num_points=10000)
    z_train, y_train, _, _, _, _ = regression_toy_example
    set_up_svgd(stein_vi_regression_example)

    nnet = stein_vi_regression_example.nnet
    particles = stein_vi_regression_example.state.particles

    # when
    predictions, precisions = jax.vmap(lambda p: nnet.predict(p, z_train))(particles)

    # then
    assert predictions.squeeze().shape == (stein_vi_regression_example.parameter.num_particles, len(y_train))
    assert precisions.squeeze().shape == (stein_vi_regression_example.parameter.num_particles, len(y_train))
