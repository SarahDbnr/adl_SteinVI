import jax
import tensorflow as tf

from fixtures import stein_vi_multiclass_example
from stein_vi.algorithm.svgd import set_up_svgd
from run_stein_vi.data.data_handling import apply_data_settings_keras


def test_predict_shape(stein_vi_multiclass_example):
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
