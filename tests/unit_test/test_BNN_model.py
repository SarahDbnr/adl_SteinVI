import jax

from fixtures import stein_vi_multiclass_example, stein_vi_regression_example, get_regression_toy_example, get_MNIST

from stein_vi.algorithm.svgd import set_up_svgd


def test_predict_shape_multiclass(stein_vi_multiclass_example, get_MNIST):
    # given
    mnist_dataset = get_MNIST
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


def test_predict_shape_regression(stein_vi_regression_example, get_regression_toy_example):
    # given
    z_train, y_train, _, _, _, _ = get_regression_toy_example
    set_up_svgd(stein_vi_regression_example)

    nnet = stein_vi_regression_example.nnet
    particles = stein_vi_regression_example.state.particles

    # when
    predictions, precisions = jax.vmap(lambda p: nnet.predict(p, z_train))(particles)

    # then
    assert predictions.squeeze().shape == (stein_vi_regression_example.parameter.num_particles, len(y_train))
    assert precisions.squeeze().shape == (stein_vi_regression_example.parameter.num_particles, len(y_train))
