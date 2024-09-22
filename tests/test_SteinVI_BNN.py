import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
from optax import adam

from stein_vi.Classes.Parameter_Class import Parameter
from stein_vi.Classes.Handler_Class import Handler
from stein_vi.algorithm.get_posteriori import logp_unnormalized_posterior_regression
from stein_vi.Classes.SteinVI_BNN_Class import SteinVI_BNN
from run_stein_vi.model.BNN_Model import build_model
from fixtures import stein_vi_multiclass_example, stein_vi_regression_example, get_regression_toy_example, get_MNIST
from stein_vi.algorithm.svgd import set_up_svgd


def test_steinvi_bnn_init_success():
    # given
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (100, 10))
    y_train = jax.random.normal(key, (100,))

    num_particles = 10
    nnet = build_model(output_size=2, hidden_layers=(2, 4))
    num_weights = 10 * 2 + 2 + 2 * 4 + 4 + 4 * 2 + 2

    # when
    model = SteinVI_BNN(
        key=key,
        x_train=x_train,
        nnet=nnet,
        use_for_regression=True,
        optimizer=adam(0.001),
        mode_training_print='full',
        mode_evaluation='minimal',
        early_stopping=True,
        image_data=False,
        batch_size=50,
        particle_batch_size=5,
        num_particles=num_particles,
        num_iterations=100,
        rf_comparison=False,
    )

    # then
    assert isinstance(model.handler, Handler), "Handler should be initialized correctly"
    assert model.handler._full_training_print, "Training print mode should be set correctly to full"
    assert model.handler._full_evaluation == False, "Evaluation mode should be set correctly"

    assert isinstance(model.parameter, Parameter), "Parameter should be initialized correctly"
    assert model.parameter.batch_size == 50, "Batch size should be set correctly"
    assert model.parameter.particle_batch_size == 5, "Particle batch size should be set correctly"
    assert model.parameter.num_particles == num_particles, "Number of particles should be set correctly"
    assert model.parameter.num_iterations == 100, "Number of iterations should be set correctly"
    assert model.parameter.early_stopping, "Early stopping should be set correctly"

    assert model.use_for_regression, "Use for regression flag should be set correctly"
    assert isinstance(model.nnet, nn.Module), "Neural network model should be initialized correctly"

    assert model.nnet.predict == model.predict
    assert model.initial_particle_vector.shape == (
    num_particles, num_weights), "Initial particle vector should have the correct shape"

    assert jnp.all(model.log_posteriori(weights=model.initial_particle_vector[0], dz=x_train, dy=y_train)
                   == logp_unnormalized_posterior_regression(weights=model.initial_particle_vector[0], dz=x_train,
                                                             dy=y_train,
                                                             nnet_model=nnet)), "Log posteriori function should be set correctly"


def test_steinvi_bnn_init_fail_batch_size():
    # given
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (10, 10))

    nnet = build_model(output_size=2, hidden_layers=(2, 4))

    # when then
    with pytest.raises(ValueError, match="Error: batch_size bigger then input data length"):
        SteinVI_BNN(
            key=key,
            x_train=x_train,
            nnet=nnet,
            use_for_regression=True,
            optimizer=adam(0.001),
            batch_size=20,
            particle_batch_size=5,
            num_particles=10,
            num_iterations=100
        )


def test_steinvi_bnn_init_fail_particle_batch_size():
    # given
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (100, 10))

    nnet = build_model(output_size=2, hidden_layers=(2, 4))

    # when then
    with pytest.raises(ValueError, match="Error: particle_batch_size bigger then number of particles"):
        SteinVI_BNN(
            key=key,
            x_train=x_train,
            nnet=nnet,
            use_for_regression=True,
            optimizer=adam(0.001),
            batch_size=50,
            particle_batch_size=15,
            num_particles=10,
            num_iterations=100
        )


def test_initial_particles_and_kernel():
    # given
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (100, 10)) 

    nnet = build_model(output_size=2, hidden_layers=(2, 4))

    # when
    model = SteinVI_BNN(
        key=key,
        x_train=x_train,
        nnet=nnet,
        use_for_regression=True,
        optimizer=adam(0.001),
        num_particles=10,
        num_iterations=100
    )

    # then
    particle_mean = jnp.mean(model.initial_particle_vector)
    particle_std = jnp.std(model.initial_particle_vector)

    assert jnp.isclose(particle_mean, 0, atol=0.1), f"Particle mean should be close to 0, but got {particle_mean}"
    assert jnp.isclose(particle_std, 1, atol=0.1), f"Particle std should be close to 1, but got {particle_std}"


def test_predict_multiclass(stein_vi_multiclass_example, get_MNIST):
    # given
    z_train, y_train, _, _, _, _ = get_MNIST
    set_up_svgd(stein_vi_multiclass_example)

    number_of_classes = 10

    nnet = stein_vi_multiclass_example.nnet
    particles = stein_vi_multiclass_example.state.particles

    # when
    predictions, precisions = jax.vmap(lambda p: nnet.predict(p, z_train))(particles)

    # then
    assert predictions.shape == (stein_vi_multiclass_example.parameter.num_particles, len(y_train))
    assert precisions.shape == (stein_vi_multiclass_example.parameter.num_particles, len(y_train), number_of_classes)
    assert jnp.all(precisions >= 0) and jnp.all(precisions <= 1), "Softmax output should be probabilities between 0 and 1"
    assert jnp.allclose(jnp.sum(precisions, axis=-1), 1), "Probabilities should sum to 1 across classes"


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


def test_predict_over_particles_classification(stein_vi_multiclass_example, get_MNIST):
    # given
    z_train, y_train, _, _, _, _ = get_MNIST
    set_up_svgd(stein_vi_multiclass_example)
    number_of_classes = 10

    # when
    predictions, precisions = stein_vi_multiclass_example.predict_over_particles(z_train)

    # then
    assert predictions.shape == (stein_vi_multiclass_example.parameter.num_particles, len(y_train))
    assert precisions.shape == (stein_vi_multiclass_example.parameter.num_particles, len(y_train), number_of_classes)
    assert jnp.all(precisions >= 0) and jnp.all(
        precisions <= 1), "Softmax output should be probabilities between 0 and 1"
    assert jnp.allclose(jnp.sum(precisions, axis=-1), 1), "Probabilities should sum to 1 across classes"


def test_predict_over_particles_regression(stein_vi_regression_example, get_regression_toy_example):
    # given
    regression_toy_example = get_regression_toy_example
    z_train, y_train, _, _, _, _ = regression_toy_example
    set_up_svgd(stein_vi_regression_example)

    # when
    predictions, precisions = stein_vi_regression_example.predict_over_particles(z_train)

    # then
    assert predictions.squeeze().shape == (stein_vi_regression_example.parameter.num_particles, len(y_train))
    assert precisions.squeeze().shape == (stein_vi_regression_example.parameter.num_particles, len(y_train))
