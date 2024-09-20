import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
from optax import adam
from jax.flatten_util import ravel_pytree
from stein_vi.Classes.Parameter_Class import Parameter
from stein_vi.Classes.Handler_Class import Handler
from stein_vi.algorithm.get_posteriori import get_posteriori
from stein_vi.Classes.SteinVI_BNN import SteinVI_BNN
from run_stein_vi.model.BNN_Model import build_model

# Test for successful initialization
def test_steinvi_bnn_init_success():
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (100, 10))
    
    nnet = build_model(output_size=2, hidden_layers=(2,4))
    
    # Successful initialization
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
        num_particles=10, 
        num_iterations=100, 
        rf_comparison=False, 
        learning_rate=0.0001
    )

    # Assertions to check if the attributes are initialized correctly
    assert isinstance(model.handler, Handler), "Handler should be initialized correctly"
    assert model.handler._full_training_print, "Training print mode should be set correctly to full"
    assert model.handler._full_evaluation == False, "Evaluation mode should be set correctly"
    
    assert isinstance(model.parameter, Parameter), "Parameter should be initialized correctly"
    assert model.parameter.batch_size == 50, "Batch size should be set correctly"
    assert model.parameter.particle_batch_size == 5, "Particle batch size should be set correctly"
    assert model.parameter.num_particles == 10, "Number of particles should be set correctly"
    assert model.parameter.num_iterations == 100, "Number of iterations should be set correctly"
    assert model.parameter.early_stopping == True, "Early stopping should be set correctly"
    
    assert model.use_for_regression == True, "Use for regression flag should be set correctly"
    assert isinstance(model.nnet, nn.Module), "Neural network model should be initialized correctly"
    
    # Check if initial_particles_and_kernel is called correctly
    assert model.initial_particle_vector.shape == (10, 44), "Initial particle vector should have the correct shape"
    nnet.predict = model.predict
    # Check if the log posteriori function is set correctly
    #???assert model.log_posteriori == get_posteriori(nnet, regression=True), "Log posteriori function should be set correctly"
    
    


# Test for batch size larger than training data
def test_steinvi_bnn_init_fail_batch_size():
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (10, 10))  # Small dataset
    
    nnet = build_model(output_size=2, hidden_layers=(2,4))
    
    
    with pytest.raises(ValueError, match="Error: batch_size bigger then input data length"):
        SteinVI_BNN(
            key=key, 
            x_train=x_train, 
            nnet=nnet, 
            use_for_regression=True, 
            optimizer=adam(0.001), 
            batch_size=20,  # This is larger than the dataset length
            particle_batch_size=5, 
            num_particles=10, 
            num_iterations=100
        )


# Test for particle batch size larger than number of particles
def test_steinvi_bnn_init_fail_particle_batch_size():
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (100, 10))
    
    nnet = build_model(output_size=2, hidden_layers=(2,4))
    
    with pytest.raises(ValueError, match="Error: particle_batch_size bigger then number of particles"):
        SteinVI_BNN(
            key=key, 
            x_train=x_train, 
            nnet=nnet, 
            use_for_regression=True, 
            optimizer=adam(0.001), 
            batch_size=50, 
            particle_batch_size=15,  # This is larger than the number of particles
            num_particles=10, 
            num_iterations=100
        )



# Test for initial_particles_and_kernel function
def test_initial_particles_and_kernel():
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (100, 10))  # Example input data
    
    nnet = build_model(output_size=2, hidden_layers=(2,4))

    # Create the SteinVI_BNN instance to call initial_particles_and_kernel
    model = SteinVI_BNN(
        key=key, 
        x_train=x_train, 
        nnet=nnet, 
        use_for_regression=True, 
        optimizer=adam(0.001), 
        num_particles=10, 
        num_iterations=100
    )

    # Ensure the method runs and initializes particles and kernel structure correctly
    model.initial_particles_and_kernel(key, x_train, num_particles=10)

    # Test that the parameter vector was flattened correctly
    init_param = model.nnet.init(key, x_train)
    param_vec, tree_def = ravel_pytree(init_param)

    # Assert that the `tree_def` is the same as the one in the model after initialization
    assert model.tree_def == tree_def, "The tree_def should match the structure of the initialized network parameters"

    # Assert that the initial particle vector has the correct shape
    expected_shape = (10, param_vec.shape[0])  # 10 particles and the flattened param vector size
    assert model.initial_particle_vector.shape == expected_shape, (
        f"Expected initial_particle_vector shape {expected_shape}, but got {model.initial_particle_vector.shape}"
    )

    # Test if the particles are drawn from the correct normal distribution
    # We expect them to have mean close to 0 and standard deviation close to 1 (for a standard normal distribution)
    particle_mean = jnp.mean(model.initial_particle_vector)
    particle_std = jnp.std(model.initial_particle_vector)
    
    assert jnp.isclose(particle_mean, 0, atol=0.1), f"Particle mean should be close to 0, but got {particle_mean}"
    assert jnp.isclose(particle_std, 1, atol=0.1), f"Particle std should be close to 1, but got {particle_std}"




# Test for classification
def test_predict_classification():
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (100, 10))  # 100 samples with 10 features
    nnet = build_model(output_size=3, hidden_layers=(2,4))
    model = SteinVI_BNN(
        key=key,
        x_train=x_train,
        nnet=nnet,
        use_for_regression=False,  # Classification task
        optimizer=adam(0.001),
        num_particles=5,
        num_iterations=100
    )
    
    # Initialize random weights and input data
    weights = model.initial_particle_vector[0]
    x_input = jax.random.normal(key, (10, 10))  # 10 new samples with 10 features
    
    # Call predict
    prediction, precision = model.predict(weights, x_input)
    
    # Validate the shapes and behavior of predictions
    assert prediction.shape == (10,), "Prediction shape should be (batch_size,) for classification"
    assert precision.shape == (10, 3), "Precision (probabilities) shape should be (batch_size, num_classes)"
    
    # Softmax should generate probabilities
    assert jnp.all(precision >= 0) and jnp.all(precision <= 1), "Softmax output should be probabilities between 0 and 1"
    assert jnp.allclose(jnp.sum(precision, axis=-1), 1), "Probabilities should sum to 1 across classes"

# Test for regression
def test_predict_regression():
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (100, 10))  # 100 samples with 10 features
    
    # Initialize a regression network
    nnet = build_model(output_size=2, hidden_layers=(2,4))
    model = SteinVI_BNN(
        key=key,
        x_train=x_train,
        nnet=nnet,
        use_for_regression=True,  # Regression task
        optimizer=adam(0.001),
        num_particles=5,
        num_iterations=100
    )
    
    # Initialize random weights and input data
    weights = model.initial_particle_vector[0]
    x_input = jax.random.normal(key, (10, 10))  # 10 new samples with 10 features
    
    # Call predict
    prediction, precision = model.predict(weights, x_input)
    
    # Validate the shapes and behavior of predictions
    assert prediction.shape == (10, 1), "Prediction shape should be (batch_size, 1) for regression"
    assert precision.shape == (10, 1), "Precision (variance) shape should be (batch_size, 1) for regression"
    


# Mock state class to hold particles, normally in Blackjax
class MockState:
    def __init__(self, particles):
        self.particles = particles
# Test predict_over_particles for classification
def test_predict_over_particles_classification():
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (100, 10))  # 100 samples, 10 features
    
    # Initialize classification model and particles
    nnet = build_model(output_size=3, hidden_layers=(2,4))
    model = SteinVI_BNN(
        key=key, 
        x_train=x_train, 
        nnet=nnet, 
        use_for_regression=False,  # Classification task
        optimizer=adam(0.001), 
        num_particles=5, 
        num_iterations=100
    )
    
    # Create dummy particle states
    num_particles = 5
    param_shape = model.initial_particle_vector.shape[1:]  # Shape of a single particle
    particles = jax.random.normal(key, (num_particles,) + param_shape)
    
    # Mock the state with particles
    model.state = MockState(particles)
    
    # Input data to predict
    x_input = jax.random.normal(key, (10, 10))  # 10 samples, 10 features
    
    # Run predict_over_particles
    predictions, precisions = model.predict_over_particles(x_input)
    
    # Check the shapes of the predictions and precisions
    assert predictions.shape == (5, 10), "Predictions shape should be (num_particles, batch_size)"
    assert precisions.shape == (5, 10, 3), "Precision shape should be (num_particles, batch_size, num_classes)"
    
    # Print outputs for inspection

# Test predict_over_particles for regression
def test_predict_over_particles_regression():
    key = jax.random.PRNGKey(0)
    x_train = jax.random.normal(key, (100, 10))  # 100 samples, 10 features
    
    # Initialize regression model and particles
    nnet = build_model(output_size=2, hidden_layers=(2,4))
    model = SteinVI_BNN(
        key=key, 
        x_train=x_train, 
        nnet=nnet, 
        use_for_regression=True,  # Regression task
        optimizer=adam(0.001), 
        num_particles=5, 
        num_iterations=100
    )
    
    # Create dummy particle states
    num_particles = 5
    param_shape = model.initial_particle_vector.shape[1:]  # Shape of a single particle
    particles = jax.random.normal(key, (num_particles,) + param_shape)
    
    # Mock the state with particles
    model.state = MockState(particles)
    
    # Input data to predict
    x_input = jax.random.normal(key, (10, 10))  # 10 samples, 10 features
    
    # Run predict_over_particles
    predictions, precisions = model.predict_over_particles(x_input)
    
    # Check the shapes of the predictions and precisions
    assert predictions.shape == (5, 10), "Predictions shape should be (num_particles, batch_size, 1)"
    assert precisions.shape == (5, 10), "Precision shape should be (num_particles, batch_size, 1)"
    



