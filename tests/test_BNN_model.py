import pytest
import jax.numpy as jnp
from flax import linen as nn
from jax import random
import jax
from jax.flatten_util import ravel_pytree
from src.model.BNN_Model import FlexibleSimpleNN, build_model

def test_flexible_simple_nn_classification():
    # Set up the model for classification
    model = FlexibleSimpleNN(
        hidden_layers=[10, 10],  # Example hidden layer sizes
        output_size=3,  # Example for a classification task with 3 classes
        activation=nn.relu,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros,
        use_for_regression=False
    )
    
    # Initialize random weights
    key = random.PRNGKey(0)
    
    # Example input for MNIST: shape (batch_size, height, width)
    x_input = random.normal(key, (1, 28, 28))  # Example input size for MNIST (28x28)

    # Initialize parameters
    params = model.init(key, x_input)
    
    # Get predictions
    predictions, precision = model.predict(params, x_input)
    # Calculate the expected number of parameters
    input_size = 28 * 28  # MNIST input flattened size
    hidden_layers = [10, 10]  # Hidden layers sizes
    output_size = 3  # Output size for classification
     # Calculate total parameters
    expected_num_params = 0
    
    # Input layer to first hidden layer
    expected_num_params += input_size * hidden_layers[0] + hidden_layers[0]  # weights + biases
    
    # Hidden layers
    for i in range(len(hidden_layers) - 1):
        expected_num_params += hidden_layers[i] * hidden_layers[i + 1] + hidden_layers[i + 1]
    
    # Last hidden layer to output layer
    expected_num_params += hidden_layers[-1] * output_size + output_size
    
    # Flatten the parameters dictionary to count elements
    def count_params(params):
        return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))
    
    actual_num_params = count_params(params)

    # Assertions
    assert predictions.shape == (1,)
    assert precision.shape == (1, 3)
    assert jnp.all(precision >= 0) and jnp.all(precision <= 1)  # softmax probabilities
    assert actual_num_params == expected_num_params, f"Expected {expected_num_params} parameters, but got {actual_num_params}"
    
    
def test_flexible_simple_nn_regression():
    # Set up the model for regression
    model = FlexibleSimpleNN(
        hidden_layers=[10, 10],  # Example hidden layer sizes
        output_size=2,  # Example for a regression task with mean and variance outputs
        activation=nn.relu,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros,
        use_for_regression=True
    )
    
    # Initialize random weights
    key = random.PRNGKey(0)
    x_input = random.normal(key, (1, 28 * 28))  # Example input size for MNIST (28x28)
    
    # Initialize parameters
    params = model.init(key, x_input)
    
    # Get predictions
    prediction, precision = model.predict(params, x_input)

    # Assertions
    assert prediction.shape == (1, 1)
    assert precision.shape == (1, 1)  # Example with regression output size of 2 split into prediction and precision


# Assuming the FlexibleSimpleNN class is imported and available

def test_flexible_simple_nn_all_ones():
    # Set up the model
    model = FlexibleSimpleNN(
        hidden_layers=(2, 2),  # Small example for simplicity
        output_size=2,  # Example output size for classification
        activation=nn.relu,
        kernel_init=nn.initializers.ones,
        bias_init=nn.initializers.ones,
        use_for_regression=False
    )
    
    # Initialize random weights with key
    key = random.PRNGKey(0)
    
    # Input data, all ones
    input_size = 4  # Example flattened input size
    x_input = jnp.ones((1, input_size))  # Batch size of 1, input size of 4 (all ones)
    
    # Initialize parameters with the custom initializers
    params = model.init(key, x_input)
    
    # Get predictions and precision from the model
    predictions, precision = model.predict(params, x_input)

    # Manually calculate the expected output
    expected_output = input_size + 1  # Each neuron in the first hidden layer receives input size * 1 + bias * 1
    
    # Expected output after applying each layer
    expected_hidden_output = jnp.array([expected_output] * model.hidden_layers[0])
    
    # Since ReLU is used and input is positive, ReLU does nothing
    expected_final_output = jnp.array([expected_output] * model.output_size)

    # For classification, precision should be softmax of final output
    expected_precision = jax.nn.softmax(expected_final_output, axis=-1)
    expected_prediction = jnp.argmax(expected_precision, axis=-1)

    # Assertions
    assert jnp.allclose(predictions, expected_prediction), \
        f"Expected predictions: {expected_prediction}, but got: {predictions}"
    
    assert jnp.allclose(precision, expected_precision), \
        f"Expected precision: {expected_precision}, but got: {precision}"
    


def test_build_model():
    # Initialize random key
    key = random.PRNGKey(0)
    
    # Create sample input data
    x_train = jnp.ones((1, 28, 28))  # Example input for MNIST-like data, with shape (1, 28, 28)
    
    # Define model parameters
    hidden_layers = (10, 10)  # Example hidden layers
    output_size = 10  # Example output size for classification (10 classes)
    activation = nn.relu
    kernel_init = nn.initializers.glorot_normal()
    bias_init = nn.initializers.zeros
    use_for_regression = False

    # Call the build_model function
    nnet_model, tree_def, param_vec = build_model(
        key, x_train, hidden_layers, output_size, activation, kernel_init, bias_init, use_for_regression
    )
    
    # Assertions to check if the model was built correctly
    assert isinstance(nnet_model, FlexibleSimpleNN), "The model is not an instance of FlexibleSimpleNN."

    # Recalculate the expected number of parameters
    input_size = x_train.shape[-1]*x_train.shape[-1]
    expected_num_params = 0
    
    # Input layer to first hidden layer
    expected_num_params += input_size * hidden_layers[0] + hidden_layers[0]  # weights + biases
    
    # Hidden layers
    for i in range(len(hidden_layers) - 1):
        expected_num_params += hidden_layers[i] * hidden_layers[i + 1] + hidden_layers[i + 1]
    
    # Last hidden layer to output layer
    expected_num_params += hidden_layers[-1] * output_size + output_size

    # Assert that the flattened parameter vector has the correct length
    assert param_vec.size == expected_num_params, f"Expected {expected_num_params} parameters, but got {param_vec.size}."

    # Test that the tree_def can be used to reconstruct the parameters correctly
    original_params = nnet_model.init(key, x_train)

    # Check that all elements in the original and reconstructed trees are close (numerically equal)
    def check_params_equal(p1, p2):
        return jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), p1, p2))
    
    assert check_params_equal(params_reconstructed, original_params), "Reconstructed parameters do not match the original initialization."
if __name__ == "__main__":
    pytest.main()