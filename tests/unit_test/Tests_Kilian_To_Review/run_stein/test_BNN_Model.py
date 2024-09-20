import pytest
import jax.numpy as jnp
from flax import linen as nn
from jax import random
import jax
from run_stein_vi.model.BNN_Model import FlexibleSimpleNN, build_model  # Update to your correct module path

def test_flexible_simple_nn_classification():
    # Set up the model for classification
    model = FlexibleSimpleNN(
        hidden_layers=[10, 10],  # Example hidden layer sizes
        output_size=3,  # Example for a classification task with 3 classes
        activation=nn.relu,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros
    )
    
    # Initialize random weights
    key = random.PRNGKey(0)
    
    # Example input for MNIST-like data: shape (batch_size, height, width)
    x_input = random.normal(key, (1, 28, 28))  # Example input size for MNIST (28x28)
    
    # Initialize parameters
    params = model.init(key, x_input)
    
    # Get predictions
    predictions = model.apply(params, x_input)
    
    # Calculate the expected number of parameters
    input_size = 28 * 28  # MNIST input flattened size
    hidden_layers = [10, 10]  # Hidden layers sizes
    output_size = 3  # Output size for classification
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
    assert predictions.shape == (1, 3), "Prediction shape should match the batch size and number of classes"
    assert actual_num_params == expected_num_params, f"Expected {expected_num_params} parameters, but got {actual_num_params}"
    

def test_flexible_simple_nn_regression():
    # Set up the model for regression
    model = FlexibleSimpleNN(
        hidden_layers=[10, 10],  # Example hidden layer sizes
        output_size=2,  # Example for a regression task with mean and variance outputs
        activation=nn.relu,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros
    )
    
    # Initialize random weights
    key = random.PRNGKey(0)
    x_input = random.normal(key, (1, 28 * 28))  # Example input size
    
    # Initialize parameters
    params = model.init(key, x_input)
    
    # Get predictions
    prediction = model.apply(params, x_input)

    # Assertions
    assert prediction.shape == (1, 2), "Regression model should output two values (mean and variance)"




def test_flexible_simple_nn_all_ones():
    # Set up the model
    model = FlexibleSimpleNN(
        hidden_layers=(2, 2),  # Small example for simplicity
        output_size=2,  # Example output size for classification
        activation=nn.relu,
        kernel_init=nn.initializers.ones,
        bias_init=nn.initializers.zeros
    )
    
    # Initialize random weights with key
    key = random.PRNGKey(0)
    
    # Input data, all ones
    input_size = 4  # Example flattened input size
    x_input = jnp.ones((1, input_size))  # Batch size of 1, input size of 4 (all ones)
    
    # Initialize parameters with the custom initializers
    params = model.init(key, x_input)
    
    # Get predictions from the model
    predictions = model.apply(params, x_input)

    # Manually calculate the expected output
    expected_output = input_size * 2 * 2  # Each neuron in the first hidden layer receives input size * 1 + bias * 1
    
    # Since ReLU is used and input is positive, ReLU does nothing
    expected_final_output = jnp.array([expected_output] * model.output_size, dtype=jnp.float32)  # Ensure float type

    # Ensure type consistency and compare values
    assert jnp.allclose(predictions, expected_final_output, atol=1e-5), \
        f"Expected predictions: {expected_final_output}, but got: {predictions}"

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

    # Call the build_model function
    nnet_model = build_model(hidden_layers, output_size, activation, kernel_init, bias_init)
    
    # Initialize parameters using the model
    params = nnet_model.init(key, x_train)
    
    # Flatten the parameters dictionary to count elements
    def count_params(params):
        return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))
    
    # Calculate the expected number of parameters
    input_size = 28 * 28  # Flattened input size
    expected_num_params = 0
    
    # Input layer to first hidden layer
    expected_num_params += input_size * hidden_layers[0] + hidden_layers[0]  # weights + biases
    
    # Hidden layers
    for i in range(len(hidden_layers) - 1):
        expected_num_params += hidden_layers[i] * hidden_layers[i + 1] + hidden_layers[i + 1]
    
    # Last hidden layer to output layer
    expected_num_params += hidden_layers[-1] * output_size + output_size

    # Test that the parameter vector has the correct length
    actual_num_params = count_params(params)
    assert actual_num_params == expected_num_params, f"Expected {expected_num_params} parameters, but got {actual_num_params}"


if __name__ == "__main__":
    test_build_model()