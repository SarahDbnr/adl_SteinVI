import pytest
import jax.numpy as jnp
from flax import linen as nn
from jax import random
import jax
from run_stein_vi.model.BNN_Model import FlexibleSimpleNN, build_model

def test_flexible_simple_nn_classification():
    # given
    model = FlexibleSimpleNN(
        hidden_layers=[10, 10], 
        output_size=3,
        activation=nn.relu,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros
    )
    key = random.PRNGKey(0)
    x_input = random.normal(key, (1, 28, 28)) 
    input_size = 28 * 28 
    hidden_layers = [10, 10]
    output_size = 3 
    expected_num_params = 0
    # then
    params = model.init(key, x_input)
    predictions = model.apply(params, x_input)
    expected_num_params += input_size * hidden_layers[0] + hidden_layers[0]
    
    for i in range(len(hidden_layers) - 1):
        expected_num_params += hidden_layers[i] * hidden_layers[i + 1] + hidden_layers[i + 1]

    expected_num_params += hidden_layers[-1] * output_size + output_size
    
    def count_params(params):
        return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))
    
    actual_num_params = count_params(params)

    # when
    assert predictions.shape == (1, 3), "Prediction shape should match the batch size and number of classes"
    assert actual_num_params == expected_num_params, f"Expected {expected_num_params} parameters, but got {actual_num_params}"
    

def test_flexible_simple_nn_regression():
    # given
    model = FlexibleSimpleNN(
        hidden_layers=[10, 10],
        output_size=2,
        activation=nn.relu,
        kernel_init=nn.initializers.glorot_uniform(),
        bias_init=nn.initializers.zeros
    )
    
    key = random.PRNGKey(0)
    x_input = random.normal(key, (1, 28 * 28))
    
    params = model.init(key, x_input)
    
    # when
    prediction = model.apply(params, x_input)

    # then
    assert prediction.shape == (1, 2), "Regression model should output two values (mean and variance)"



def test_build_model():
    # given 
    key = random.PRNGKey(0)
    
    x_train = jnp.ones((1, 28, 28))  
    hidden_layers = (10, 10)  
    output_size = 10  
    activation = nn.relu
    kernel_init = nn.initializers.glorot_normal()
    bias_init = nn.initializers.zeros
    # when
    nnet_model = build_model(hidden_layers, output_size, activation, kernel_init, bias_init)
    
    params = nnet_model.init(key, x_train)
    
    def count_params(params):
        return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))
    
    input_size = 28 * 28
    expected_num_params = 0
    
    expected_num_params += input_size * hidden_layers[0] + hidden_layers[0]  
    
    for i in range(len(hidden_layers) - 1):
        expected_num_params += hidden_layers[i] * hidden_layers[i + 1] + hidden_layers[i + 1]
    
    expected_num_params += hidden_layers[-1] * output_size + output_size

    actual_num_params = count_params(params)
    # then 
    assert actual_num_params == expected_num_params, f"Expected {expected_num_params} parameters, but got {actual_num_params}"





def test_flexible_simple_nn_all_ones():
    # given
    model = FlexibleSimpleNN(
        hidden_layers=(2, 2), 
        output_size=2,  
        activation=nn.relu,
        kernel_init=nn.initializers.ones,
        bias_init=nn.initializers.zeros
    )
    key = random.PRNGKey(0)
    input_size = 4  
    x_input = jnp.ones((1, input_size)) 
    
    
    params = model.init(key, x_input)
    expected_output = input_size * 2 * 2  
    
    expected_final_output = jnp.array([expected_output] * model.output_size, dtype=jnp.float32)
    # when 
    predictions = model.apply(params, x_input)
    # then
    assert jnp.allclose(predictions, expected_final_output, atol=1e-5), \
        f"Expected predictions: {expected_final_output}, but got: {predictions}"
