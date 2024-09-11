import flax.linen as nn
from typing import Sequence
from jax.flatten_util import ravel_pytree
import jax
import jax.numpy as jnp


class SimpleCNN(nn.Module):
    """
    Defines a simple convolutional neural network (CNN) for image classification.

    Attributes:
        num_classes (int): Number of output classes for the classification task. :no-index:
    """
    num_classes: int = 10

    @nn.compact
    def __call__(self, x):
        # First convolutional layer
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Second convolutional layer
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Third convolutional layer
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flattening layer
        x = x.reshape((x.shape[0], -1))

        # First dense layer
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)

        # Optional Dropout layer for regularization
        # x = nn.Dropout(rate=0.5)(x, deterministic=False)

        # Second dense layer
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        # Output layer
        x = nn.Dense(features=self.num_classes)(x)
        return x


class FlexibleSimpleNN(nn.Module):
    """
    A flexible neural network model that can be customized for different architectures and tasks.

    Attributes:
        hidden_layers (Sequence[int]): Sequence of integers where each integer defines the number of neurons in a hidden layer. :no-index:
        output_size (int): Size of the output layer. :no-index:
        activation (callable): Activation function to use in the hidden layers. :no-index:
        kernel_init (callable): Initialization function for kernel weights. :no-index:
        bias_init (callable): Initialization function for biases. :no-index:
        use_for_regression (bool): Flag to determine whether the network is used for regression or classification. :no-index:
    """
    hidden_layers: Sequence[int] = (50,)
    output_size: int = 1
    activation: callable = nn.relu
    kernel_init: callable = nn.initializers.glorot_uniform()  # nn.initializers.lecun_normal()
    bias_init: callable = nn.initializers.zeros
    use_for_regression: bool = False

    @nn.compact
    def __call__(self, *inputs):
        # Combine all inputs
        # x = jnp.stack(inputs, axis=-1) #TODO: Musste dies auskommentieren für mnist
        x = inputs[0]
        x = x.reshape((x.shape[0], -1))
        # Apply hidden layers
        for units in self.hidden_layers:
            x = nn.Dense(features=units,
                         kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = self.activation(x)
            # x = nn.Dropout(rate=0.5)(x, deterministic=False)

        # Output layer
        x = nn.Dense(features=self.output_size,
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init)(x)
        return x.squeeze(-1) if self.output_size == 1 else x

    def predict(self, weights, x_input):
        if self.use_for_regression:
            output = self.apply(weights, x_input)
            prediction, precision = jnp.split(output, 2, axis=-1)
        else:
            predictions = self.apply(weights, x_input)
            precision = jax.nn.softmax(predictions, axis=-1)
            prediction = jnp.argmax(precision, axis=-1)
        return prediction, precision



def build_model(key, x_train, hidden_layers=(50,), output_size=10, activation=nn.relu,
                kernel_init=nn.initializers.lecun_normal(),
                bias_init=nn.initializers.zeros, use_CNN=False, use_for_regression=False):
    """
    Builds and initializes a neural network model based on specified configurations.

    Args:
        key: JAX random key for initialization.
        x_train (jax.numpy.ndarray): Sample input data used to define input shape.
        hidden_layers (tuple, optional): Tuple defining the number of units in each hidden layer.
        output_size (int, optional): The size of the output layer.
        activation (callable, optional): Activation function for the hidden layers.
        kernel_init (callable, optional): Weight initialization function.
        bias_init (callable, optional): Bias initialization function.
        use_CNN (bool, optional): Flag to choose between CNN and a simple flexible network.
        use_for_regression (bool, optional): Flag to specify if the model is intended for regression.

    Returns:
        tuple: The initialized model, the tree definition for parameter transformation, and a flattened parameter vector.
    """
    if use_CNN:
        nnet_model = SimpleCNN() # TODO: doesnt run through validation and evaluation, has no predict
    else:
        nnet_model = FlexibleSimpleNN(hidden_layers, output_size, activation, kernel_init, bias_init,
                                      use_for_regression)

    init_param = nnet_model.init(key, x_train)
    param_vec, tree_def = ravel_pytree(init_param)
    return nnet_model, tree_def, param_vec
