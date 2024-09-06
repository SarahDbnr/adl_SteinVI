import flax.linen as nn
from typing import Sequence
from jax.flatten_util import ravel_pytree
import jax
import jax.numpy as jnp


class SimpleCNN(nn.Module):
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
    if use_CNN:
        nnet_model = SimpleCNN() # TODO: doesnt run through validation and evaluation, has no predict
    else:
        nnet_model = FlexibleSimpleNN(hidden_layers, output_size, activation, kernel_init, bias_init,
                                      use_for_regression)

    init_param = nnet_model.init(key, x_train)
    param_vec, tree_def = ravel_pytree(init_param)
    return nnet_model, tree_def, param_vec
