import flax.linen as nn
from typing import Sequence
import jax.numpy as jnp
import jax
from jax.flatten_util import ravel_pytree


class FlexibleSimpleNN(nn.Module):
    hidden_layers: Sequence[int] = (50,)
    output_size: int = 1
    activation: callable = nn.relu
    kernel_init: callable = nn.initializers.lecun_normal()
    bias_init: callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, *inputs):
        # Combine all inputs
        x = jnp.stack(inputs, axis=-1)

        # Apply hidden layers
        for units in self.hidden_layers:
            x = nn.Dense(features=units,
                         kernel_init=self.kernel_init,
                         bias_init=self.bias_init)(x)
            x = self.activation(x)

        # Output layer
        x = nn.Dense(features=self.output_size,
                     kernel_init=self.kernel_init,
                     bias_init=self.bias_init)(x)

        return x.squeeze(-1) if self.output_size == 1 else x


def build_model(key, input_1, input_2):
    nnet_model = create_model()
    init_param = nnet_model.init(key, input_1, input_2)

    param_vec, tree_def = ravel_pytree(init_param)

    return nnet_model, tree_def, param_vec


def create_model(hidden_layers=(50,), output_size=1, activation=nn.relu,
                 kernel_init=nn.initializers.lecun_normal(), bias_init=nn.initializers.zeros):
    return FlexibleSimpleNN(hidden_layers=hidden_layers,
                            output_size=output_size,
                            activation=activation,
                            kernel_init=kernel_init,
                            bias_init=bias_init)


def generate_cubic_data_2d(rng_key, num_points=1000):
    """Generate synthetic data with a cubic relationship for 2D input."""
    key1, key2 = jax.random.split(rng_key)
    x1 = jax.random.uniform(key1, (num_points, 1), minval=-2, maxval=2)
    x2 = jax.random.uniform(key2, (num_points, 1), minval=-2, maxval=2)
    y = x1 ** 3 - x2 ** 3 + 0.5 * x1 * x2  # + noise_std * jax.random.normal(rng_key, (num_points, 1))
    return x1, x2, y


def get_true_y(x1, x2):
    return x1 ** 3 - x2 ** 3 + 0.5 * x1 * x2


def apply_model(model, params, x):
    return model.apply(params, x)
