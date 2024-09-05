import jax
import jax.numpy as jnp

from BNN_Example_clean_version.get_posteriori import logp_unnormalized_posterior_regression, link_function
from BNN_Example_clean_version.regression_toy_example import get_regression_toy_example
from BNN_Example_clean_version.BNN_Model import build_model

def test_logp_unnormalized_posterior_regression():
    # given
    key = jax.random.PRNGKey(1)
    z_train, y_train, z_val, y_val, z_test, y_test = get_regression_toy_example(num_points=10, input_dimension=2)
    nnet_model, tree_def, param_vec = build_model(key, z_train, output_size=2,
                                                  hidden_layers=(200, 75, 40),
                                                  use_for_regression=True)
    # when
    likelihood = logp_unnormalized_posterior_regression(param_vec, z_train, y_train, nnet_model, tree_def)
    print(likelihood)


def test_link_function_in_vmap():
    # given
    prediction = jnp.arange(10)
    # when
    scale = jax.vmap(lambda p: link_function(p))(prediction)
    # then
    assert scale.shape == prediction.shape
    assert scale[0] == 0
