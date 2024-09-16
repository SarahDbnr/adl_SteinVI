from optax import adam, exponential_decay
import jax

from stein_vi.Classes.SteinVI_BNN import SteinVI_BNN
from run_stein_vi.model.BNN_Model import build_model
from run_stein_vi.data.regression_toy_example import get_regression_toy_example
from stein_vi.stein_vi import train_with_stein_vi


def run_regression_toy_example():
    """
    Run SVGD on a synthetic regression toy example.

    Args:
        info (bool, optional): If True, prints dataset information. Defaults to False.
    """

    key = jax.random.PRNGKey(1)

    regression_toy_example = get_regression_toy_example(num_points=10000)
    z_train, _, _, _, z_test, y_test = regression_toy_example

    optimizer = adam(
        exponential_decay(
            init_value=0.1,
            transition_steps=20,
            decay_rate=0.95,
            staircase=True
        )
    )

    nnet_model = build_model(output_size=2)

    steinvi_svdg = SteinVI_BNN(key, z_train, nnet_model, use_for_regression=True, optimizer=optimizer)

    steinvi_svdg = train_with_stein_vi(steinvi_svdg, regression_toy_example, key, algorithm="svgd")

    steinvi_svdg.plot_val_metric_over_iter()


if __name__ == "__main__":
    run_regression_toy_example()
