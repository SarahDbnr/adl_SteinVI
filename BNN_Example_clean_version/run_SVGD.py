import jax

from BNN_Model import build_model
from get_posteriori import logp_unnormalized_posterior_regression
from regression_toy_example import get_regression_toy_example
from svgd import run_svgd
from validation_and_evaluation import evaluate_mse_on_test_data


def run_svgd_on_regression_toy_example(num_particles=100, batch_size=20, output_size=2):
    # Network structure for the NNet
    network_structure = (200, 75, 40)

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(1)

    # Load the dataset and split into training, validation, and test sets
    regression_toy_example_data = get_regression_toy_example(num_points=100, key=key)
    z_train, y_train, z_val, y_val, z_test, y_test = regression_toy_example_data

    # Build the Neural Network model based on set input parameters
    nn_model = build_model(key, z_train, output_size=output_size, hidden_layers=network_structure,
                           use_for_regression=True)
    nnet_model, tree_def, param_vec = nn_model

    @jax.jit
    def logp_model(params, dz, dy):
        return logp_unnormalized_posterior_regression(
            params,
            nnet_model=nnet_model,
            dz=dz,
            dy=dy,
            treedef=tree_def,
        )

    out, mse, val_accuracies = run_svgd(regression_toy_example_data, batch_size, nn_model, logp_model, num_particles,
                                        key)

    evaluate_mse_on_test_data(out.particles, z_test, y_test, nnet_model, tree_def)


if __name__ == "__main__":
    run_svgd_on_regression_toy_example()
