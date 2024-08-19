import functools
import jax
import jax.numpy as jnp
from optax import adam
import blackjax
from blackjax.vi.svgd import rbf_kernel, update_median_heuristic
from tqdm import tqdm


from BNN_Model import generate_cubic_data_2d, build_model
from get_posteriori import logp_unnormalized_posterior
from plot_3dim_CI import get_predictions_from_test_grid, plot_3d_scatterplot, get_test_samples_grid
from plot_mse import calculate_mse, plot_mse, create_mse_calc_data


def main():
    # Set random keys
    key = jax.random.PRNGKey(1)
    rng_key_observed, rng_key_init = jax.random.split(key, 2)

    # Set parameters for BNN run
    num_iterations = 10       # Set number of iterations
    num_points = 100              # Set number of generated data points
    num_particles = 50         # Set number of particles

    mse_wanted = True          # Set to True if you want to calculate the MSE over the iterations

    # Simulate the data
    input_1, input_2, true_output = generate_cubic_data_2d(rng_key_observed, num_points)

    # Get model
    nnet_model, tree_def, param_vec = build_model(key, input_1, input_2)

    # Init particles
    prior_mu, prior_prec, initial_particles_vector = initialize_particles(param_vec, rng_key_init, num_particles)

    # Create the function of the unnomalized log posterior
    logp_model = functools.partial(
        logp_unnormalized_posterior,
        nnet_model=nnet_model,
        input_1=input_1,
        input_2=input_2,
        true_output=true_output,
        prior_mu=prior_mu,
        prior_prec=prior_prec,
        treedef=tree_def,
    )

    # Run SVGD training loop with Adam optimizer
    out, mse = svgd_training_loop(
        log_p=logp_model,
        initial_position=initial_particles_vector,
        initial_kernel_parameters={"length_scale": 1.0},
        kernel=rbf_kernel,
        optimizer=adam(0.2),
        num_iterations=num_iterations,
        nnet_model=nnet_model,
        tree_def=tree_def,
        mse_wanted=mse_wanted,
    )

    x_grid = get_test_samples_grid()
    mean_predictions, std_predictions, predictions, true_y, x_grid = (
        get_predictions_from_test_grid(num_particles, nnet_model, out, tree_def, x_grid))
    plot_3d_scatterplot(predictions, x_grid, true_y, mean_predictions, num_particles)
    if mse_wanted:
        plot_mse(mse)


def initialize_particles(param_vec, rng_key_init, num_particles):
    inital_param_len = len(param_vec)

    prior_mu = jnp.zeros(inital_param_len)  # constant value as mean for prior
    prior_prec = jnp.eye(inital_param_len)  # constant value as precision for prior

    initial_particles_vector = jax.random.multivariate_normal(
        rng_key_init, prior_mu, prior_prec, shape=(num_particles,)
    )
    return prior_mu, prior_prec, initial_particles_vector


def svgd_training_loop(
        log_p,
        initial_position,
        initial_kernel_parameters,
        kernel,
        optimizer,
        *,
        num_iterations=1000,
        nnet_model,
        tree_def,
        mse_wanted=False,
):
    # Initialize SVGD
    grad_log_posterior = jax.grad(log_p)
    svgd = blackjax.svgd(grad_log_posterior, optimizer, kernel, update_median_heuristic)
    state = svgd.init(initial_position, initial_kernel_parameters)
    step = jax.jit(svgd.step)

    # Is MSE calculation wanted?
    if mse_wanted:
        random_input_1, random_input_2, mean_squared_errors = create_mse_calc_data(num_mse_calc_points=1000, minval=-3, maxval=3)
        for _ in tqdm(range(num_iterations), desc="Training"):
            state = step(state)
            mean_squared_errors = calculate_mse(nnet_model, tree_def, random_input_1, random_input_2, state, mean_squared_errors)
        return state, mean_squared_errors
    else:
        # Run the SVGD training loop
        for _ in tqdm(range(num_iterations), desc="Training"):
            state = step(state)
        return state, None


# Run the main function
if __name__ == "__main__":
    main()
