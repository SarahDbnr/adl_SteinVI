import jax
import jax.numpy as jnp

from BNN_Example_clean_version.plots.plot_mse import plot_mse


# Evaluate the particles by averaging the predictions and calculating the accuracy
def evaluate_particles(out, nnet_model, tree_def, x_input, true_output, model_regression):
    predictions, precisions = jax.vmap(lambda p: nnet_model.predict(tree_def(p), x_input))(out.particles)
    mse = get_mse(predictions.squeeze(), true_output)
    if model_regression:
        averaged_precision = precisions.squeeze().mean(0)
        print(f"Averaged precision: {averaged_precision}, mean squared error: {mse}")
        return mse, averaged_precision
    else:
        averaged_precision = precisions.squeeze().mean(0)
        predicted_classes = jnp.argmax(averaged_precision, axis=-1)
        accuracy = jnp.mean(predicted_classes == true_output)
        print(f"Validation accuracy: {averaged_precision}, mean squared error: {mse}")
        return mse, accuracy


def evaluate_mse_on_test_data(particles, x_input, true_output, nnet_model, tree_def):
    # Average predictions across particles
    predictions, precisions = jax.vmap(lambda p: nnet_model.predict(tree_def(p), x_input))(particles)
    mse = get_mse(predictions.squeeze(), true_output)

    jax.debug.print("Prediction[1:5]: {}", predictions[1:5])
    jax.debug.print("Test[1:5]: {}", true_output[1:5])
    jax.debug.print("Test max: {}", true_output.max())
    jax.debug.print("Test min: {}", true_output.min())

    print(f"Test MSE: {mse}")
    # TODO plot_mse(precisions)
    # TODO plot_mse(mse)


def get_mse(predictions, true_output):
    mse = jnp.mean((predictions.mean(0) - true_output) ** 2) # does not fit very well to classification
    return mse
