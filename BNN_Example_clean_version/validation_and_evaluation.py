import jax
import jax.numpy as jnp

from BNN_Example_clean_version.get_posteriori import link_function

ALPHA = 0.05


# Evaluate the particles by averaging the predictions and calculating the accuracy
def get_evaluation_metrics_over_predictions(out, nnet_model, tree_def, x_input, true_output, model_regression):
    predictions, precisions = jax.vmap(lambda p: nnet_model.predict(tree_def(p), x_input))(out.particles)
    if model_regression:
        mse = calculate_mse(predictions.squeeze(), true_output)
        scale = jax.vmap(lambda p: link_function(p))(precisions.squeeze())
        averaged_var = jnp.sqrt(scale).mean()
        print(f"\nMSE: {mse}, Average Variance: {averaged_var} with mean predictions of {predictions.squeeze().mean()}")
        return mse, averaged_var, predictions
    else:
        accuracy = calculate_accuracy(precisions, true_output)
        print(f"\nAccuracy: {accuracy} with mean predictions of {predictions.squeeze().mean()}")
        return accuracy, None, predictions


def calculate_mse(predictions, true_output):
    mse = jnp.mean((predictions.mean(0) - true_output) ** 2)
    return mse


def calculate_accuracy(precisions, true_output):
    # TODO: Set order precisions, mean, argmax or precisions, argmax, mean
    averaged_precision = precisions.mean(0)
    predicted_classes = jnp.argmax(averaged_precision, axis=-1)
    return jnp.mean(predicted_classes == true_output)


def print_summary_over_particles(predictions):
    predictions = predictions.squeeze()
    prediction_span = calculate_mean_span_over_particles(predictions)
    print("\nAverage prediction span including " + str(1 - ALPHA) + "% of particles :" + str(prediction_span.mean()) +
          " with mean_predictions of " + str(predictions.mean()))


def calculate_mean_span_over_particles(predictions):
    upper_quantile_prediction_over_particles = jnp.quantile(predictions, 1 - ALPHA / 2)
    lower_quantile_prediction_over_particles = jnp.quantile(predictions, ALPHA / 2)
    return upper_quantile_prediction_over_particles - lower_quantile_prediction_over_particles
