import jax
import jax.numpy as jnp

ALPHA = 0.05


# Evaluate the particles by averaging the predictions and calculating the accuracy
def get_evaluation_metrics_over_predictions(out, nnet_model, tree_def, x_input, true_output, model_regression):
    predictions, precisions = jax.vmap(lambda p: nnet_model.predict(tree_def(p), x_input))(out.particles)
    if model_regression:
        mse = calculate_mse(predictions.squeeze(), true_output)
        averaged_precision = precisions.squeeze().mean()
        return mse, averaged_precision, predictions
    else:
        # TODO: Set order precisions, mean, argmax or precisions, argmax, mean
        averaged_precision = precisions.squeeze().mean(0)
        predicted_classes = jnp.argmax(averaged_precision, axis=-1)
        accuracy = jnp.mean(predicted_classes == true_output)
        return accuracy, None, predictions


def calculate_mse(predictions, true_output):
    mse = jnp.mean((predictions.mean(0) - true_output) ** 2)
    return mse


def print_summary_over_particles(predictions):
    predictions = predictions.squeeze()
    upper_quantile_prediction_over_particles = jnp.quantile(predictions, 1 - ALPHA / 2)
    lower_quantile_prediction_over_particles = jnp.quantile(predictions, ALPHA / 2)
    prediction_span = upper_quantile_prediction_over_particles - lower_quantile_prediction_over_particles
    print("\nAverage prediction span including " + str(1 - ALPHA) + "% of particles :" + str(prediction_span.mean())+
          " with mean_predictions of "+str(predictions.mean()))
