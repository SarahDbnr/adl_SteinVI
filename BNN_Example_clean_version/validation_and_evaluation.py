import jax
import jax.numpy as jnp

# from plots.plot_mse import plot_mse


# Evaluate the particles by averaging the predictions and calculating the accuracy
def get_evaluation_metrics_over_predictions(out, nnet_model, tree_def, x_input, true_output, model_regression):
    predictions, precisions = jax.vmap(lambda p: nnet_model.predict(tree_def(p), x_input))(out.particles)
    if model_regression:
        mse = calculate_mse(predictions.squeeze(), true_output)
        averaged_precision = precisions.squeeze().mean()
        return mse, averaged_precision
    else:
        # TODO: Set order precisions, mean, argmax or precisions, argmax, mean
        averaged_precision = precisions.squeeze().mean(0)
        predicted_classes = jnp.argmax(averaged_precision, axis=-1)
        accuracy = jnp.mean(predicted_classes == true_output)
        return accuracy, None 


def calculate_mse(predictions, true_output):
    mse = jnp.mean((predictions.mean(0) - true_output) ** 2)
    return mse
