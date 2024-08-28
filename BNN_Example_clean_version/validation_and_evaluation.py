import jax
import jax.numpy as jnp

# from plots.plot_mse import plot_mse


# Evaluate the particles by averaging the predictions and calculating the accuracy
def get_mse_and_accuracy_over_predictions(out, nnet_model, tree_def, x_input, true_output, model_regression):
    predictions, precisions = jax.vmap(lambda p: nnet_model.predict(tree_def(p), x_input))(out.particles)
    if model_regression:
        mse = calculate_mse(predictions.squeeze(), true_output)
        averaged_precision = precisions.squeeze().mean()
        print(f"Averaged precision: {averaged_precision}, mean squared error: {mse}")
        return mse, averaged_precision
    else:
        # TODO: Set order precisions, mean, argmax or precisions, argmax, mean
        averaged_precision = precisions.squeeze().mean(0)
        predicted_classes = jnp.argmax(averaged_precision, axis=-1)
        accuracy = jnp.mean(predicted_classes == true_output)
        print(f"Validation accuracy: {accuracy}")
        return None, accuracy


def calculate_mse(predictions, true_output):
    mse = jnp.mean((predictions.mean(0) - true_output) ** 2)
    return mse
