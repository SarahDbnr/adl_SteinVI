import jax
import jax.numpy as jnp

from stein_vi.algorithm.get_posteriori import link_function

ALPHA = 0.05


def get_evaluation_metrics_over_predictions(out, nnet_model, x_input, true_output, model_regression, print_eva):
    """
    Evaluates predictions from a model, calculating either mean squared error (MSE) and average variance for regression or accuracy for classification.

    Args:
        out (blackjax.vi.svgd.SVGD_State): Output object containing particles representing different model parameters.
        nnet_model (flax.linen.Module): The neural network model used for making predictions.
        tree_def (jax.tree_util.PyTreeDef): Tree structure used for parameter transformation in JAX.
        x_input (jax.numpy.ndarray): Input features to the model.
        true_output (jax.numpy.ndarray): True output labels or values for the given input.
        model_regression (bool): A flag indicating whether the model is used for regression or classification.
        print_eva (str): A flag indictating whether the model should print the metrics or not.

    Returns:
        tuple: For regression, returns (MSE, averaged variance, predictions); for classification, returns (accuracy, predictions, None).
    """
    predictions, precisions = jax.vmap(lambda p: nnet_model.predict(p, x_input))(out.particles)
    if model_regression:
        mse = calculate_mse(predictions.squeeze(), true_output)
        scale = jax.vmap(lambda p: link_function(p))(precisions.squeeze())
        averaged_var = jnp.sqrt(scale).mean()
        if print_eva:
            print(f"MSE: {mse}, Average Variance: {averaged_var} with mean predictions of {predictions.squeeze().mean()}")
        return mse, averaged_var, predictions
    else:
        accuracy = calculate_accuracy(precisions, true_output)
        if print_eva:
            print(f"Accuracy: {accuracy} ")
        return accuracy, None, predictions


def calculate_mse(predictions, true_output):
    """
    Calculates the mean squared error (MSE) between predicted values and the true output.

    Args:
        predictions (jax.numpy.ndarray): Predicted values from the model, expected to be means over an ensemble of particles.
        true_output (jax.numpy.ndarray): True output values.

    Returns:
        float: Computed mean squared error.
    """
    mse = jnp.mean((predictions.mean(0) - true_output) ** 2)
    return mse


def calculate_accuracy(precisions, true_output):
    """
    Calculates the accuracy of classification predictions.

    Args:
        precisions (jax.numpy.ndarray): Model precisions or probabilities output for each class.
        true_output (jax.numpy.ndarray): True class labels.

    Returns:
        float: The classification accuracy as the percentage of correct predictions.
    """
    averaged_precision = precisions.mean(0)
    predicted_classes = jnp.argmax(averaged_precision, axis=-1)
    return jnp.mean(predicted_classes == true_output)


def print_summary_over_particles_regression(predictions):
    """
    Prints a summary of predictions over particles for regression, including the average prediction span and the mean prediction.

    Args:
        predictions (jax.numpy.ndarray): Predictions from an ensemble of particles.
    """
    predictions = predictions.squeeze()
    prediction_span = calculate_mean_span_over_particles(predictions)
    print("\nAverage prediction span including " + str(1 - ALPHA) + "% of particles :" + str(prediction_span.mean()) +
          " with mean_predictions of " + str(predictions.mean()))


def calculate_mean_span_over_particles(predictions):
    """
    Calculates the span of predictions that include the central (1-ALPHA)% of data over particles.

    Args:
        predictions (jax.numpy.ndarray): Predictions from an ensemble of particles.

    Returns:
        float: The difference between the upper and lower quantiles of the predictions.
    """
    upper_quantile_prediction_over_particles = jnp.quantile(predictions, 1 - ALPHA / 2)
    lower_quantile_prediction_over_particles = jnp.quantile(predictions, ALPHA / 2)
    return upper_quantile_prediction_over_particles - lower_quantile_prediction_over_particles


def print_summary_over_particles_multiclass(predictions):
    predictions = predictions.squeeze()
    number_of_different_classified_by_particles = jnp.array(
        calculate_number_of_different_classified_by_particles(predictions))
    print("\nAverage number of different classifications over all particles "
          + str(number_of_different_classified_by_particles.mean()))


def calculate_number_of_different_classified_by_particles(predictions):
    """
    Calculates the number of different classifications made by particles for each input instance.

    Args:
        predictions (jax.numpy.ndarray): Predictions from an ensemble of particles.

    Returns:
        list: A list where each element is the number of different predictions made by particles for an input.
    """
    num_input_values = predictions.shape[1]
    difference_classified_by_particles = []
    for i in range(num_input_values):
        unique_vals, col_counts = jnp.unique(predictions[:, i], return_counts=True)
        difference_classified_by_particles.append(col_counts.sum() - col_counts.max())
    return difference_classified_by_particles


def get_most_common_class_over_particles(predictions):
    """
    Determines the most common class predicted by particles for each input instance.

    Args:
        predictions (jax.numpy.ndarray): Predictions from an ensemble of particles.

    Returns:
        list: A list of the most commonly predicted class for each input instance.
    """
    num_input_values = predictions.shape[1]
    most_common_prediction_over_particles = []
    for i in range(num_input_values):
        most_common_class = get_most_common_class(predictions[:, i])
        most_common_prediction_over_particles.append(most_common_class)
    return most_common_prediction_over_particles


def get_most_common_class(column):
    """
    Identifies the most common class in a set of predictions.

    Args:
        column (jax.numpy.ndarray): A column vector of predictions for a single input across multiple particles.

    Returns:
        scalar: The most frequently occurring class in the column.
    """
    unique_vals, col_counts = jnp.unique(column, return_counts=True)
    max_index = jnp.argmax(col_counts)
    return unique_vals[max_index]


def compute_confidence_intervals_with_2_neurons(nnet_model, out, dz):
    """This function computes the normal distribution p(y|x) for the test data based on the paper:
    A Deeper Look into Aleatoric and Epistemic Uncertainty Disentanglement by Matias Valdenegro-Toro and Daniel Saromo Mori.
    Based on the equations (1), (2) and (3).

    Args:
        nnet_model flax.linen.Module): Underlying neural network of the training process. 
        tree_def (jax.tree_util.PyTreeDef): Tree structure used for parameter transformation in JAX.
        out (blackjax.vi.svgd.SVGD_State): Output from the training process, containing the state of the particles.
        dz (jax.numpy.ndarray): dataset of input features.
    """
    predictions, precision = jax.vmap(lambda p: nnet_model.predict(p, dz))(out.particles)
    predictions = predictions.squeeze()
    precision = precision.squeeze()
    mean_star = predictions.mean(0)  
    
    squared_means_i = jnp.square(predictions)

    variance_i = jax.vmap(lambda p: link_function(p))(precision.squeeze())

    variance_star = variance_i.mean(0) + squared_means_i.mean(0) - jnp.square(mean_star)
    return mean_star, variance_star
