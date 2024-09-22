import pandas as pd
import jax.numpy as jnp

from stein_vi.metrics.validation_and_evaluation import (calculate_mse, calculate_mean_span_over_particles,
                                                        calculate_accuracy,
                                                        calculate_number_of_different_classified_by_particles,
                                                        get_most_common_class_over_particles,
                                                        get_most_common_class)


def print_evaluation_regression_to_csv(name, parameter, true_output, test_predictions, test_precision, init_value,
                                       decay_rate):
    """
    Saves evaluation metrics for a regression model into a CSV file. If the file already exists, it appends the data.

    Args:
        name (str): The name of the model or experiment for identifying the file.
        parameter (object): Object containing SVGD training parameters like batch size, num_particles, 
                            kernel length, and early stopping parameters.
        true_output (jax.numpy.ndarray): The true output values (ground truth) for the test set.
        test_predictions (jax.numpy.ndarray): Predicted values from the SVGD model over particles.
        test_precision (jax.numpy.ndarray): Predicted precision values from the model over particles.
        init_value (float): The initial learning rate.
        decay_rate (float): The decay rate choosen for the optimizer.

    Returns:
        None: The function saves the evaluation to a CSV file.
    """
    data = {
        # network
        "name": name,
        "init_value": init_value,
        "decay_rate": decay_rate,
        "num_particles": parameter.num_particles,
        "batch_size": parameter.batch_size,
        "num_iterations": parameter.num_iterations,
        "stopped_at_iteration": parameter.stopped_at_iteration,
        "kernel_length": parameter.kernel_length,
        "warm_up_iterations_early_stopping": parameter.warm_up_iterations_early_stopping,
        "patience_early_stopping": parameter.patience_early_stopping,
        "min_delta_early_stopping": parameter.min_delta_early_stopping,
        "mean_true_output": true_output.mean(),
        "var_true_output": true_output.var(),
        "mean_prediction": test_predictions.mean(),
        "average_var_prediction": test_predictions.var(0).mean(),
        "particle_span_predictions": calculate_mean_span_over_particles(test_predictions),
        "mean_precision": test_precision.mean(),
        "var_precision": test_precision.var(),
        "mse": calculate_mse(test_predictions, true_output)
    }
    df = pd.DataFrame([data])
    file_path = name + "_EvaluationRegression.csv"

    try:
        df_csv = pd.read_csv(file_path)
        pd.concat([df, df_csv], axis=0).to_csv(file_path, index=False)
    except FileNotFoundError:
        df.to_csv(file_path, index=False)


def print_evaluation_multiclass_to_csv(name, parameter, true_output, test_predictions, init_value, decay_rate):
    """
    Saves evaluation metrics for a multiclass classification model into a CSV file. If the file already exists, it appends the data.

    Args:
        name (str): The name of the model or experiment for identifying the file.
        parameter (object): Object containing SVGD training parameters like batch size, num_particles, 
                            kernel length, and early stopping parameters.
        true_output (jax.numpy.ndarray): The true output values (ground truth) for the test set.
        test_predictions (jax.numpy.ndarray): Predicted class probabilities from the SVGD model over particles.
        init_value (float): The initial learning rate.
        decay_rate (float): The decay rate choosen for the optimizer.
    Returns:
        None: The function saves the evaluation to a CSV file.
    """
    most_common_prediction_over_particles = jnp.array(get_most_common_class_over_particles(test_predictions))
    number_of_different_classified_by_particles = jnp.array(
        calculate_number_of_different_classified_by_particles(test_predictions))
    data = {
        "name": name,
        "init_value": init_value,
        "decay_rate": decay_rate,
        "num_particles": parameter.num_particles,
        "batch_size": parameter.batch_size,
        "num_iterations": parameter.num_iterations,
        "stopped_at_iteration": parameter.stopped_at_iteration,
        "kernel_length": parameter.kernel_length,
        "warm_up_iterations_early_stopping": parameter.warm_up_iterations_early_stopping,
        "patience_early_stopping": parameter.patience_early_stopping,
        "min_delta_early_stopping": parameter.min_delta_early_stopping,
        "most_common_true_output": get_most_common_class(true_output),
        "most_common_prediction": get_most_common_class(most_common_prediction_over_particles),
        "number_of_different_classified_by_particles": number_of_different_classified_by_particles.mean(),
        "accuracy": calculate_accuracy(test_predictions, true_output)
    }
    df = pd.DataFrame([data])
    file_path = name + "_EvaluationMulticlass.csv"
    try:
        df_csv = pd.read_csv(file_path)
        pd.concat([df, df_csv], axis=0).to_csv(file_path, index=False)
    except FileNotFoundError:
        df.to_csv(file_path, index=False)
