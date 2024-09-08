from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import jax.numpy as jnp
import tensorflow as tf
import src.data.datasets_info as datasets_info
from src.data.data_handling import apply_data_settings_sklearn, apply_data_settings_keras, newsgroup_datahandling, \
    adult_income_datahandling
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_iris
from src.data.regression_toy_example import get_regression_toy_example
def random_forest(dataset, task_type='classification'):
    """
    Performs Random Forest for a given dataset consisting of a train dataset, a validation dataset, 
    and a test dataset. The datasets can be different. If it is a regression task, it 
    computes the mean squared error (MSE) and precision. For classification, the accuracy is computed.

    The validation set is used both for initial validation and for training the final model.

    Parameters:
    - dataset: A tuple containing the train, validation, and test datasets. The expected format is
               (X_train, y_train, X_val, y_val, X_test, y_test).
    - task_type: A string indicating the type of task ('classification' or 'regression').

    Returns:
    - metrics: A dictionary containing the relevant metrics for the task.
    """

    # Unpack the dataset directly into 6 variables
    X_train, y_train, X_val, y_val, X_test, y_test = dataset

    # Combine training and validation datasets
    X_train_combined = jnp.concatenate((X_train, X_val), axis=0)
    y_train_combined = jnp.concatenate((y_train, y_val), axis=0)
    X_train_combined = X_train_combined.reshape((X_train_combined.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    if task_type == 'classification':
        # Initialize the RandomForestClassifier
        model = RandomForestClassifier()
        # Train the model on the combined training and validation data
        model.fit(X_train_combined, y_train_combined)
        
        # Predict on the validation and test sets
        y_test_pred = model.predict(X_test)

        # Compute accuracy
        accuracy_test = accuracy_score(y_test, y_test_pred)

        # Output metrics
        metrics = {
            'Test Accuracy': accuracy_test
        }
        
    elif task_type == 'regression':
        # Initialize the RandomForestRegressor
        model = RandomForestRegressor()
        # Train the model on the combined training and validation data
        model.fit(X_train_combined, y_train_combined)
        
        # Predict on the validation and test sets
        y_test_pred = model.predict(X_test)
        
        # Compute MSE
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        # Compute precision as standard deviation of predictions
        precision_test = jnp.std(y_test_pred)

        # Output metrics
        metrics = {
            'Test MSE': mse_test,
            'Test Precision': precision_test
        }

    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    return metrics

