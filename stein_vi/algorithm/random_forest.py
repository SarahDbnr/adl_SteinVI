from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import jax.numpy as jnp


def random_forest(dataset, task_type='classification'):
    """
    Trains and evaluates a Random Forest model on the provided dataset for either classification or regression tasks.
    
    Args:
        dataset (tuple): A tuple containing the training, validation, and test datasets. Expected format:
                         (X_train, y_train), (X_val, y_val), (X_test, y_test), where X represents the input data and y represents the labels.
        task_type (str, optional): The type of task to perform, either 'classification' or 'regression'. Defaults to 'classification'.
        
    Returns:
        dict: A dictionary containing the computed performance metrics based on the task type. For classification, this includes:
              - 'Test Accuracy': Accuracy score on the test set.
              For regression, this includes:
              - 'Test MSE': Mean Squared Error on the test set.
              - 'Test Precision': Standard deviation of the predicted values.
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
