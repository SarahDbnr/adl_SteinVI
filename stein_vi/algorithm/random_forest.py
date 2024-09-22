from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import jax.numpy as jnp


def random_forest(dataset, task_type='classification'):
    """
    Trains and evaluates a Random Forest model on the provided dataset for either classification or regression tasks.
    
    Args:
        dataset (tuple): A tuple containing the training, validation, and test datasets. Expected format:
                         (X_train, y_train), (X_val, y_val), (X_test, y_test), where X represents
                         the input data and y represents the labels.
        task_type (str, optional): The type of task to perform, either 'classification' or 'regression'.
        Defaults to 'classification'.
        
    Returns:
        dict: A dictionary containing the computed performance metrics based on the task type.
        For classification, this includes:
              'Test Accuracy': Accuracy score on the test set.
              For regression, this includes:
              'Test MSE': Mean Squared Error on the test set.
              'Test Precision': Standard deviation of the predicted values.
    """

    x_train_combined, y_train_combined, x_test, y_test = prepare_dataset(dataset)

    if task_type == 'classification':

        model = RandomForestClassifier()
        model.fit(x_train_combined, y_train_combined)

        y_test_pred = model.predict(x_test)

        accuracy_test = accuracy_score(y_test, y_test_pred)

        metrics = {
            'Test Accuracy': accuracy_test
        }
        
    elif task_type == 'regression':

        model = RandomForestRegressor()
        model.fit(x_train_combined, y_train_combined)

        y_test_pred = model.predict(x_test)

        mse_test = mean_squared_error(y_test, y_test_pred)

        precision_test = jnp.std(y_test_pred)

        metrics = {
            'Test MSE': mse_test,
            'Test Precision': precision_test
        }

    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    return metrics


def prepare_dataset(dataset):
    """
    prepares the data for random forest:
    
    Args:
        dataset (tuple): A tuple containing the training, validation, and test datasets. Expected format:
                         (X_train, y_train), (X_val, y_val), (X_test, y_test), where X represents the input data
                         and y represents the labels.
        
    Returns:
        tuple: A tuple containing x_train, y_train, x_test,y_test
    """
    x_train, y_train, x_val, y_val, x_test, y_test = dataset

    x_train_combined = jnp.concatenate((x_train, x_val), axis=0)
    y_train_combined = jnp.concatenate((y_train, y_val), axis=0)
    x_train_combined = x_train_combined.reshape((x_train_combined.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))
    return x_train_combined, y_train_combined, x_test, y_test
