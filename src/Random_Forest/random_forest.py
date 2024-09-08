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


def run_MNIST(info=False):
    """_summary_

    Args:
        info (bool, optional): _description_. Defaults to False.
    """    
    if info:
        datasets_info.print_mnist_dataset_info()
    mnist = tf.keras.datasets.mnist
    dataset = apply_data_settings_keras(mnist.load_data(), with_flattening=False)
    print(random_forest(dataset=dataset,task_type="classification"))

def run_FashionMNIST(info=False):
    """_summary_

    Args:
        info (bool, optional): _description_. Defaults to False.
    """    
    if info:
        datasets_info.print_fashion_mnist_dataset_info()

    fashion_mnist = tf.keras.datasets.fashion_mnist
    dataset = apply_data_settings_keras(fashion_mnist.load_data(), with_flattening=False)
    print(random_forest(dataset=dataset,task_type="classification"))
def run_CIFAR10(info=False):
    """_summary_

    Args:
        info (bool, optional): _description_. Defaults to True.
    """    
    if info:
        datasets_info.print_cifar10_dataset_info()
    cifar10 = tf.keras.datasets.cifar10
    dataset = apply_data_settings_keras(cifar10.load_data(), with_flattening=True)
    print(random_forest(dataset=dataset,task_type="classification"))

def run_20_newsgroups(info=False):
    """_summary_

    Args:
        info (bool, optional): _description_. Defaults to True.
    """    
    if info:
        datasets_info.print_20_newsgroups_dataset_info()
    dataset = newsgroup_datahandling()
    dataset = apply_data_settings_sklearn(dataset)
    print(random_forest(dataset=dataset,task_type="classification"))
    


def run_adult_income(info=False):
    """_summary_

    Args:
        info (bool, optional): _description_. Defaults to False.
    """    
    if info:
        datasets_info.print_adult_income_dataset_info()
    dataset = adult_income_datahandling()
    dataset = apply_data_settings_sklearn(dataset)
    print(random_forest(dataset=dataset,task_type="classification"))
    



def run_iris(info=False):
    """_summary_

    Args:
        info (bool, optional): _description_. Defaults to False.
    """    
    if info:
        datasets_info.print_iris_dataset_info()
    iris = load_iris()  # Loading the Iris dataset
    dataset = apply_data_settings_sklearn(iris)
    print(random_forest(dataset=dataset,task_type="classification"))


def run_regression_toy_example(info=False):
    """_summary_

    Args:
        info (bool, optional): _description_. Defaults to False.
    """    
    regression_toy_example = get_regression_toy_example(num_points=10000)
    print(random_forest(dataset=regression_toy_example,task_type="regression"))


def run_california_housing(info=False):
    """_summary_

    Args:
        info (bool, optional): _description_. Defaults to False.
    """    
    if info:
        datasets_info.print_california_housing_dataset_info()
    california_housing = fetch_california_housing()
    dataset = apply_data_settings_sklearn(california_housing)

    print(random_forest(dataset=dataset,task_type="regression"))


def run_diabetes(info=False):
    """_summary_

    Args:
        info (bool, optional): _description_. Defaults to False.
    """    
    if info:
        datasets_info.print_diabetes_dataset_info()
    diabetes = load_diabetes()
    dataset = apply_data_settings_sklearn(diabetes)
    print(random_forest(dataset=dataset,task_type="regression"))
    

def run_wine_quality(info=False):
    """test

    Args:
        info (bool, optional): test. Defaults to False.
    """    
    if info:
        datasets_info.print_wine_quality_dataset_info()
    wine_quality = load_wine()
    dataset = apply_data_settings_sklearn(wine_quality)
    print(random_forest(dataset=dataset,task_type="regression"))
# if __name__ == "__main__":
#     run_CIFAR10()
