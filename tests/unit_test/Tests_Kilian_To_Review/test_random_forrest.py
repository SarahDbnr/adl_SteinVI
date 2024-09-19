import pytest
import jax.numpy as jnp
import numpy as np
from tests.unit_test.fixtures import synthetic_classification_dataset, synthetic_regression_dataset
from stein_vi.algorithm.random_forest import random_forest  


def test_random_forest_classification(synthetic_classification_dataset):
    """Test the random_forest function for classification task."""
    
    # Get the synthetic classification dataset
    dataset = synthetic_classification_dataset
    
    # Call the random_forest function for classification
    metrics = random_forest(dataset, task_type='classification')
    
    # Check if the metrics dictionary has 'Test Accuracy'
    assert 'Test Accuracy' in metrics, "The output metrics should contain 'Test Accuracy'"
    
    # Ensure the accuracy is within a reasonable range
    accuracy = metrics['Test Accuracy']
    assert 0.5 <= accuracy <= 1.0, f"Accuracy should be between 0.5 and 1.0, but got {accuracy}, so it is worse than random guessing "


def test_random_forest_regression(synthetic_regression_dataset):
    """Test the random_forest function for regression task."""
    
    # Get the synthetic regression dataset
    dataset = synthetic_regression_dataset
    
    # Call the random_forest function for regression
    metrics = random_forest(dataset, task_type='regression')
    
    # Check if the metrics dictionary has 'Test MSE' and 'Test Precision'
    assert 'Test MSE' in metrics, "The output metrics should contain 'Test MSE'"
    assert 'Test Precision' in metrics, "The output metrics should contain 'Test Precision'"
    
    # Ensure MSE is non-negative and precision is a reasonable value
    mse = metrics['Test MSE']
    precision = metrics['Test Precision']
    
    assert mse >= 0, f"MSE should be non-negative, but got {mse}"
    assert precision >= 0, f"Precision (std of predictions) should be non-negative, but got {precision}"


