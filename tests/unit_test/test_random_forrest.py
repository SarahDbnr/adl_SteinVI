from tests.unit_test.fixtures import get_regression_toy_example, get_MNIST
from stein_vi.algorithm.random_forest import random_forest  


def test_random_forest_classification(get_MNIST):
    """Test the random_forest function for classification task."""

    # when
    metrics = random_forest(get_MNIST, task_type='classification')
    # then
    accuracy = metrics['Test Accuracy']
    assert 'Test Accuracy' in metrics, "The output metrics should contain 'Test Accuracy'"
    assert 0.5 <= accuracy <= 1.0, f"Accuracy should be between 0.5 and 1.0, but got {accuracy}, so it is worse than random guessing "


def test_random_forest_regression(get_regression_toy_example):
    """Test the random_forest function for regression task."""

    # when
    metrics = random_forest(get_regression_toy_example, task_type='regression')
    
    # then
    mse = metrics['Test MSE']
    precision = metrics['Test Precision']
    assert 'Test MSE' in metrics, "The output metrics should contain 'Test MSE'"
    assert 'Test Precision' in metrics, "The output metrics should contain 'Test Precision'"
    assert mse >= 0, f"MSE should be non-negative, but got {mse}"
    assert precision >= 0, f"Precision (std of predictions) should be non-negative, but got {precision}"
