import jax
import pytest
from unittest.mock import patch

from fixtures import stein_vi_regression_example, stein_vi_multiclass_example, get_regression_toy_example, get_MNIST

from stein_vi.stein_vi import train_with_stein_vi


def test_train_with_stein_vi(get_MNIST, stein_vi_multiclass_example):
    # given
    key = jax.random.PRNGKey(1)
    algorithm = "svgd"
    # when
    train_with_stein_vi(stein_vi_multiclass_example, get_MNIST, key, algorithm=algorithm)
    # then
    assert len(stein_vi_multiclass_example.evaluation_metrics_1) == stein_vi_multiclass_example.parameter.num_iterations
    assert all(item is None for item in stein_vi_multiclass_example.evaluation_metrics_2)


def test_train_with_stein_vi_regression(get_regression_toy_example, stein_vi_regression_example):
    # given
    key = jax.random.PRNGKey(1)
    algorithm = "svgd"
    # when
    train_with_stein_vi(stein_vi_regression_example, get_regression_toy_example, key, algorithm=algorithm)
    # then
    assert len(stein_vi_regression_example.evaluation_metrics_1) == stein_vi_regression_example.parameter.num_iterations
    assert len(stein_vi_regression_example.evaluation_metrics_2) == stein_vi_regression_example.parameter.num_iterations


@patch('stein_vi.stein_vi.train_general_algorithm')
def test_train_with_stein_vi_general_algorithm_is_called(mock_train_algo, get_regression_toy_example, stein_vi_regression_example):
    # given
    key = jax.random.PRNGKey(1)
    algorithm = "svgd"
    # when
    train_with_stein_vi(stein_vi_regression_example, get_regression_toy_example, key, algorithm=algorithm)
    # then
    mock_train_algo.assert_called_once_with(steinvi=stein_vi_regression_example, dataset=get_regression_toy_example, key=key)


def test_train_with_stein_vi_error_for_unsupported_algorithm(get_regression_toy_example, stein_vi_regression_example):
    # given
    key = jax.random.PRNGKey(1)
    _, _, _, _, z_test, y_test = get_regression_toy_example
    # when then
    with pytest.raises(ValueError, match="Unsupported algorithm: unsupported"):
        train_with_stein_vi(stein_vi_regression_example, get_regression_toy_example, key, algorithm="unsupported")
