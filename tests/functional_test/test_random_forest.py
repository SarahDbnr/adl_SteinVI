import pytest
import jax
from sklearn.datasets import load_diabetes
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, mean_squared_error
from optax import adam, exponential_decay

from stein_vi.Classes.SteinVI_BNN import SteinVI_BNN
from run_stein_vi.model.BNN_Model import build_model
from stein_vi.stein_vi import train_with_stein_vi
from stein_vi.algorithm.random_forest import random_forest
from run_stein_vi.data.data_handling import apply_data_settings_keras, apply_data_settings_sklearn

    

@pytest.mark.filterwarnings("ignore::DeprecationWarning:jax")
def test_random_forest_vs_svgd_mnist():
    """Test Random Forest vs SVGD on the MNIST dataset for classification."""
    mnist_data = mnist.load_data()
    mnist_dataset = apply_data_settings_keras(mnist_data, with_flattening=False)

    # Unpack the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = mnist_dataset

    # Combine training and validation for Random Forest
    X_train_combined = jax.numpy.concatenate([X_train, X_val])
    y_train_combined = jax.numpy.concatenate([y_train, y_val])

    # Random Forest
    rf_metrics = random_forest((X_train, y_train, X_val, y_val, X_test, y_test), task_type='classification')
    rf_accuracy = rf_metrics['Test Accuracy']

    # SVGD with SteinVI_BNN
    key = jax.random.PRNGKey(1)
    optimizer = adam(
        exponential_decay(
            init_value=0.05,
            transition_steps=100,
            decay_rate=0.95,
            staircase=True
        )
    )
    
    # Create the BNN model using SteinVI
    nnet_model = build_model(output_size=10, hidden_layers=(200, 70, 40))
    steinvi_svdg = SteinVI_BNN(key, X_train_combined, nnet_model, use_for_regression=False, optimizer=optimizer, num_iterations=30, batch_size=300, num_particles=5)
    
    # Train with Stein VI using SVGD
    train_with_stein_vi(steinvi_svdg, (X_train, y_train, X_val, y_val, X_test, y_test), key, algorithm="svgd")
    
    # Make predictions using the BNN model
    _, y_pred_svgd = steinvi_svdg.predict_over_particles(X_test)
    y_pred_svgd = jax.numpy.argmax(jax.numpy.mean(y_pred_svgd, axis=0), axis=-1)
    
    # Calculate accuracy for SVGD
    svgd_accuracy = accuracy_score(y_test.squeeze(), y_pred_svgd)

    # Compare results
    assert svgd_accuracy >= rf_accuracy - 0.1, \
        f"SVGD accuracy ({svgd_accuracy}) should be within 10% of Random Forest accuracy ({rf_accuracy})"


@pytest.mark.filterwarnings("ignore::DeprecationWarning:jax")
def test_random_forest_vs_svgd_diabetes():
    """Test Random Forest vs SVGD on the Diabetes dataset for regression."""
    diabetes_data = load_diabetes()
    diabetes_dataset = apply_data_settings_sklearn(diabetes_data)

    # Unpack the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = diabetes_dataset

    # Combine training and validation for Random Forest
    X_train_combined = jax.numpy.concatenate([X_train, X_val])
    y_train_combined = jax.numpy.concatenate([y_train, y_val])

    # Random Forest
    rf_metrics = random_forest((X_train, y_train, X_val, y_val, X_test, y_test), task_type='regression')
    rf_mse = rf_metrics['Test MSE']

    # SVGD with SteinVI_BNN
    key = jax.random.PRNGKey(1)
    optimizer = adam(
        exponential_decay(
            init_value=0.01,
            transition_steps=50,
            decay_rate=0.95,
            staircase=True
        )
    )
    
    # Create the BNN model using SteinVI
    nnet_model = build_model(output_size=2, hidden_layers=(200, 70, 40))
    steinvi_svdg = SteinVI_BNN(key, X_train_combined, nnet_model, use_for_regression=True, optimizer=optimizer, num_iterations=100, num_particles=5)
    
    # Train with Stein VI using SVGD
    train_with_stein_vi(steinvi_svdg, (X_train, y_train, X_val, y_val, X_test, y_test), key, algorithm="svgd")
    
    # Make predictions using the BNN model
    y_pred_svgd, _ = steinvi_svdg.predict_over_particles(X_test)
    y_pred_svgd = jax.numpy.mean(y_pred_svgd, axis=0)

    # Calculate MSE for SVGD
    svgd_mse = mean_squared_error(y_test, y_pred_svgd)

    # Compare results
    assert svgd_mse <= rf_mse + 0.1 * rf_mse, \
        f"SVGD MSE ({svgd_mse}) should be within 10% of Random Forest MSE ({rf_mse})"

