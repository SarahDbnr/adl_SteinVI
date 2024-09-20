import pytest
import jax.numpy as jnp
from run_stein_vi.data.data_handling import apply_data_settings_keras, apply_data_settings_sklearn, print_data_information
from sklearn.utils import Bunch


def test_apply_data_settings_keras_normalization():
    """Test that the input data is normalized to the range [0, 1]."""
    # Create mock Keras-style dataset (3 samples, 3x3 images, 1 channel)
    x_train = jnp.array([
        [[[0], [128], [255]], [[0], [128], [255]], [[0], [128], [255]]],[[[255], [128], [0]], [[255], [128], [0]], [[255], [128], [0]]]
    ], dtype='uint8')  

    x_test = jnp.array([
        [[[0], [255], [128]], [[128], [255], [0]], [[255], [0], [128]]]
    ], dtype='uint8')  

    y_train = jnp.array([0, 1])  # Two training labels
    y_test = jnp.array([1])  

    dataset = ((x_train, y_train), (x_test, y_test))

    # Apply data settings
    x_train_proc, y_train_proc, x_val_proc, y_val_proc, x_test_proc, y_test_proc = apply_data_settings_keras(dataset, val_split=0.5)

    # Check if the arrays are not empty before applying max/min operations
    assert x_train_proc.size > 0, "Training data should not be empty"
    assert x_test_proc.size > 0, "Test data should not be empty"
    
    # Check if data is normalized
    assert x_train_proc.max() <= 1.0 and x_train_proc.min() >= 0.0, "Training data should be normalized"
    assert x_test_proc.max() <= 1.0 and x_test_proc.min() >= 0.0, "Test data should be normalized"


def test_apply_data_settings_keras_validation_split():
    """Test that the validation split is correctly applied."""
    # Create mock dataset
    x_train = jnp.ones((10, 28, 28, 1))  
    x_test = jnp.ones((2, 28, 28, 1))    
    y_train = jnp.arange(10) 
    y_test = jnp.arange(2)   

    dataset = ((x_train, y_train), (x_test, y_test))

    # Apply data settings with a 20% validation split
    val_split = 0.2
    x_train_proc, y_train_proc, x_val_proc, y_val_proc, x_test_proc, y_test_proc = apply_data_settings_keras(dataset, val_split=val_split)

    # Check the size of the validation and training sets
    assert len(x_val_proc) == int(10 * val_split), "Validation set should have 20% of the training data"
    assert len(x_train_proc) == int(10 * (1 - val_split)), "Training set should have 80% of the data after splitting"


def test_apply_data_settings_keras_flattening():
    """Test that the labels are correctly flattened when with_flattening=True."""
    # Create mock dataset
    x_train = jnp.ones((10, 28, 28, 1))  
    x_test = jnp.ones((2, 28, 28, 1))    
    y_train = jnp.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])  # Labels 0-9, not yet flattened
    y_test = jnp.array([[0], [1]])       

    dataset = ((x_train, y_train), (x_test, y_test))

    # Apply data settings with flattening
    x_train_proc, y_train_proc, x_val_proc, y_val_proc, x_test_proc, y_test_proc = apply_data_settings_keras(dataset, with_flattening=True)

    # Check if labels are flattened
    assert len(y_train_proc.shape) == 1, "Training labels should be flattened"
    assert len(y_test_proc.shape) == 1, "Test labels should be flattened"


def test_apply_data_settings_keras_fraction():
    """Test that the dataset size is reduced correctly when using the fraction parameter."""
    # Create mock dataset
    x_train = jnp.ones((100, 28, 28, 1))  
    x_test = jnp.ones((20, 28, 28, 1))    
    y_train = jnp.arange(100) 
    y_test = jnp.arange(20)   

    dataset = ((x_train, y_train), (x_test, y_test))

    # Apply data settings with a fraction of 0.5 (half the data)
    fraction = 0.5
    x_train_proc, y_train_proc, x_val_proc, y_val_proc, x_test_proc, y_test_proc = apply_data_settings_keras(dataset,val_split=0.1, fraction=fraction)

    # Check the reduced size
    assert len(x_train_proc) == int((100 * fraction)*0.9), "Training set should be reduced by the fraction"
    assert len(x_test_proc) == int(20 * fraction), "Test set should be reduced by the fraction"


def test_apply_data_settings_keras_output_structure():
    """Test that the output structure is correct (6 elements, with proper shapes)."""
    # Create mock dataset
    x_train =jnp.ones((100, 28, 28, 1))  
    x_test =jnp.ones((20, 28, 28, 1))    
    y_train =jnp.arange(100)  
    y_test =jnp.arange(20)    

    dataset = ((x_train, y_train), (x_test, y_test))

    # Apply data settings
    x_train_proc, y_train_proc, x_val_proc, y_val_proc, x_test_proc, y_test_proc = apply_data_settings_keras(dataset)

    # Check if the output is a tuple of length 6
    assert isinstance(x_train_proc,jnp.ndarray), "x_train_proc should be a numpy array"
    assert isinstance(x_val_proc,jnp.ndarray), "x_val_proc should be a numpy array"
    assert isinstance(x_test_proc,jnp.ndarray), "x_test_proc should be a numpy array"
    assert isinstance(y_train_proc,jnp.ndarray), "y_train_proc should be a numpy array"
    assert isinstance(y_val_proc,jnp.ndarray), "y_val_proc should be a numpy array"
    assert isinstance(y_test_proc,jnp.ndarray), "y_test_proc should be a numpy array"

    # Check that training, validation, and test sets have the expected shape
    assert x_train_proc.shape[0] > 0, "x_train_proc should have elements"
    assert x_val_proc.shape[0] > 0, "x_val_proc should have elements"
    assert x_test_proc.shape[0] == x_test.shape[0], "Test set should retain its original size"


import jax

def mock_sklearn_dataset():
    """Creates a mock sklearn-style dataset."""
    # JAX random key
    key = jax.random.PRNGKey(0)

    # Example dataset with 100 samples and 10 features
    x = jax.random.normal(key, (100, 10))  # Generates 100 samples with 10 features
    y = jax.random.randint(key, shape=(100,), minval=0, maxval=2)  # Binary labels (0 or 1)

    # Convert the JAX arrays to numpy arrays (for compatibility with Bunch)
    x = jnp.array(x)
    y = jnp.array(y)

    return Bunch(data=x, target=y)
def test_apply_data_settings_sklearn_output_format():
    """Test that the output format is correct."""
    dataset = mock_sklearn_dataset()

    x_train, y_train, x_val, y_val, x_test, y_test = apply_data_settings_sklearn(dataset)

    # Ensure output is of correct type and length
    assert isinstance(x_train, jnp.ndarray), "x_train should be a jax.numpy array"
    assert isinstance(x_val, jnp.ndarray), "x_val should be a jax.numpy array"
    assert isinstance(x_test, jnp.ndarray), "x_test should be a jax.numpy array"
    assert isinstance(y_train, jnp.ndarray), "y_train should be a jax.numpy array"
    assert isinstance(y_val, jnp.ndarray), "y_val should be a jax.numpy array"
    assert isinstance(y_test, jnp.ndarray), "y_test should be a jax.numpy array"

    # Check that the shapes match the expected split (80% train, 20% test, and val_split from train)
    assert len(x_train) == 72, "Training data should contain 72 samples"
    assert len(x_val) == 8, "Validation data should contain 8 samples"
    assert len(x_test) == 20, "Test data should contain 20 samples"



def test_apply_data_settings_sklearn_fraction():
    """Test that the dataset size is reduced correctly when using the fraction parameter."""
    dataset = mock_sklearn_dataset()
    fraction = 0.5

    x_train, y_train, x_val, y_val, x_test, y_test = apply_data_settings_sklearn(dataset, fraction=fraction)

    # Check the reduced size
    assert len(x_train) == int(72 * fraction), "Training set should be reduced by the fraction"
    assert len(x_test) == int(20 * fraction), "Test set should be reduced by the fraction"

def test_apply_data_settings_sklearn_validation_split():
    """Test that the validation split is correctly applied."""
    dataset = mock_sklearn_dataset()

    val_split = 0.2
    x_train, y_train, x_val, y_val, x_test, y_test = apply_data_settings_sklearn(dataset, val_split=val_split)

    # Check the size of the validation and training sets
    expected_train_size = int(100 *0.8 * (1 - val_split))
    expected_val_size = int(100 * 0.8 * val_split)

    assert len(x_train) == expected_train_size, f"Training set should have {expected_train_size} samples"
    assert len(x_val) == expected_val_size, f"Validation set should have {expected_val_size} samples"
test_apply_data_settings_sklearn_validation_split()




def test_apply_data_settings_sklearn_shuffling():
    """Test that the dataset is shuffled using the provided random key."""
    dataset = mock_sklearn_dataset()

    # Apply the function with a fixed random key
    key = jax.random.PRNGKey(0)
    x_train1, y_train1, x_val1, y_val1, x_test1, y_test1 = apply_data_settings_sklearn(dataset, key=key)

    # Apply again with the same key, the result should be the same
    key = jax.random.PRNGKey(0)
    x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = apply_data_settings_sklearn(dataset, key=key)

    # Check if shuffling with the same key gives the same results
    assert jnp.array_equal(x_train1, x_train2), "Shuffling with the same key should produce the same training data"
    assert jnp.array_equal(y_train1, y_train2), "Shuffling with the same key should produce the same training labels"
    assert jnp.array_equal(x_test1, x_test2), "Shuffling with the same key should produce the same test data"
    assert jnp.array_equal(y_test1, y_test2), "Shuffling with the same key should produce the same test labels"

    # Apply with a different key, the result should be different
    key = jax.random.PRNGKey(1)
    x_train3, y_train3, x_val3, y_val3, x_test3, y_test3 = apply_data_settings_sklearn(dataset, key=key)

    # Check if different key gives different results
    assert not jnp.array_equal(x_train1, x_train3), "Shuffling with different keys should produce different training data"
    assert not jnp.array_equal(y_train1, y_train3), "Shuffling with different keys should produce different training labels"
    assert not jnp.array_equal(x_test1, x_test3), "Shuffling with different keys should produce different test data"
    assert not jnp.array_equal(y_test1, y_test3), "Shuffling with different keys should produce different test labels"



def test_print_data_information_basic(capfd):
    """Test the basic functionality of print_data_information by checking the printed shapes."""
    key = jax.random.PRNGKey(0)

    # Mock dataset with 80 samples for training, 10 for validation, and 20 for testing
    x_train = jax.random.normal(key, (80, 10))  # 80 samples, 10 features
    y_train = jax.random.randint(key, minval=0, maxval=2, shape=(80,))  # Binary labels for training

    x_val = jax.random.normal(key, (10, 10))  # 10 samples, 10 features
    y_val = jax.random.randint(key, minval=0, maxval=2, shape=(10,))  # Binary labels for validation

    x_test = jax.random.normal(key, (20, 10))  # 20 samples, 10 features
    y_test = jax.random.randint(key, minval=0, maxval=2, shape=(20,))  # Binary labels for testing

    # Call the function and capture the printed output
    print_data_information(x_train, y_train, x_val, y_val, x_test, y_test)

    # Capture the stdout
    captured = capfd.readouterr()

    # Expected output
    expected_output = (
        f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}\n"
        f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}\n"
        f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}\n"
    )

    # Check if the output matches the expected output
    assert captured.out == expected_output, "The printed output does not match the expected shapes."