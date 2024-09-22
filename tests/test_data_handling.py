import pytest
import jax.numpy as jnp
import jax
from run_stein_vi.data.data_handling import apply_data_settings_keras, apply_data_settings_sklearn, print_data_information
from sklearn.utils import Bunch


def test_apply_data_settings_keras_normalization():
    """Test that the input data is normalized to the range [0, 1]."""
    # given
    x_train = jnp.array([
        [[[0], [128], [255]], [[0], [128], [255]], [[0], [128], [255]]],[[[255], [128], [0]], [[255], [128], [0]], [[255], [128], [0]]]
    ], dtype='uint8')  

    x_test = jnp.array([
        [[[0], [255], [128]], [[128], [255], [0]], [[255], [0], [128]]]
    ], dtype='uint8')  

    y_train = jnp.array([0, 1]) 
    y_test = jnp.array([1])  

    dataset = ((x_train, y_train), (x_test, y_test))

    # when
    x_train_proc, _, _, _, x_test_proc, _ = apply_data_settings_keras(dataset, val_split=0.5)

    # then
    assert x_train_proc.size > 0, "Training data should not be empty"
    assert x_test_proc.size > 0, "Test data should not be empty"
    
    assert x_train_proc.max() <= 1.0 and x_train_proc.min() >= 0.0, "Training data should be normalized"
    assert x_test_proc.max() <= 1.0 and x_test_proc.min() >= 0.0, "Test data should be normalized"


def test_apply_data_settings_keras_validation_split():
    """Test that the validation split is correctly applied."""
    # given
    x_train = jnp.ones((10, 28, 28, 1))  
    x_test = jnp.ones((2, 28, 28, 1))    
    y_train = jnp.arange(10) 
    y_test = jnp.arange(2)   

    dataset = ((x_train, y_train), (x_test, y_test))
    val_split = 0.2
    # when
    x_train_proc, _, x_val_proc, _, _, _ = apply_data_settings_keras(dataset, val_split=val_split)

    # then 
    assert len(x_val_proc) == int(10 * val_split), "Validation set should have 20% of the training data"
    assert len(x_train_proc) == int(10 * (1 - val_split)), "Training set should have 80% of the data after splitting"


def test_apply_data_settings_keras_flattening():
    """Test that the labels are correctly flattened when with_flattening=True."""
    # given 
    x_train = jnp.ones((10, 28, 28, 1))  
    x_test = jnp.ones((2, 28, 28, 1))    
    y_train = jnp.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])  # Labels 0-9, not yet flattened
    y_test = jnp.array([[0], [1]])       

    dataset = ((x_train, y_train), (x_test, y_test))

    # when
    _, y_train_proc, _, _, _, y_test_proc = apply_data_settings_keras(dataset, with_flattening=True)

    # then
    assert len(y_train_proc.shape) == 1, "Training labels should be flattened"
    assert len(y_test_proc.shape) == 1, "Test labels should be flattened"


def test_apply_data_settings_keras_fraction():
    """Test that the dataset size is reduced correctly when using the fraction parameter."""
    # given
    x_train = jnp.ones((100, 28, 28, 1))  
    x_test = jnp.ones((20, 28, 28, 1))    
    y_train = jnp.arange(100) 
    y_test = jnp.arange(20)   

    dataset = ((x_train, y_train), (x_test, y_test))
    fraction = 0.5

    # when
    x_train_proc, _, _, _, x_test_proc, _ = apply_data_settings_keras(dataset,val_split=0.1, fraction=fraction)

    # then
    assert len(x_train_proc) == int((100 * fraction)*0.9), "Training set should be reduced by the fraction"
    assert len(x_test_proc) == int(20 * fraction), "Test set should be reduced by the fraction"


def test_apply_data_settings_keras_output_structure():
    """Test that the output structure is correct (6 elements, with proper shapes)."""
    # given
    x_train =jnp.ones((100, 28, 28, 1))  
    x_test =jnp.ones((20, 28, 28, 1))    
    y_train =jnp.arange(100)  
    y_test =jnp.arange(20)    

    dataset = ((x_train, y_train), (x_test, y_test))

    # when
    x_train_proc, y_train_proc, x_val_proc, y_val_proc, x_test_proc, y_test_proc = apply_data_settings_keras(dataset)

    # then
    assert isinstance(x_train_proc,jnp.ndarray), "x_train_proc should be a numpy array"
    assert isinstance(x_val_proc,jnp.ndarray), "x_val_proc should be a numpy array"
    assert isinstance(x_test_proc,jnp.ndarray), "x_test_proc should be a numpy array"
    assert isinstance(y_train_proc,jnp.ndarray), "y_train_proc should be a numpy array"
    assert isinstance(y_val_proc,jnp.ndarray), "y_val_proc should be a numpy array"
    assert isinstance(y_test_proc,jnp.ndarray), "y_test_proc should be a numpy array"

    assert x_train_proc.shape[0] > 0, "x_train_proc should have elements"
    assert x_val_proc.shape[0] > 0, "x_val_proc should have elements"
    assert x_test_proc.shape[0] == x_test.shape[0], "Test set should retain its original size"


def mock_sklearn_dataset():
    """Creates a mock sklearn-style dataset."""
    key = jax.random.PRNGKey(0)

    x = jax.random.normal(key, (100, 10))
    y = jax.random.randint(key, shape=(100,), minval=0, maxval=2)

    x = jnp.array(x)
    y = jnp.array(y)

    return Bunch(data=x, target=y)


def test_apply_data_settings_sklearn_output_format():
    """Test that the output format is correct."""
    # given
    dataset = mock_sklearn_dataset()
    # when
    x_train, y_train, x_val, y_val, x_test, y_test = apply_data_settings_sklearn(dataset)

    # then
    assert isinstance(x_train, jnp.ndarray), "x_train should be a jax.numpy array"
    assert isinstance(x_val, jnp.ndarray), "x_val should be a jax.numpy array"
    assert isinstance(x_test, jnp.ndarray), "x_test should be a jax.numpy array"
    assert isinstance(y_train, jnp.ndarray), "y_train should be a jax.numpy array"
    assert isinstance(y_val, jnp.ndarray), "y_val should be a jax.numpy array"
    assert isinstance(y_test, jnp.ndarray), "y_test should be a jax.numpy array"

    assert len(x_train) == 72, "Training data should contain 72 samples"
    assert len(x_val) == 8, "Validation data should contain 8 samples"
    assert len(x_test) == 20, "Test data should contain 20 samples"


def test_apply_data_settings_sklearn_fraction():
    """Test that the dataset size is reduced correctly when using the fraction parameter."""
    # given
    dataset = mock_sklearn_dataset()
    fraction = 0.5
    # when
    x_train, _, _, _, x_test, _ = apply_data_settings_sklearn(dataset, fraction=fraction)

    # then
    assert len(x_train) == int(72 * fraction), "Training set should be reduced by the fraction"
    assert len(x_test) == int(20 * fraction), "Test set should be reduced by the fraction"


def test_apply_data_settings_sklearn_validation_split():
    """Test that the validation split is correctly applied."""
    # given
    dataset = mock_sklearn_dataset()

    val_split = 0.2
    # when 
    x_train, y_train, x_val, y_val, x_test, y_test = apply_data_settings_sklearn(dataset, val_split=val_split)    
    expected_train_size = int(100 *0.8 * (1 - val_split))
    expected_val_size = int(100 * 0.8 * val_split)
    # then
    assert len(x_train) == expected_train_size, f"Training set should have {expected_train_size} samples"
    assert len(x_val) == expected_val_size, f"Validation set should have {expected_val_size} samples"


def test_apply_data_settings_sklearn_shuffling():
    """Test that the dataset is shuffled using the provided random key."""
    # given
    dataset = mock_sklearn_dataset()
    # when
    key = jax.random.PRNGKey(0)
    x_train1, y_train1, x_val1, y_val1, x_test1, y_test1 = apply_data_settings_sklearn(dataset, key=key)

    key = jax.random.PRNGKey(0)
    x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = apply_data_settings_sklearn(dataset, key=key)
    # then
    assert jnp.array_equal(x_train1, x_train2), "Shuffling with the same key should produce the same training data"
    assert jnp.array_equal(y_train1, y_train2), "Shuffling with the same key should produce the same training labels"
    assert jnp.array_equal(x_test1, x_test2), "Shuffling with the same key should produce the same test data"
    assert jnp.array_equal(y_test1, y_test2), "Shuffling with the same key should produce the same test labels"

    # when
    key = jax.random.PRNGKey(1)
    x_train3, y_train3, x_val3, y_val3, x_test3, y_test3 = apply_data_settings_sklearn(dataset, key=key)

    # then
    assert not jnp.array_equal(x_train1, x_train3), "Shuffling with different keys should produce different training data"
    assert not jnp.array_equal(y_train1, y_train3), "Shuffling with different keys should produce different training labels"
    assert not jnp.array_equal(x_test1, x_test3), "Shuffling with different keys should produce different test data"
    assert not jnp.array_equal(y_test1, y_test3), "Shuffling with different keys should produce different test labels"


def test_print_data_information_basic(capfd):
    """Test the basic functionality of print_data_information by checking the printed shapes."""
    # given
    key = jax.random.PRNGKey(0)

    x_train = jax.random.normal(key, (80, 10)) 
    y_train = jax.random.randint(key, minval=0, maxval=2, shape=(80,))  

    x_val = jax.random.normal(key, (10, 10))  
    y_val = jax.random.randint(key, minval=0, maxval=2, shape=(10,))  

    x_test = jax.random.normal(key, (20, 10))  
    y_test = jax.random.randint(key, minval=0, maxval=2, shape=(20,))  
    # when
    print_data_information(x_train, y_train, x_val, y_val, x_test, y_test)

    captured = capfd.readouterr()


    expected_output = (
        f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}\n"
        f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}\n"
        f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}\n"
    )

    # then
    assert captured.out == expected_output, "The printed output does not match the expected shapes."