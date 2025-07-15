import jax
import jax.numpy as jnp


def apply_data_settings_keras(new_dataset, with_flattening=False, fraction=1, val_split=0.1, key=jax.random.PRNGKey(1)):
    """
    Processes a dataset obtained from Keras, normalizing and optionally flattening it, and splitting it into training,
    validation, and test sets. Used for image data with values between 0 and 255.

    Args:
        new_dataset (tuple): A tuple containing training and test datasets as (x_train, y_train), (x_test, y_test).
        with_flattening (bool): If True, flattens the output arrays.
        fraction (float): Fraction of the data to use for reducing dataset size.
        val_split (float): Fraction of the data used for validation during training.
        key (jax.random.PRNGKey): A JAX PRNG key used for deterministic selection of samples.
    Returns:
        tuple: Tuple containing processed training, validation, and test data as (x_train, y_train, x_val, y_val,
        x_test, y_test).
    """

    (x_train, y_train), (x_test, y_test) = new_dataset
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    if with_flattening:
        y_train = y_train.flatten()
        y_test = y_test.flatten()

    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    print_data_information(x_train, y_train, x_val, y_val, x_test, y_test)

    if fraction < 1:
        x_train, y_train, x_test, y_test = reduce_size_of_dataframe(fraction, x_train, x_test, y_train, y_test, key)

    return x_train, y_train, x_val, y_val, x_test, y_test


def apply_data_settings_sklearn(new_dataset, fraction=1, val_split=0.1, key=jax.random.PRNGKey(1),
                                train_test_split=0.8):
    """
    Processes a dataset obtained from scikit-learn, shuffling and splitting it into training, validation, and test sets,
    and optionally reducing its size.

    Args:
        new_dataset (Bunch): An object containing 'data' and 'target' attributes, typical of scikit-learn datasets.
        fraction (float): Fraction of the data to use for reducing dataset size.
        val_split (float): Fraction of the data used for validation during training.
        key (jax.random.PRNGKey): A JAX PRNG key used for deterministic selection of samples.
        train_test_split (float): Fraction of the data used for training
    Returns:
        tuple: Tuple containing processed training, validation, and test data as (x_train, y_train, x_val, y_val,
        x_test, y_test).
    """

    x = new_dataset.data
    y = new_dataset.target

    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, x.shape[0])
    x, y = x[perm], y[perm]

    num_train = int(train_test_split * x.shape[0])
    num_val = int(val_split * num_train)

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x_train[-num_val:], y_train[-num_val:]
    x_train, y_train = x_train[:-num_val], y_train[:-num_val]
    x_test, y_test = x[num_train:], y[num_train:]

    print_data_information(x_train, y_train, x_val, y_val, x_test, y_test)

    if fraction < 1:
        x_train, y_train, x_test, y_test = reduce_size_of_dataframe(fraction, x_train, x_test, y_train, y_test, key)

    return x_train, y_train, x_val, y_val, x_test, y_test


def reduce_size_of_dataframe(fraction, x_train, x_test, y_train, y_test, key):
    """
    Reduces the size of training and test datasets to a specified fraction by random sampling.

    Args:
        fraction (float): Fraction of the original size to retain.
        x_train (jnp.ndarray): Training input data.
        x_test (jnp.ndarray): Test input data.
        y_train (jnp.ndarray): Training labels.
        y_test (jnp.ndarray): Test labels.
        key (jax.random.PRNGKey): A JAX PRNG key used for deterministic selection of samples.

    Returns:
        tuple: Tuple containing the reduced training and test data as (x_train, y_train, x_test, y_test).
    """
    key, subkey = jax.random.split(key)
    train_sample_size = int(x_train.shape[0] * fraction)
    test_sample_size = int(x_test.shape[0] * fraction)
    train_indices = jax.random.choice(subkey, jnp.arange(x_train.shape[0]), (train_sample_size,), replace=False)
    test_indices = jax.random.choice(subkey, jnp.arange(x_test.shape[0]), (test_sample_size,), replace=False)
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]
    return x_train, y_train, x_test, y_test


def print_data_information(x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Prints the shapes of training, validation, and test data and labels.

    Args:
        x_train (jnp.ndarray): Training input data.
        y_train (jnp.ndarray): Training labels.
        x_val (jnp.ndarray): Validation input data.
        y_val (jnp.ndarray): Validation labels.
        x_test (jnp.ndarray): Test input data.
        y_test (jnp.ndarray): Test labels.
    """
    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")
