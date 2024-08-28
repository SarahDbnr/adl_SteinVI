import jax
import jax.numpy as jnp

VAL_SPLIT = 0.1


def apply_data_settings_keras(new_dataset, with_flattening=False, fraction=1):
    (x_train, y_train), (x_test, y_test) = new_dataset
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    if with_flattening:
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    # Split training data into train and validation sets
    val_size = int(len(x_train) * VAL_SPLIT)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    print_data_information(x_train, y_train, x_val, y_val, x_test, y_test)

    if fraction < 1:
        x_train, y_train, x_test, y_test = reduce_size_of_dataframe(fraction, x_train, x_test, y_train, y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def apply_data_settings_sklearn(new_dataset, fraction=1):
    key = jax.random.PRNGKey(1)
    x = new_dataset.data
    y = new_dataset.target
    # Shuffle and split data
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, len(x))
    x, y = x[perm], y[perm]

    num_train = int(0.8 * len(x))
    num_val = int(VAL_SPLIT * num_train)

    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x_train[-num_val:], y_train[-num_val:]
    x_train, y_train = x_train[:-num_val], y_train[:-num_val]
    x_test, y_test = x[num_train:], y[num_train:]

    print_data_information(x_train, y_train, x_val, y_val, x_test, y_test)

    if fraction < 1:
        x_train, y_train, x_test, y_test = reduce_size_of_dataframe(fraction, x_train, x_test, y_train, y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def reduce_size_of_dataframe(fraction, x_train, x_test, y_train, y_test):
    train_sample_size = int(x_train.shape[0] * fraction)
    test_sample_size = int(x_test.shape[0] * fraction)
    train_indices = jnp.random.choice(x_train.shape[0], train_sample_size, replace=False)
    test_indices = jnp.random.choice(x_test.shape[0], test_sample_size, replace=False)
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]
    return x_train, y_train, x_test, y_test


def print_data_information(x_train, y_train, x_val, y_val, x_test, y_test):
    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")
