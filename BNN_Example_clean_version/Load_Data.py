import tensorflow as tf
import jax.numpy as jnp
import jax
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine

VAL_SPLIT = 0.1
FRACTION = 0.1


def load_real_world_data(dataset_name):
    if dataset_name == 'california_housing':
        dataset = fetch_california_housing()
    elif dataset_name == 'diabetes':
        dataset = load_diabetes()
    elif dataset_name == "wine_quality":
        dataset = load_wine()
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    x = dataset.data
    y = dataset.target

    return x, y


def load_data(dataset, reduce_size=False):
    key = jax.random.PRNGKey(0)

    if dataset == "MNIST":
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        # Split training data into train and validation sets
        val_size = int(len(x_train) * VAL_SPLIT)
        x_val, y_val = x_train[-val_size:], y_train[-val_size:]
        x_train, y_train = x_train[:-val_size], y_train[:-val_size]
    elif dataset == "FashionMNIST":
        # Load the Fashion MNIST dataset
        fashion_mnist = tf.keras.datasets.fashion_mnist
        # Separate it into train and test sets
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        # Split training data into train and validation sets
        val_size = int(len(x_train) * VAL_SPLIT)
        x_val, y_val = x_train[-val_size:], y_train[-val_size:]
        x_train, y_train = x_train[:-val_size], y_train[:-val_size]
    elif dataset == "CIFAR10":
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        # Split training data into train and validation sets
        val_size = int(len(x_train) * VAL_SPLIT)
        x_val, y_val = x_train[-val_size:], y_train[-val_size:]
        x_train, y_train = x_train[:-val_size], y_train[:-val_size]
    elif dataset == "california_housing" or dataset == "diabetes" or dataset == "wine_quality":
        x, y = load_real_world_data(dataset)
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

    else:
        raise ValueError(f"Dataset {dataset} not recognized. Available options: 'MNIST', 'FashionMNIST', 'CIFAR-10'")

    if reduce_size:
        x_train, y_train, x_test, y_test = reduce_size_of_dataframe(FRACTION, x_train, x_test, y_train, y_test)

    print_data_information(x_train, y_train, x_val, y_val, x_test, y_test)

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
