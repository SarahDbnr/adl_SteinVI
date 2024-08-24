import tensorflow as tf
import jax.numpy as jnp
from jax.scipy.stats import norm
import jax
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine


def true_function_mean(x):
    mean =x[:1]**2+x[:0]**2 #jnp.sin(x)
    return mean

def true_function_variance(x):
    var = x**2
    return var

def true_function(x, true_function_mean=true_function_mean, true_function_variance=true_function_variance):
    x1 = x[:, 0]
    x2 = x[:, 1]
    
    mean_outputs = x1**3 - x2**3 + 0.5 * x1 * x2
    return mean_outputs

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

def load_data(dataset, reduce_size=False, val_split=0.1, fraction=0.1, num_points = 10000,rng_key=None, true_function=None):
    if dataset_name is None:
        # Split the number of points into training and testing
        num_train = int(0.8 * num_points)
        num_test = num_points - num_train
        
        # Generate random input data within the specified range
        key, subkey = jax.random.split(key)
        x_train = jax.random.uniform(subkey, shape=(num_train, input_dimension), minval=range[0], maxval=range[1])
        
        key, subkey = jax.random.split(key)
        x_test = jax.random.uniform(subkey, shape=(num_test, input_dimension), minval=range[0], maxval=range[1])
        
        # Generate output data using the true function
        y_train = true_function(x_train)
        y_test = true_function(x_test)
        
        val_size = int(len(x_train) * val_split)
        x_val, y_val = x_train[-val_size:], y_train[-val_size:]
        x_train, y_train = x_train[:-val_size], y_train[:-val_size]
    elif dataset == "MNIST":
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    elif dataset == "FashionMNIST":
        # Load the Fashion MNIST dataset
        fashion_mnist = tf.keras.datasets.fashion_mnist
        # Separate it into train and test sets
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    elif dataset == "CIFAR10":
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    elif dataset == "california_housing" or dataset == "diabetes" or dataset == "wine_quality":
        x, y = load_real_world_data(dataset_name)
        # Shuffle and split data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(x))
        x, y = x[perm], y[perm]
        
        num_train = int(0.8 * len(x))
        num_val = int(val_split * num_train)
        num_test = len(x) - num_train
        
        x_train, y_train = x[:num_train], y[:num_train]
        x_val, y_val = x_train[-num_val:], y_train[-num_val:]
        x_train, y_train = x_train[:-num_val], y_train[:-num_val]
        x_test, y_test = x[num_train:], y[num_train:]
    else:
        raise ValueError(f"Dataset {dataset} not recognized. Available options: 'MNIST', 'FashionMNIST', 'CIFAR-10'")
    
    if reduce_size:
        train_sample_size = int(x_train.shape[0] * fraction)
        test_sample_size = int(x_test.shape[0] * fraction)
        train_indices = jnp.random.choice(x_train.shape[0], train_sample_size, replace=False)
        test_indices = jnp.random.choice(x_test.shape[0], test_sample_size, replace=False)
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
        x_test = x_test[test_indices]
        y_test = y_test[test_indices]
    
    # Split training data into train and validation sets
    val_size = int(len(x_train) * val_split)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]
    
    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test
