import tensorflow as tf
import numpy as np

def load_data(dataset, reduce_size=False, val_split=0.1, fraction=0.1):
    if dataset == "MNIST":
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
    else:
        raise ValueError(f"Dataset {dataset} not recognized. Available options: 'MNIST', 'FashionMNIST', 'CIFAR-10'")
    
    if reduce_size:
        train_sample_size = int(x_train.shape[0] * fraction)
        test_sample_size = int(x_test.shape[0] * fraction)
        train_indices = np.random.choice(x_train.shape[0], train_sample_size, replace=False)
        test_indices = np.random.choice(x_test.shape[0], test_sample_size, replace=False)
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
