import jax
import jax.numpy as jnp
from sklearn.datasets import fetch_20newsgroups, fetch_openml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import issparse
VAL_SPLIT = 0.1


def apply_data_settings_keras(new_dataset, with_flattening=False, fraction=1):
    """
    Processes a dataset obtained from Keras, normalizing and optionally flattening it, and splitting it into training, validation, and test sets.

    Args:
        new_dataset (tuple): A tuple containing training and test datasets as (x_train, y_train), (x_test, y_test).
        with_flattening (bool): If True, flattens the output arrays.
        fraction (float): Fraction of the data to use for reducing dataset size.

    Returns:
        tuple: Tuple containing processed training, validation, and test data as (x_train, y_train, x_val, y_val, x_test, y_test).
    """
    (x_train, y_train), (x_test, y_test) = new_dataset
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    if with_flattening:
        y_train = y_train.flatten()
        y_test = y_test.flatten()

    val_size = int(len(x_train) * VAL_SPLIT)
    x_val, y_val = x_train[-val_size:], y_train[-val_size:]
    x_train, y_train = x_train[:-val_size], y_train[:-val_size]

    print_data_information(x_train, y_train, x_val, y_val, x_test, y_test)

    if fraction < 1:
        x_train, y_train, x_test, y_test = reduce_size_of_dataframe(fraction, x_train, x_test, y_train, y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def apply_data_settings_sklearn(new_dataset, fraction=1):
    """
    Processes a dataset obtained from scikit-learn, shuffling and splitting it into training, validation, and test sets, and optionally reducing its size.

    Args:
        new_dataset (object): An object containing 'data' and 'target' attributes, typical of scikit-learn datasets.
        fraction (float): Fraction of the data to use for reducing dataset size.

    Returns:
        tuple: Tuple containing processed training, validation, and test data as (x_train, y_train, x_val, y_val, x_test, y_test).
    """
    key = jax.random.PRNGKey(1)
    x = new_dataset.data
    y = new_dataset.target

    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, x.shape[0])
    x, y = x[perm], y[perm]

    num_train = int(0.8 * x.shape[0])
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
    """
    Reduces the size of training and test datasets to a specified fraction by random sampling.

    Args:
        fraction (float): Fraction of the original size to retain.
        x_train (np.ndarray): Training input data.
        x_test (np.ndarray): Test input data.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Test labels.

    Returns:
        tuple: Tuple containing the reduced training and test data as (x_train, y_train, x_test, y_test).
    """
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
    """
    Prints the shapes of training, validation, and test data and labels.

    Args:
        x_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training labels.
        x_val (np.ndarray): Validation input data.
        y_val (np.ndarray): Validation labels.
        x_test (np.ndarray): Test input data.
        y_test (np.ndarray): Test labels.
    """
    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}")
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")


def newsgroup_datahandling():
    """
    Loads and processes the 20 Newsgroups dataset for text classification, converting the text data to a TF-IDF feature matrix.

    Returns:
        CustomDataset: A dataset object containing the processed features and targets.
    """
    newsgroups = fetch_20newsgroups(subset='all')

    # Convert the text data to a TF-IDF feature matrix
    vectorizer = TfidfVectorizer(max_features=2000)  # Limiting to 2000 features to make it manageable
    X = vectorizer.fit_transform(newsgroups.data).toarray()  # Convert sparse matrix to dense array
    y = newsgroups.target


    class CustomDataset:
        def __init__(self, data, target):
            self.data = data
            self.target = target

    dataset = CustomDataset(X, y)
    return dataset

def adult_income_datahandling():
    """
    Fetches and preprocesses the Adult Income dataset for binary classification. Features are scaled and encoded appropriately.

    Returns:
        CustomDataset: A dataset object containing the processed features and targets.
    """
    adult_income = fetch_openml(data_id=1590, as_frame=True)
    

    X = adult_income.data
    y = (adult_income.target == '>50K').astype(int)  # Binary classification: '>50K' is class 1, otherwise 0
    
    # Define preprocessing for numerical and categorical data
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Convert the processed features to a dense array if necessary
    if issparse(X_processed):
        X_processed = X_processed.toarray()

    # Convert y to a numpy array
    y = y.to_numpy()

    # Create a new dataset object as expected by `apply_data_settings_sklearn`
    class CustomDataset:
        def __init__(self, data, target):
            self.data = data
            self.target = target

    dataset = CustomDataset(X_processed, y)
    return dataset