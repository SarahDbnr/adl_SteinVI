from stein_vi.algorithm.model_training import train_general_algorithm
from stein_vi.algorithm.svgd import set_up_svgd
from stein_vi.algorithm.sSVGD.ssvgd import set_up_ssvgd
from stein_vi.metrics.validation_and_evaluation import (print_summary_over_particles_regression,
                                                        print_summary_over_particles_multiclass)
from stein_vi.algorithm.random_forest import random_forest


def train_with_stein_vi(steinvi, dataset, key, algorithm="svgd"):
    """
    Performs the training using one of the specified algorithms (svgd, ssvgd) for a Bayesian Neural Network (BNN).
    Also prints out a score for the performance of the network on the test dataset.

    Algorithms:
        - **svgd (Stein Variational Gradient Descent)**: Uses the blackjax implementation of SVGD.
        - **ssvgd (Stochastic Stein Variational Gradient Descent)**: A variant of svgd where noise is added to the particle updates. 

    Args:
        steinvi (SteinVI_BNN): An instance of the `SteinVI_BNN` class, containing the Bayesian neural network, particles, and other relevant parameters.
        dataset (tuple): A tuple containing processed training, validation, and test data as (x_train, y_train, x_val, y_val, x_test, y_test).
        key (jax.random.PRNGKey): A key used to control the generation of random objects. Here it is used, for example, for the selection of batch subsets.
        algorithm (str, optional): Specifies the algorithm used to train the BNN. Defaults to "svgd". Available options: "svgd","ssvgd".

    Raises:
        ValueError: If the specified algorithm is not supported or if particle batching is attempted with `quasi_svn`.

    Returns:
        SteinVI_BNN: Returns an updated instance of the `SteinVI_BNN` class, including the trained particles and evaluation metrics.

    Prints: 
        Tests scores for a given data test dataset.
    """
    if algorithm == "svgd":
        set_up_svgd(steinvi)
    elif algorithm == "ssvgd":
        set_up_ssvgd(steinvi)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    train_general_algorithm(
        steinvi=steinvi,
        dataset=dataset,
        key=key
    )

    _, _, _, _, z_test, y_test = dataset
    print("\nFor Test Data:")
    if steinvi.use_for_regression:
        _, _, predictions_test = steinvi.evaluate_fn(steinvi.state, z_test, y_test, print_out=True)
        print_summary_over_particles_regression(predictions_test)
    else:
        _, _, predictions_test = steinvi.evaluate_fn(steinvi.state, z_test, y_test, print_out=True)
        print_summary_over_particles_multiclass(predictions_test)

    if steinvi.handler.rf_comparison and steinvi.use_for_regression:
        print("\nRandom Forest comparison")
        print("Test MSE:       ", random_forest(dataset, 'regression')['Test MSE'])
        print("Test Precision: ", random_forest(dataset, 'regression')['Test Precision'])
    elif steinvi.handler.rf_comparison and steinvi.use_for_regression == False:
        print("\nRandom Forest comparison")
        print("Test Accuracy:   ", random_forest(dataset)['Test Accuracy'])
