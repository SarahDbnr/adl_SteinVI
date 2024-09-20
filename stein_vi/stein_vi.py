from stein_vi.algorithm.model_training import train_general_algorithm
from stein_vi.algorithm.svgd import set_up_svgd
from stein_vi.algorithm.sSVGD.ssvgd import set_up_ssvgd
from stein_vi.algorithm.quasiSVN.quasiSVN import set_up_quasi_SVN
from stein_vi.metrics.validation_and_evaluation import (print_summary_over_particles_regression,
                                                        print_summary_over_particles_multiclass)


def train_with_stein_vi(steinvi, dataset, key, algorithm="svgd"):
    """
    Performs the training using one of the specified algorithms (svgd, plain_svgd, ssvgd, quasi_svn) for a Bayesian Neural Network (BNN).
    Also prints out a score for the performance of the network on the test dataset.

    Algorithms:
        - **svgd (Stein Variational Gradient Descent)**: Uses the blackjax implementation of SVGD.
        - **plain_svgd**: Uses a modified version of the blackjax svgd.py file where the optimizer is removed. So now only plain gradient decent will be performed with a given learning rate. (Attention: Gets stuck really easily and als much worse than normal SVGD with an adam optimizer)
        - **ssvgd (Stochastic Stein Variational Gradient Descent)**: A variant of plain_svgd where noise is added to the particle updates. (Attention: Often in our BNN the gradient and the weights have quite high values e.g. (-40000;40000) and noise calculated based on the paper "A STOCHASTIC STEIN VARIATIONAL NEWTON METHOD"  is very small since the kernel is used to scale the variance and when using the RBF_Kernel the values are between 0 and 1.
        - **quasi_svn**: Here the optax optimizer LBFGS needs to be used, allowing for second-order updates. Note: This method does not support particle batching, and will raise an error if attempted.

    Args:
        steinvi (SteinVI_BNN): An instance of the `SteinVI_BNN` class, containing the Bayesian neural network, particles, and other relevant parameters.
        dataset (tuple): A tuple containing processed training, validation, and test data as (x_train, y_train, x_val, y_val, x_test, y_test).
        key (jax.random.PRNGKey): A key used to control the generation of random objects. Here it is used, for example, for the selection of batch subsets.
        algorithm (str, optional): Specifies the algorithm used to train the BNN. Defaults to "svgd". Available options: "svgd", "plain_svgd", "ssvgd", "quasi_svn".

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
    elif algorithm == "quasi_svn":
        if steinvi.parameter.particle_batch_size != 0:
            raise ValueError(f"Particle batching is not supported for {algorithm}")
        set_up_quasi_SVN(steinvi)
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
        mse_test, averaged_precision_test, predictions_test = steinvi.evaluate_fn(steinvi.state, z_test, y_test,
                                                                                  print_out=True)
        print_summary_over_particles_regression(predictions_test)
    else:
        accuracy_test, _, predictions_test = steinvi.evaluate_fn(steinvi.state, z_test, y_test, print_out=True)
        print_summary_over_particles_multiclass(predictions_test)

    return steinvi
