from stein_vi.algorithm.model_training import train_general_algorithm
from stein_vi.algorithm.svgd import set_up_svgd
from stein_vi.algorithm.plain_SVGD.plain_svgd import set_up_plain_svgd
from stein_vi.algorithm.sSVGD.ssvgd import set_up_ssvgd
from stein_vi.algorithm.quasi_SVN_with_lbfgs.quasi_SVN import set_up_quasi_SVN
from stein_vi.metrics.validation_and_evaluation import (print_summary_over_particles_regression,
                                                        print_summary_over_particles_multiclass)


def train_with_stein_vi(steinvi, dataset, key, algorithm="svgd"):
    """Performs the training with the given algorithms (svgd, plain_svgd, ssvgd, quasi_svn).
    Also prints out a score for the performance of the network on the test dataset.

    Args:
        steinvi (SteinVI_BNN): _description_
        dataset (tuple): Tuple containing processed training, validation, and test data as (x_train, y_train, x_val, y_val, x_test, y_test).
        key (jaxlib.xla_extension.ArrayImpl): A key used to control the generation of random objects.
        algorithm (str, optional): Specifies the algorithm used to train the BNN Defaults to "svgd".

    Raises:
        ValueError: If the specified algorithm is not supported. The error message will specify the unsupported algorithm.

    Returns:
        SteinVI_BNN: Returns an object of the class SteinVI_BNN with the updated status of the particels and the evaluation metrics.
    """
    if algorithm == "svgd":
        steinvi = set_up_svgd(steinvi)
    elif algorithm == "plain_svgd":
        steinvi = set_up_plain_svgd(steinvi)
    elif algorithm == "ssvgd":
        steinvi = set_up_ssvgd(steinvi)
    elif algorithm == "quasi_svn":
        steinvi = set_up_quasi_SVN(steinvi)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    steinvi = train_general_algorithm(
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
