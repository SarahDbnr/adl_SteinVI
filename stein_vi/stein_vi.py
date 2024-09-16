from stein_vi.algorithm.model_training import train_general_algorithm
from stein_vi.algorithm.svgd import set_up_svgd
from stein_vi.metrics.validation_and_evaluation import (print_summary_over_particles_regression,
                                                        print_summary_over_particles_multiclass)

def train_with_stein_vi(steinvi, dataset, key, algorithm="svgd"):

    if algorithm == "svgd":
        steinvi = set_up_svgd(steinvi)

    steinvi = train_general_algorithm(
        steinvi=steinvi,
        dataset=dataset,
        key=key
    )

    _, _, _, _, z_test, y_test = dataset
    print("\nFor Test Data:")
    if steinvi.use_for_regression:
        mse_test, averaged_precision_test, predictions_test = steinvi.evaluate_fn(steinvi.state, z_test, y_test,
                                                                                  print=True)
        print_summary_over_particles_regression(predictions_test)
    else:
        accuracy_test, _, predictions_test = steinvi.evaluate_fn(steinvi.state, z_test, y_test, print=True)
        print_summary_over_particles_multiclass(predictions_test)

    return steinvi
