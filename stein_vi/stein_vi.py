from stein_vi.algorithm.model_training import train_general_algorithm
from stein_vi.algorithm.svgd import set_up_svgd


def train_with_stein_vi(steinvi, dataset, key, algorithm="svgd"):

    if algorithm == "svgd":
        steinvi = set_up_svgd(steinvi)

    steinvi = train_general_algorithm(
        steinvi=steinvi,
        dataset=dataset,
        key=key
    )

    _, _, _, _, z_test, y_test = dataset
    if steinvi.use_for_regression:
        mse_test, averaged_precision_test, predictions_test = steinvi.evaluate_fn(steinvi.state,z_test,y_test)
        print("For Test Data: MSE ", mse_test, ", Averaged Variance ", averaged_precision_test,
              "with mean prediction of ", predictions_test.mean())
    else:
        accuracy_test, _, predictions_test = steinvi.evaluate_fn(steinvi.state, z_test, y_test)
        print("For Test Data: Accuracy ", accuracy_test)

    return steinvi
