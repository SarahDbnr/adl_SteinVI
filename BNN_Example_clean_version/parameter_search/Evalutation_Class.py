from BNN_Example_clean_version.Parameter_Class import Parameter


class EvaluationRegression:
    name: str
    parameter_setting: Parameter
    mean_true_output: float
    var_true_output: float
    mean_prediction: float
    average_var_prediction: float
    particle_span_predictions: float
    mean_precision: float
    var_precision: float
    mse: float

    def __init__(self, name, true_output, test_predictions, test_precision):
        self.name = name
        mean_true_output = true_output.mean()
        var_true_output = true_output.var()
        mean_prediction = test_predictions.mean()
        average_var_prediction = test_predictions.var(0).mean()
        # TODO: particle_span_predictions
        mean_precision = test_precision.mean()
        var_precision = test_precision.var()
        # TODO: mse


class EvaluationMulticlass:
    name: str
    parameter_setting: Parameter
    output_options: []
    mean_true_output: float
    mean_prediction: float
    particle_span: float
    accuracy: float

