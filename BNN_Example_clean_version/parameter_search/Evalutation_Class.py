from BNN_Example_clean_version.Parameter_Class import Parameter
from BNN_Example_clean_version.validation_and_evaluation import calculate_mse, calculate_mean_span_over_particles


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
        self.mean_true_output = true_output.mean()
        self.var_true_output = true_output.var()
        self.mean_prediction = test_predictions.mean()
        self.average_var_prediction = test_predictions.var(0).mean()
        self.particle_span_predictions = calculate_mean_span_over_particles(test_predictions)
        self.mean_precision = test_precision.mean()
        self.var_precision = test_precision.var()
        self.mse = calculate_mse(test_predictions, true_output)


class EvaluationMulticlass:
    name: str
    parameter_setting: Parameter
    output_options: []
    mean_true_output: float
    mean_prediction: float
    particle_span: float
    accuracy: float

