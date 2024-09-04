class EvaluationRegression:
    name: str
    parameter_setting: Parameter
    mean_true_output: float
    var_true_output: float
    mean_prediction: float
    var_prediction: float
    particle_span_predictions: float
    mean_precision: float
    var_precision: float
    mse: float


class EvaluationMulticlass:
    name: str
    parameter_setting: Parameter
    output_options: []
    mean_true_output: float
    mean_prediction: float
    particle_span: float
    accuracy: float

