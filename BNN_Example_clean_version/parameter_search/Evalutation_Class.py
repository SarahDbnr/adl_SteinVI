class Parameter:
    optimizer: None
    network_structure: None
    output_size: int
    num_particles: int = 10
    batch_size: int = None
    iterations: int = 100
    kernel_length: float = 0.05
    warm_up_iterations_early_stopping: int = 150
    patience_early_stopping: int = 100
    min_delta_early_stopping: float = 0.01
    use_for_regression: bool = False


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

