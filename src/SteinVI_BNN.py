from src.Parameter_Class import Parameter
import src.metrics.plots_validation_metrics as plots
from src.Handler_Class import Handler

from blackjax.vi.svgd import SVGDState
import flax.linen
import jax.numpy as jnp
import jax
from optax import adam

class SteinVI_BNN:
    state: None
    parameter: Parameter
    handler: Handler

    use_for_regression: bool

    log_posterior: object
    nnet: flax.linen.Module

    kernel_fn: object
    update_fn: object
    evaluate_model_fn: object
    early_stopping_fn: object
    init_update_fn: object

    eval_metrics_1: float
    eval_metrics_2: float

    def __init__(self, mode_training_print='none', mode_evaluation='minimal', use_for_regression=None, optimizer=adam(0.01), early_stopping=False, image_data=False, batch_size=0, particle_batch_size=0, num_particles=10, num_iterations=100, rf_comparison=False):
        if use_for_regression == None:
            ValueError("The variable use_for_regression needs to be defined.")
        self.handler = Handler(rf_comparison)
        self.handler.set_training_print_mode(mode_training_print)
        self.handler.set_evaluation_mode(mode_evaluation)

        self.parameter = Parameter(optimizer, early_stopping, image_data, batch_size, particle_batch_size, num_particles, num_iterations)
        self.use_for_regression = use_for_regression

    def predict(self, weights, x_input):
        if self.use_for_regression:
            output = self.nnet.apply(weights, x_input)
            prediction, precision = jnp.split(output, 2, axis=-1)
        else:
            predictions = self.nnet.apply(weights, x_input)
            precision = jax.nn.softmax(predictions, axis=-1)
            prediction = jnp.argmax(precision, axis=-1)
        return prediction, precision

    def plot_val_metric_over_iter(self):
        plots.plot_and_save_evaluation_metric(self.state, self)

    plot_val_metric_over_iter: bool
    plot_classification_detail: int
    plot_val_aver_prec_over_iter: bool
    plot_residual: bool
    plot_loc_relation_scale: bool

        