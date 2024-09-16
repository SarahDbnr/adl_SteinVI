from src.Parameter_Class import Parameter
import src.metrics.plots_validation_metrics as plots
from src.metrics.view_misclassified_images import view_misclassified
from src.Handler_Class import Handler

import flax.linen
import jax.numpy as jnp
import jax
from optax import adam
from jax.flatten_util import ravel_pytree


class SteinVI_BNN:
    state: None
    parameter: Parameter
    handler: Handler

    use_for_regression: bool

    log_posterior: object
    nnet: flax.linen.Module
    tree_def: None

    update_fn: object
    evaluate_model_fn: object
    early_stopping_fn: object

    eval_metrics_1: float = []
    eval_metrics_2: float = []

    def __init__(self, mode_training_print='none', mode_evaluation='minimal', use_for_regression=None,
                 optimizer=adam(0.01), early_stopping=False, image_data=False, batch_size=0, particle_batch_size=0,
                 num_particles=10, num_iterations=100, rf_comparison=False):
        if use_for_regression == None:
            ValueError("The variable use_for_regression needs to be defined.")
        self.handler = Handler(rf_comparison)
        self.handler.set_training_print_mode(mode_training_print)
        self.handler.set_evaluation_mode(mode_evaluation)

        self.parameter = Parameter(optimizer, early_stopping, image_data, batch_size, particle_batch_size,
                                   num_particles, num_iterations)
        self.use_for_regression = use_for_regression

    def initial_particles_vector(self, key, x_train, num_particles):
        init_param = self.nnet_model.init(key, x_train)
        param_vec, self.tree_def = ravel_pytree(init_param)
        return jax.random.normal(key, shape=(num_particles,) + param_vec.shape)

    def set_training_fns(self, kernel_fn, update_fn, evaluate_model_fn, early_stopping_fn, init_update_fn):
        self.kernel_fn = kernel_fn
        self.update_fn = update_fn
        self.evaluate_model_fn = evaluate_model_fn
        self.early_stopping_fn = early_stopping_fn
        self.init_update_fn = init_update_fn

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
        if self.handler.minimal_evaluation:
            ValueError(
                "Error: Minimal evaluation mode does not gather information while training for efficiency reasons.")
        else:
            if self.use_for_regression:
                plots.plot_and_save_evaluation_metric(evaluation_metric_val=self.eval_metrics_1,
                                                      num_particles=self.parameter.num_particles,
                                                      eval_metric="MSE")
                plots.plot_and_save_evaluation_metric(evaluation_metric_val=self.eval_metrics_2,
                                                      num_particles=self.parameter.num_particles,
                                                      eval_metric="averaged_precision")
            else:
                plots.plots.plot_and_save_evaluation_metric(evaluation_metric_val=self.eval_metrics_1,
                                                            num_particles=self.parameter.num_particles,
                                                            eval_metric="Accuracy")

    def plot_residuals(self, z_test, y_test):
        plots.plot_residuals(self.nnet, self.tree_def, self.state, z_test, y_test,
                             num_particles=self.parameter.num_particles)

    def plot_location_in_relation_to_scale(self, z_test):
        if self.use_for_regression:
            plots.plot_location_in_relation_to_scale(self.nnet, self.tree_def, self.state, z_test,
                                                     num_particles=self.parameter.num_particles)
        else:
            ValueError("This plot is only available for regression problems.")

    def view_misclassified(self, z_test, y_test, image_data=False):
        if self.use_for_regression:
            ValueError("This plot is only available for classification problems.")
        else:
            view_misclassified(self.nnet, self.tree_def, self.state, z_test, y_test,
                               image_data=image_data)
