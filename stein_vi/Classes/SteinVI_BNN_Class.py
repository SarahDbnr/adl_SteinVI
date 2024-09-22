import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import flax.linen
from optax import adam

from stein_vi.Classes.Parameter_Class import Parameter
import stein_vi.metrics.plots_validation_metrics as plots
from stein_vi.metrics.view_misclassified import view_misclassified
from stein_vi.Classes.Handler_Class import Handler
from stein_vi.algorithm.get_posteriori import get_posteriori



class SteinVI_BNN:

    """SteinVI_BNN is a class using Stein VI methods to optimize a BNN.
    This class manages the initialization, training, and evaluation of a BNN with support for both classification and regression tasks.
    It integrates various Stein Variational inference algorithms to update particles representing the network's parameters.
    Key functionalities:
    - Initialize particles for variational inference.
    - Evaluate the BNN using different evaluation metrics.
    - Plot various metrics and visualizations for both regression and classification tasks.

    Attributes:
        state (SVGD_State): A class containing informations about the (optimizer), Kernel and the particles.
        parameter (Parameter): Configuration of training parameters like learning rate, batch size, and number of particles.
        handler (Handler): Manages modes for training and evaluation (e.g., minimal or full evaluation).
        use_for_regression (bool): A variable to determine if the network is being used for regression or classification tasks.
        log_posteriori (callable): A function representing the log posterior of the model.
        nnet (flax.linen.Module): The neural network model.
        tree_def (jax.tree_util.PyTreeDef): Tree structure used for parameter transformation in JAX.
        initial_particle_vector (jax.numpy.ndarray): The initial particle vectors for the network's parameters.
        update_fn (callable???): The function used for updating particles during training. ???
        evaluate_fn (callable???): The function used for evaluating the model's performance.???
        evaluation_metrics_1 (list): Stores evaluation metrics (like accuracy or MSE) over training iterations.
        evaluation_metrics_2 (list): Stores additional evaluation metrics (like precision or variance) over training iterations.
    """

    state: None
    parameter: Parameter
    handler: Handler

    use_for_regression: bool

    log_posteriori: callable
    nnet: flax.linen.Module
    tree_def: None
    initial_particle_vector: jnp.array

    update_fn: object
    evaluate_fn: object

    evaluation_metrics_1: list = []
    evaluation_metrics_2: list = []

    def __init__(self, key, x_train, nnet, use_for_regression,
                 optimizer=adam(0.01), mode_training_print='none', mode_evaluation='full', early_stopping=False,
                 image_data=False, batch_size=0, particle_batch_size=0,
                 num_particles=10, num_iterations=100, rf_comparison=False, kernel_length=0.005):
        """_summary_

        Args:
            key (jax.random.PRNGKey): A JAX PRNG key used for random number generation to initialize particles.
            x_train (jax.numpy.ndarray): The input training data used to initialize the neural network's parameters.
            nnet (flax.linen.Module): The neural network model that will be optimized using Stein Variational Inference.
            use_for_regression (bool): Determines if the model is used for regression (True) or classification (False).
            optimizer (optax.GradientTransformation): The optimizer used to update the model parameters for svgd. It must conform to the `optax.GradientTransformation` protocol, such as Adam or RMSProp. For plain_svgd and ssvgd no optimizer is used. For quasi_SVN the optimizer must be the optax optimizer "LBFGS".optimizer (optax.optim)
            mode_training_print (str, optional): The verbosity level for printing training logs. Options are 'none', 'minimal', or 'full'. Defaults to 'none'. ??? more exact
            mode_evaluation (str, optional): The level of evaluation performed during training. Options are 'minimal' or 'full'. Defaults to 'full'. ??? more exact
            early_stopping (bool, optional): Whether to enable early stopping during training. Defaults to False.
            image_data (bool, optional): Indicates if the input data is image-based, which affects certain visualizations. Defaults to False.
            batch_size (int, optional): The size of the mini-batches for training. Defaults to 0 (i.e., no mini-batching).  Defaults to 0 (i.e., no batching).
            particle_batch_size (int): The number of particles in a mini-batch of particles used for training.  Defaults to 0 (i.e., no batching).
            num_particles (int): The number of particles used in the process to approximate the posterior. Defaults to 10.
            num_iterations (int): The total number of training iterations. Defaults to 100.
            rf_comparison (bool, optional): If random forest comparisson should be done. Defaults to False.
            learning_rate (float, optional): Learning rate used for ssvgd for the svgd algorithm the learning rate is included in the optimizer. Defaults to 0.0001.
        """

        self.handler = Handler(rf_comparison)
        self.handler.set_training_print_mode(mode_training_print)
        self.handler.set_evaluation_mode(mode_evaluation)

        if len(x_train) < batch_size:
            raise ValueError("Error: batch_size bigger then input data length")
        if num_particles < particle_batch_size:
            raise ValueError("Error: particle_batch_size bigger then number of particles")
        self.parameter = Parameter(optimizer, early_stopping, image_data, batch_size, particle_batch_size,
                                   num_particles, num_iterations, kernel_length)
        self.use_for_regression = use_for_regression
        self.nnet = nnet
        self.nnet.predict = self.predict

        self.initial_particles_and_kernel(key, x_train, num_particles)

        self.log_posteriori = get_posteriori(self.nnet, self.use_for_regression)

    def initial_particles_and_kernel(self, key, x_train, num_particles):
        """"Initializes the particle vectors and kernel structure for the Bayesian Neural Network.

        Args:
            key (jax.random.PRNGKey): A key used for random number generation to initialize particle vectors.
            x_train (jax.numpy.ndarray): The input training data, used to initialize the parameters of the neural network.
            num_particles (int): The number of particles to initialize for the variational inference process.
        """

        init_param = self.nnet.init(key, x_train)
        param_vec, self.tree_def = ravel_pytree(init_param)

        self.initial_particle_vector = jax.random.normal(key, shape=(num_particles,) + param_vec.shape)

    def predict(self, weights, x_input, use_softmax=False):
        """Generates predictions using the neural network with the provided weights and input data

        Args:
            weights (dict):  A dictionary containing the weights (parameters) of the neural network, where the keys correspond to different parts of the model (e.g., layers).
            x_input (jax.numpy.ndarray): The input data for which predictions are to be made.
            use_softmax (bool): True is log_softmax should be used for multiclass. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - prediction (jax.numpy.ndarray): The predicted class labels or regression outputs.
                - precision (jax.numpy.ndarray): The precision values for classification (as probabilities) or predictive variance for regression.
        """

        if self.use_for_regression:
            output = self.nnet.apply(self.tree_def(weights), x_input)
            prediction, precision = jnp.split(output, 2, axis=-1)
        elif use_softmax:
            predictions = self.nnet.apply(self.tree_def(weights), x_input)
            precision = jax.nn.log_softmax(predictions, axis=-1)
            prediction = jnp.argmax(precision, axis=-1)
        else:
            predictions = self.nnet.apply(self.tree_def(weights), x_input)
            precision = jax.nn.softmax(predictions, axis=-1)
            prediction = jnp.argmax(precision, axis=-1)
        return prediction, precision

    def predict_over_particles(self, input_data):
        predictions, precisions = jax.vmap(lambda p: self.predict(p, input_data))(self.state.particles)
        return predictions.squeeze(), precisions.squeeze()
    
    def plot_val_metric_over_iter(self):
        """Plots the evaluation metrics over the course of training iterations.
        This function plots validation metrics (such as Mean Squared Error for regression tasks or Accuracy for classification tasks) for each iteration of training, based on the evaluation mode.
        - For regression tasks, the function plots the MSE and averaged precision over iterations.
        - For classification tasks, the function plots the accuracy over iterations.

        Raises: 
            ValueError: If `minimal_evaluation` mode is enabled, which skips metric tracking for efficiency and does not gather information during training.
        """
        if self.handler.minimal_evaluation:
            ValueError(
                "Error: Minimal evaluation mode does not gather information while training for efficiency reasons.")
        else:
            if self.use_for_regression:
                plots.plot_evaluation_metric(evaluation_metric_val=self.evaluation_metrics_1,
                                                      num_particles=self.parameter.num_particles,
                                                      eval_metric="MSE")
                plots.plot_evaluation_metric(evaluation_metric_val=self.evaluation_metrics_2,
                                                      num_particles=self.parameter.num_particles,
                                                      eval_metric="averaged_precision")
            else:
                plots.plot_evaluation_metric(evaluation_metric_val=self.evaluation_metrics_1,
                                                            num_particles=self.parameter.num_particles,
                                                            eval_metric="Accuracy")

    def plot_residuals(self, z_test, y_test):
        """Plots the residduals between the predictions and the true values for regression tasks.

        Args:
            z_test (jax.numpy.ndarray): The input features for the test set.
            y_test (jax.numpy.ndarray): The true output labels for the test set.

        Raises: 
            ValueError: If this method is called for a classification task instead of regression.
        """

        if not self.use_for_regression:
            ValueError("This plot is only available for regression problems.")
        plots.plot_residuals(self.predict_over_particles(z_test)[0], y_test, num_particles=self.parameter.num_particles)

    def plot_location_in_relation_to_scale(self, z_test):
        """Plots the relationship between the location of predictions and the scale (uncertainty) for regression task.

        Args:
            z_test jax.numpy.ndarray): The input features to generate predictions and analyze their relationship to uncertainty.
        
        Raises: 
            ValueError: If this method is called for a classification task instead of regression.
        """

        if self.use_for_regression:
            plots.plot_location_in_relation_to_scale(self.nnet, self.state, z_test,
                                                     num_particles=self.parameter.num_particles)
        else:
            ValueError("This plot is only available for regression problems.")

    def view_misclassified(self, z_test, y_test, key= jax.random.PRNGKey(1), num_plots=3):
        """  This function will display the image if it is image_data and will show the probabilities for the classes are distributed.
        num_plots random misclassified and num_plots random right classified data examples.
        
        Args:
            z_test (jax.numpy.ndarray): Input features to the model.
            y_test (jax.numpy.ndarray): True output labels for the given input.
            key (jax.random.PRNGKey): A JAX PRNG key used for deterministic selection of samples.
            num_plots (int): Number of plots to be shown. Defaults to 3.
        """

        if self.use_for_regression:
            ValueError("This plot is only available for classification problems.")
        else:
            view_misclassified(self.state,self.nnet, z_test, y_test,
                               self.parameter.image_data,key,num_plots)
