class Parameter:
    """
    A class to define and manage the parameters for training a model using Stein Variational Inference methods for example SVGD.

    This class stores and handles various parameters required for training, such as the optimizer, learning rate, batch sizes, 
    number of particles, and early stopping configurations. It provides functionality to set early stopping criteria, 
    which can help terminate the training process early if performance stabilizes.

    Attributes: 
        optimizer (optax.GradientTransformation): The optimizer used to update the model parameters for svgd. It must conform to the `optax.GradientTransformation` protocol, such as Adam or RMSProp. For plain_svgd and ssvgd no optimizer is used. For quasi_SVN the optimizer must be the optax optimizer "LBFGS".optimizer (optax.optim)
        learning_rate (float): learning rate used for plain_svgd and ssvgd for the other algorithms the learning rate is included in the optimizer.
        num_particles (int): The number of particles used in the process to approximate the posterior.
        batch_size (int): The size of the mini-batches for data used during training.
        particle_batch_size (int): The number of particles in a mini-batch of particles used for training.
        num_iterations (int): The total number of training iterations.
        stopped_at_iteration (int): The iteration at which the training stopped, used especially when early stopping is applied.
        kernel_length (float): The length scale of the RBF kernel used in SVGD (default is 0.05).
        early_stopping (bool): A variable indicating whether early stopping is enabled.
        image_data (bool): A variable indicating whether the data is image data or not.
        warm_up_iterations_early_stopping (int): The number of iterations to perform before early stopping is allowed.
        patience_early_stopping (int): The number of iterations to wait for an improvement before stopping early.
        min_delta_early_stopping (float): The minimum change in the monitored metric to qualify as an improvement.
    """
    
    optimizer: None
    num_particles: int
    batch_size: int
    particle_batch_size: int
    num_iterations: int
    stopped_at_iteration: int
    kernel_length: float = 0.05
    early_stopping: bool
    image_data: bool
    warm_up_iterations_early_stopping: int = 30
    patience_early_stopping: int = 15
    min_delta_early_stopping: float = 0.005

    def __init__(self, optimizer, early_stopping, image_data, batch_size, particle_batch_size, num_particles,
                 num_iterations, kernel_length):
        """Initializes the Parameter class with the specified training settings.

        Args:
            optimizer (optax.GradientTransformation): The optimizer used to update the model parameters for svgd. It must conform to the `optax.GradientTransformation` protocol, such as Adam or RMSProp. For plain_svgd and ssvgd no optimizer is used. For quasi_SVN the optimizer must be the optax optimizer "LBFGS".optimizer (optax.optim)
            early_stopping (bool): A flag indicating whether early stopping is enabled.
            image_data (bool): A variable indicating whether the data is image data or not.
            batch_size (int): The size of the mini-batches for data used during training.
            particle_batch_size (int): The number of particles in a mini-batch of particles used for training.
            num_particles (int): The number of particles used in the process to approximate the posterior.
            num_iterations (int): The total number of training iterations.
            learning_rate (float): learning rate used for plain_svgd and ssvgd for the other algorithms the learning rate is included in the optimizer.
        """
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.particle_batch_size = particle_batch_size
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.stopped_at_iteration = num_iterations
        self.early_stopping = early_stopping
        self.image_data = image_data
        self.kernel_length = kernel_length

    def set_early_stopping(self, warm_up_iterations, patience, min_delta):
        """
        Sets the parameters for early stopping during training.

        Args:
            warm_up_iterations (int): Number of iterations before early stopping can be applied (warm-up period).
            patience (int): The number of iterations to wait for an improvement before early stopping.
            min_delta (float): The minimum change in the monitored value to be considered as an improvement.
        """
        self.warm_up_iterations_early_stopping = warm_up_iterations
        self.patience_early_stopping = patience
        self.min_delta_early_stopping = min_delta
