class Parameter:
    """
    A class to define the parameters for training a model using Stein Variational Gradient Descent (SVGD).
    """
    optimizer: None
    num_particles: int
    batch_size: int
    particle_batch_size: int
    num_iterations: int
    stopped_at_iteration: int
    kernel_length: float = 0.05
    early_stopping: bool
    warm_up_iterations_early_stopping: int = 30
    patience_early_stopping: int = 15
    min_delta_early_stopping: float = 0.005

    def __init__(self, optimizer, early_stopping, image_data, batch_size, particle_batch_size, num_particles,
                 num_iterations):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.particle_batch_size = particle_batch_size
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.stopped_at_iteration = num_iterations
        self.early_stopping = early_stopping

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
