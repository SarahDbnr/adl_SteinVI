class Parameter:
    optimizer: None
    use_for_regression: bool
    num_particles: int
    batch_size: int
    num_iterations: int
    stopped_at_iteration: int
    kernel_length: float = 0.05
    warm_up_iterations_early_stopping: int = 50
    patience_early_stopping: int = 30
    min_delta_early_stopping: float = 0.05

    def __init__(self, optimizer, regression, batch_size=None, num_particles=10, num_iterations=100):
        self.optimizer = optimizer
        self.use_for_regression = regression
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.stopped_at_iteration= num_iterations

    def set_early_stopping(self, warm_up_iterations, patience, min_delta):
        self.warm_up_iterations_early_stopping = warm_up_iterations
        self.patience_early_stopping = patience
        self.min_delta_early_stopping = min_delta
