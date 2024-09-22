class Handler:
    """
    A class to manage the parameters for printing and evaluation modes during training and evaluation.

    This class controls how much information is printed during training and how thoroughly the model is evaluated during training and testing. It allows setting different modes (full, reduced, or none) for both training prints and evaluation.

    Attributes:
        rf_comparison (bool): A flag indicating whether random feature comparison is enabled during training.
        _full_training_print (bool): Indicates if the full training print mode is active.
        _reduced_training_print (bool): Indicates if the reduced training print mode is active.
        _no_training_print (bool): Indicates if no training print mode is active.
        _full_evaluation (bool): Indicates if the full evaluation mode is active.
        _minimal_evaluation (bool): Indicates if the minimal evaluation mode is active.
    """

    rf_comparison: bool
    _full_training_print: bool
    _reduced_training_print: bool
    _no_training_print: bool
    _full_evaluation: bool
    _minimal_evaluation: bool

    def __init__(self, rf_comparison= False):
        """Initializes the Handler class with the option to enable random forest comparisson.
        Args:
            rf_comparison (bool, optional): _description_. Defaults to False.
        """
        self.rf_comparison = rf_comparison
    def set_training_print_mode(self, mode):
        """
        Sets the print mode for the training process. Only one mode can be active at any time.

        Args:
            mode (str): One of 'full', 'reduced', or 'none', which controls the verbosity of training prints.
                        - 'full': Prints all details during training.
                        - 'reduced': Prints minimal information during training.
                        - 'none': Disables all training prints.
        Raises:
            ValueError: If an invalid mode is provided.
        """
        if mode == 'full':
            self._full_training_print = True
            self._reduced_training_print = False
            self._no_training_print = False
        elif mode == 'reduced':
            self._full_training_print = False
            self._reduced_training_print = True
            self._no_training_print = False
        elif mode == 'none':
            self._full_training_print = False
            self._reduced_training_print = False
            self._no_training_print = True
        else:
            raise ValueError("Invalid mode. Choose from 'full', 'reduced', or 'none'.")

    def set_evaluation_mode(self, mode):
        """Sets the evaluation mode for the model during training.

        Args:
            mode (str): One of 'full' or 'minimal', which controls the level of evaluation performed.
                        - 'full': Gathers all possible information during evaluation.
                        - 'minimal': Collects only essential evaluation metrics, providing faster but less detailed results.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        if mode == 'full':
            self._full_evaluation = True
            self._minimal_evaluation = False
        elif mode == 'minimal':
            self._full_evaluation = False
            self._minimal_evaluation = True
            print("Warning: Minimal evaluation mode does not gather information while training for efficiency reasons.")
        else:
            raise ValueError("Invalid mode. Choose from 'full', or 'minimal'.")

    @property
    def minimal_evaluation(self):
        """Returns the status of the minimal evaluation mode.

        Returns:
            bool: True if minimal evaluation mode is active, False otherwise.
        """
        return self._minimal_evaluation
