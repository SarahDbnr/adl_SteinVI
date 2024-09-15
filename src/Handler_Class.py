class Handler:
    """
    A class to define the parameters for printing and visualization during the training.
    """

    rf_comparison: bool

    def __init__(self, rf_comparison= False):
        self.rf_comparison = rf_comparison

    def set_training_print_mode(self, mode):
        """
        Sets the print mode for the training process. Only one mode can be active at any time.

        Args:
            mode (str): One of 'full', 'reduced', or 'none'.
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
        if mode == 'full':
            self._full_evaluation = True
            self._minimal_evaluation = False
        elif mode == 'minimal':
            self._full_evaluation = False
            self._minimal_evaluation = True
            print("Warning: Minimal evaluation mode does not gather information while training for efficiency reasons.")
        else:
            raise ValueError("Invalid mode. Choose from 'full', or 'minimal'.")


