class Handler:
    """
    A class to define the parameters for printing and visualization during the training.
    """

    rf_comparison: bool
    num_diff_classifications_over_part: bool

    plot_val_acc_over_iter: bool
    plot_classification_detail: int
    plot_val_aver_prec_over_iter: bool
    plot_residual: bool
    plot_loc_relation_scale: bool


    def __init__(self, rf_comparison= False, num_diff_classifications_over_part= False, plot_val_acc_over_iter=False, plot_val_aver_prec_over_iter=False, plot_residual=False, plot_loc_relation_scale=False, plot_classification_detail=0):
        self._full_training_print = False
        self._reduced_training_print = True
        self._no_training_print = False
        self.rf_comparison = rf_comparison
        self.num_diff_classifications_over_part = num_diff_classifications_over_part

        self.plot_classification_detail = plot_classification_detail
        self.plot_val_acc_over_iter = plot_val_acc_over_iter
        self.plot_val_aver_prec_over_iter = plot_val_aver_prec_over_iter
        self.plot_residual = plot_residual
        self.plot_loc_relation_scale = plot_loc_relation_scale

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

