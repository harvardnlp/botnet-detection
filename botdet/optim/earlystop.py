class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, mode='min', verbose=False, logger=None):
        """
        Args:
            patience (int, optional): How long to wait after last time validation loss improved.
                                      Default: 7
            mode (str, optional): 'min' or 'max'.
                                  Default: 'min'
            verbose (bool, optional): If True, prints a message for each validation loss improvement.
                                      Default: False
            logger (logging.Logger, optional): logger object. If not given, use print.
        Note:
            In the case of doing learning rate scheduling, the patience here could be set to
            (scheduling_patience + 1) * n + 1, where n is the maximum number of times the learning
            rate has been decayed.
        """
        assert mode in ['min', 'max']

        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.logging = logger.info if logger else print
        self.counter = 0
        self.best = None
        self.improved = False
        self.early_stop = False

    def __call__(self, val_metric):

        if self.best is None:
            self.best = val_metric
            self.improved = True
        elif (self.mode == 'min' and val_metric < self.best) or \
             (self.mode == 'max' and val_metric > self.best):
            self.best = val_metric
            self.improved = True
            self.counter = 0
        else:
            self.improved = False
            self.counter += 1
            if self.verbose:
                self.logging(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
