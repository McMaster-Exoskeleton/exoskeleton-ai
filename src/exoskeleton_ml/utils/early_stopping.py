"""Early stopping utility to prevent overfitting."""


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve.

    Tracks the validation loss and stops training if it doesn't improve
    for a specified number of epochs (patience).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping if no improvement.
            min_delta: Minimum change in monitored value to qualify as improvement.
            mode: One of 'min' or 'max'. In 'min' mode, training stops when the
                  monitored quantity stops decreasing. In 'max' mode, it stops
                  when the quantity stops increasing.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, current_score: float) -> bool:
        """Check if training should stop.

        Args:
            current_score: Current validation metric (loss or accuracy).

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n⚠️  Early stopping triggered after {self.counter} epochs without improvement")
                return True

        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False
