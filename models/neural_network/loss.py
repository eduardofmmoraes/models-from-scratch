import numpy as np

class Loss:
    def __init__(self):
        pass

class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_true_idx = y_true
        if len(y_true.shape) == 2:
            y_true_idx = y_true.argmax(axis=1)
        y_pred_clipped = np.clip(y_pred[np.arange(len(y_true_idx)), y_true_idx], 1e-7, 1 - 1e-7)
        return np.mean(-np.log(y_pred_clipped))