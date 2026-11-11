import numpy as np

class Loss:
    def __init__(self):
        pass

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        pass

    def backward(self, dvalues: np.ndarray):
        pass

    def reset(self):
        self.acc_sum = 0
        self.acc_count = 0

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray):
        loss = self.forward(y_pred, y_true)

        data_loss = np.mean(loss)
        self.acc_sum += np.sum(loss)
        self.acc_count += len(loss)

        return data_loss

    def calculate_acc(self):
        return self.acc_sum / self.acc_count

class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_true_idx = y_true
        if len(y_true.shape) == 2:
            y_true_idx = y_true.argmax(axis=1)
        y_pred_clipped = np.clip(y_pred[np.arange(len(y_true_idx)), y_true_idx], 1e-7, 1 - 1e-7)
        return -np.log(y_pred_clipped)