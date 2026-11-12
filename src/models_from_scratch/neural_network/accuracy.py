import numpy as np

class Accuracy:
    def __init__(self):
        pass

    def reset(self):
        self.acc_sum = 0
        self.acc_count = 0
    
    def calculate(self, y_pred, y_true):
        comp = self.compare(y_pred, y_true)

        accuracy = np.mean(comp)
        self.acc_sum += np.sum(comp)
        self.acc_count += len(comp)

        return accuracy

    def calculate_acc(self):
        return self.acc_sum / self.acc_count

class Accuracy_Category(Accuracy):
    def __init__(self):
        super().__init__()

    def compare(self, y_pred, y_true):
        if len(y_true.shape) == 2: y_true = np.argmax(y_true, axis=-1)
        return np.argmax(y_pred, axis=-1) == y_true