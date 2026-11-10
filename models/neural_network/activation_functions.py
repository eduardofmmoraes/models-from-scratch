import numpy as np

class ActivationFunction:
    def __init__(self):
        pass

class ReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.dinputs = X
        return np.maximum(X, 0)

    def backward(self, dvalues):
        return np.where(self.dinputs <= 0, 0, dvalues)

class Softmax(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        exp_values = np.exp(X - np.max(X, axis=-1).reshape(-1, 1))
        return exp_values / np.sum(exp_values, axis=-1).reshape(-1, 1)

class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return 1 / (1 + np.exp(-X))

class Linear(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X