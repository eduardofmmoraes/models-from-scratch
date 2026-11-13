import numpy as np

class ActivationFunction:
    def __init__(self):
        pass

    def forward(self, X):
        pass

    def backward(self, dvalues):
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
        exp_values = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.inputs = X
        self.outputs = 1 / (1 + np.exp(-X))
        return self.outputs

    def backward(self, dvalues):
        return dvalues * self.outputs * (1 - self.outputs)

class Linear(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X

    def backward(self, dvalues):
        return dvalues