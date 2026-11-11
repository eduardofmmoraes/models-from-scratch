import numpy as np

from .activation_functions import ActivationFunction, ReLU

class Layer:
    def __init__(self):
        pass

    def forward(self, X):
        pass

    def backward(self, dvalues):
        pass

class Dense(Layer):
    def __init__(self, n_inputs, size, activation_function: ActivationFunction = ReLU):
        super().__init__()
        self.weights = 0.01 * np.random.randn(n_inputs, size)
        self.biases = np.zeros((1, size))
        self.act_func = activation_function()

    def forward(self, X):
        self.inputs = X
        return np.dot(X, self.weights) + self.biases

    def backward(self, dvalues):
        dinputs = np.dot(dvalues, self.weights.T)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        return dinputs