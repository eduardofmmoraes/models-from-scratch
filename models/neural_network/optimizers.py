import numpy as np

from .layers import Layer

class Optimizer:
    def __init__(self):
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update_params(self, layer):
        if not hasattr(layer, "weights_mom"):
            layer.weights_mom = np.zeros(layer.weights)
            layer.biases_mom = np.zeros(layer.biases)

        layer.weights_mom = (layer.weights_mom * self.momentum) + layer.dweights
        layer.weights -= self.learning_rate * layer.dweights

        layer.biases_mom = (layer.biases_mom * self.momentum) + layer.dbiases
        layer.biases -= self.learning_rate * layer.dbiases