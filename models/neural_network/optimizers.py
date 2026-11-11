import numpy as np

from .layers import Layer

class Optimizer:
    def __init__(self):
        self.step = 0
        pass

    def pre_update_params(self, step):
        pass

    def update_params(self, layers):
        pass

    def post_update_params(self, step):
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0, decay_rate=0):
        super().__init__()
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay_rate = decay_rate

    def pre_update_params(self, step):
        pass

    def update_params(self, layer):
        if not hasattr(layer, "weights_mom"):
            layer.weights_mom = np.zeros_like(layer.weights)
            layer.biases_mom = np.zeros_like(layer.biases)

        weights_updates = (layer.weights_mom * self.momentum) - self.learning_rate * layer.dweights
        layer.weights_mom = weights_updates
        layer.weights += weights_updates

        biases_updates = (layer.biases_mom * self.momentum) - self.learning_rate * layer.dbiases
        layer.biases_mom = biases_updates
        layer.biases += biases_updates

    def post_update_params(self, step):
        self.learning_rate = self.base_learning_rate * np.exp(-self.decay_rate * step)
        self.step += 1