from .layers import Layer
from .loss import Loss, CrossEntropy
from .activation_functions import Softmax
from .optimizers import Optimizer

class NeuralNetwork:
    def __init__(self, layers: list[Layer], loss: Loss):
        self.layers = layers
        self.loss = loss()

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
            output = layer.act_func.forward(output)

        return output

    def backward(self, y, y_pred, optimizer):
        if isinstance(self.loss, CrossEntropy) and isinstance(self.layers[-1].act_func, Softmax):
            dvalues = (y_pred - y) / len(y)
        else:
            dvalues = self.loss.backward(y, y_pred)
            dvalues = self.layers[-1].act_func.backward(dvalues)
        dvalues = self.layers[-1].backward(dvalues)

        for layer in self.layers[-2::-1]:
            dvalues = layer.act_func.backward(dvalues)
            dvalues = layer.backward(dvalues)
            optimizer.update_params(layer)

    def fit(self, X, y, optimizer: Optimizer, epochs=20):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            epoch_loss = self.loss.forward(y_pred, y)
            print(f"Epoch {epoch} || loss: {epoch_loss}")
            self.backward(y, y_pred, optimizer)