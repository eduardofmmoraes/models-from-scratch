
import numpy as np

from .layers import Layer
from .loss import Loss, CrossEntropy
from .activation_functions import Softmax
from .optimizers import Optimizer
from .accuracy import Accuracy

class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def set(self, optimizer: Optimizer, accuracy: Accuracy, loss: Loss):
        self.loss = loss
        self.accuracy = accuracy
        self.optimizer = optimizer

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
            output = layer.act_func.forward(output)

        return output

    def backward(self, y, y_pred):
        if len(y.shape) == 2: y = np.argmax(y, axis=-1)
        if isinstance(self.loss, CrossEntropy) and isinstance(self.layers[-1].act_func, Softmax):
            dvalues = y_pred.copy()
            dvalues[range(len(dvalues)), y] -= 1
            dvalues /= len(dvalues)
        else:
            dvalues = self.loss.backward(y, y_pred)
            dvalues = self.layers[-1].act_func.backward(dvalues)
        dvalues = self.layers[-1].backward(dvalues)

        for layer in self.layers[-2::-1]:
            dvalues = layer.act_func.backward(dvalues)
            dvalues = layer.backward(dvalues)

    def get_batches(self, X, y, batch_size):
        n = y.shape[0]
        batches_num = int(n / batch_size)

        X_batch, y_batch = [], []
        for i in range(batches_num):
            X_batch.append(X[i * batch_size:(i + 1) * batch_size])
            y_batch.append(y[i * batch_size:(i + 1) * batch_size])

        if batches_num * batch_size < n:
            X_batch.append(X[batches_num * batch_size:])
            y_batch.append(y[batches_num * batch_size:])
            batches_num += 1

        return X_batch, y_batch, batches_num

    def fit(self, train_ds, epochs=20, batch_size=16):
        X_train, y_train = train_ds
        X_train_batches, y_train_batches, n_batches = self.get_batches(X_train, y_train, batch_size)
        for epoch in range(epochs):
            self.accuracy.reset()
            self.loss.reset()
            print(f"========== {epoch} ==========")
            for batch in range(n_batches):
                X_train_batch, y_train_batch = X_train_batches[batch], y_train_batches[batch]
                y_pred = self.forward(X_train_batch)

                loss = self.loss.calculate(y_pred, y_train_batch)
                accuracy = self.accuracy.calculate(y_pred, y_train_batch)

                print(f"\tBatch {batch} || loss: {loss} acc: {accuracy} lr: {self.optimizer.learning_rate}")
            
                epoch_loss = self.loss.calculate_acc()
                epoch_accuracy = self.accuracy.calculate_acc()
                self.backward(y_train_batch, y_pred)

                self.optimizer.pre_update_params(batch + epoch * n_batches)
                for layer in self.layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params(batch + epoch * n_batches)

            print(f"Epoch {epoch} || loss: {epoch_loss} acc: {epoch_accuracy} lr: {self.optimizer.learning_rate}")

    def predict(self, X, y):
        y_pred = self.forward(X)
        return self.accuracy.calculate(y_pred, y)