import numpy as np

from models_from_scratch.neural_network.layers import Dense
from models_from_scratch.neural_network.activation_functions import Softmax
from models_from_scratch.neural_network.loss import CrossEntropy
from models_from_scratch.neural_network.optimizers import SGD
from models_from_scratch.neural_network.accuracy import Accuracy_Category
from models_from_scratch.neural_network.network import NeuralNetwork
from utils import train_test_split

from sklearn.datasets import load_digits

def preprocessing():
    digits = load_digits()
    X, y, y_options = digits["data"], digits["target"], digits["target_names"]
    X /= 16
    zeroes = np.zeros((y.shape[0], len(y_options)))
    zeroes[np.arange(len(y)), y] = 1
    y = zeroes
    return X, y, y_options

def main():
    X, y, y_options = preprocessing()
    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, 0.8, shuffle=True)

    layers = [
        Dense(X[0].shape[0], 64),
        Dense(64, 64),
        Dense(64, len(y_options), Softmax)
    ]
    nn = NeuralNetwork(layers)
    optimizer = SGD(learning_rate=1e-2, momentum=0.9, decay_rate=1e-4)   
    accuracy = Accuracy_Category()
    loss = CrossEntropy() 
    nn.set(optimizer, accuracy, loss)

    nn.fit((X_train, y_train), epochs=100, batch_size=32)
    print(nn.predict(X_test, y_test))

if __name__ == "__main__":
    main()
