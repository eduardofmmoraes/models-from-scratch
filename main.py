import numpy as np

from models.neural_network.layers import Dense
from models.neural_network.activation_functions import Softmax
from models.neural_network.loss import CrossEntropy
from models.neural_network.network import NeuralNetwork

def main():
    layers = [
        Dense(4, 16),
        Dense(16, 16),
        Dense(16, 8),
        Dense(8, 3, Softmax)
    ]
    nn = NeuralNetwork(layers, CrossEntropy)

    inputs = np.random.randn(5, 4)
    X, y = inputs[:, :4], np.random.choice(3, 5)
    nn.fit(X, y)

if __name__ == "__main__":
    # main()
    print([1, 2, 3, 4, 5][-2::-1])