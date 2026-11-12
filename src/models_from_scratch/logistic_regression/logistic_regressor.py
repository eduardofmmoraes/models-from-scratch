import numpy as np

class LogisticRegressor:
    def __init__(self, learning_rate, threshold=0.5):
        self.learning_rate = learning_rate
        self.threshold = threshold

    def forward(self, X):
        return np.dot(X, self.weights) + self.intercept

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def loss(self, y, pred):
        return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    def accuracy(self, y, pred):
        return np.mean((pred >= self.threshold) == y)
    
    def fit(self, X, y, epochs=100):
        self.weights = 0.01 * np.random.randn(X.shape[1],)
        self.intercept = 0.01 * np.random.randn(1,)

        for epoch in range(epochs):
            z = self.forward(X)
            pred = self.sigmoid(z)
            pred = np.clip(pred, 1e-7, 1 - 1e-7)

            loss = self.loss(y, pred)
            accuracy = self.accuracy(y, pred)
            print(f"Epoch {epoch} || Loss: {loss} Accuracy: {accuracy}")

            error = pred - y
            dvalues = np.dot(error, X) / X.shape[0]
            self.weights -= self.learning_rate * dvalues
            self.intercept -= self.learning_rate * np.mean(error, axis=0)

    def predict(self, X):
        return self.forward(X)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return self.accuracy(y, pred)