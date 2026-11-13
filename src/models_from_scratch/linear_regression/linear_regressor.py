import numpy as np

from typing import Literal

class LinearRegressor:
    def __init__(self):
        pass

    def matrix_based(self, X, y):
        X = np.vstack([np.ones(y.shape), X])
        XtX1 = np.linalg.inv(np.dot(X.T, X))
        X_dot = np.dot(XtX1, X.T)
        return np.dot(X_dot, y)

    def gradient_descent(self, X, y):
        pass

    def fit(self, X, mode: Literal["matrix", "gradient"]):
