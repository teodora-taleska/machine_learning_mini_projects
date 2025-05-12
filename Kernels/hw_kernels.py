import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import pandas as pd


class RBF:
    """
    Radial Basis Function (Gaussian) kernel implementation.

    ```
        The RBF kernel is defined as:
        K(x, y) = exp(-gamma * ||x - y||^2)
        where gamma = 1/(2*sigma^2)
    """
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.gamma = 1.0 / (2 * sigma ** 2)

    def __call__(self, A, B):
        if B is None:
            B = A

        # Convert inputs to 2D arrays if they're 1D
        A_2d = np.atleast_2d(A)
        B_2d = np.atleast_2d(B)

        # squared norms
        A_sq = np.sum(A_2d ** 2, axis=1)[:, np.newaxis]
        B_sq = np.sum(B_2d ** 2, axis=1)[np.newaxis, :]

        distances = A_sq + B_sq - 2 * np.dot(A_2d, B_2d.T)

        result = np.exp(-self.gamma * distances)

        if A.ndim == 1 and B.ndim == 1:
            return result[0, 0] # scalar
        elif A.ndim == 1 or B.ndim == 1:
            return result.ravel() # 1D array
        else:
            return result

class Polynomial:
    """
    Polynomial kernel implementation.

    ```
        The polynomial kernel is defined as:
        K(x, y) = (x^T y + c)^d

        In our implementation:
        - M parameter corresponds to the degree d
        - We use c = 1 as a default offset
    """
    def __init__(self, M=2):
        self.M = M

    def __call__(self, A, B):
        A_2d = np.atleast_2d(A)
        B_2d = np.atleast_2d(B)

        dot_product = np.dot(A_2d, B_2d.T)

        # apply the kernel
        result = (dot_product + 1) ** self.M

        if A.ndim == 1 and B.ndim == 1:
            return result[0, 0]
        elif A.ndim == 1 or B.ndim == 1:
            return result.ravel()
        else:
            return result

class KernelizedRidgeRegression:
    """
    Kernel ridge regression:
    - Uses the kernel trick to perform regression in high-dimensional feature spaces
    - Solves for alpha coefficients using the closed-form solution
    - Makes predictions using kernel evaluations between test points and training points

    The mathematical formulation is:
    alpha = (K + lambda*I)^-1 y
    where K is the kernel matrix of training data
    """
    def __init__(self, kernel, lambda_=1.0):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.alpha = None
        self.X_train = None

    def fit(self, X, y):
        self.X_train = X

        K = self.kernel(X, X)

        n = K.shape[0]
        K_reg = K + self.lambda_ * np.eye(n) # Add regularization to the diagonal

        self.alpha = np.linalg.solve(K_reg, y)

        return self

    def predict(self, X):
        if self.alpha is None or self.X_train is None:
            raise RuntimeError("Model must be fitted before making predictions")
        K_test = self.kernel(X, self.X_train)
        return K_test @ self.alpha



# if __name__ == "__main__":
#     plot_sine_regression_demo()
