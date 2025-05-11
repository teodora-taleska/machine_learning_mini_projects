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

        # Compute squared norms
        A_sq = np.sum(A_2d ** 2, axis=1)[:, np.newaxis]
        B_sq = np.sum(B_2d ** 2, axis=1)[np.newaxis, :]

        distances = A_sq + B_sq - 2 * np.dot(A_2d, B_2d.T)

        result = np.exp(-self.gamma * distances)

        # Return scalar if both inputs were 1D
        if A.ndim == 1 and B.ndim == 1:
            return result[0, 0]
        # Return 1D array if one input was 1D and the other 2D
        elif A.ndim == 1 or B.ndim == 1:
            return result.ravel()
        # Return 2D array if both inputs were 2D
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
        # Convert inputs to 2D arrays if they're 1D
        A_2d = np.atleast_2d(A)
        B_2d = np.atleast_2d(B)

        # Compute dot product
        dot_product = np.dot(A_2d, B_2d.T)

        # Apply polynomial kernel
        result = (dot_product + 1) ** self.M

        # Return scalar if both inputs were 1D
        if A.ndim == 1 and B.ndim == 1:
            return result[0, 0]
        # Return 1D array if one input was 1D and the other 2D
        elif A.ndim == 1 or B.ndim == 1:
            return result.ravel()
        # Return 2D array if both inputs were 2D
        else:
            return result




if __name__ == "__main__":
    plot_sine_regression_demo()
