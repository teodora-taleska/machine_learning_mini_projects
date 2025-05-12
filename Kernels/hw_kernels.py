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

class SVR:
    def __init__(self, kernel, lambda_, epsilon):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.b = None

    def fit(self, X, y):
        n = X.shape[0]
        K = np.asarray(self.kernel(X, X))
        C = 1 / self.lambda_

        P = np.zeros((2 * n, 2 * n))
        for i in range(n):
            for j in range(n):
                kij = K[i, j]
                # interleaved: P[2i, 2j] = K, P[2i+1, 2j+1] = K, P[2i, 2j+1] = -K, ...
                P[2 * i, 2 * j] = kij           # α_i * α_j
                P[2 * i + 1, 2 * j + 1] = kij   # α*_i * α*_j
                P[2 * i, 2 * j + 1] = -kij      # α_i * α*_j
                P[2 * i + 1, 2 * j] = -kij      # α*_i * α_j
        P = matrix(P)

        q = np.zeros(2 * n)
        for i in range(n):
            q[2 * i]     = self.epsilon - y[i]  # for α_i
            q[2 * i + 1] = self.epsilon + y[i]  # for α*_i
        q = matrix(q)

        G = np.vstack([-np.eye(2 * n), np.eye(2 * n)])
        h = np.hstack([np.zeros(2 * n), np.ones(2 * n) * C])
        G, h = matrix(G), matrix(h)

        # Constraints: α_i + α*_i = 0
        A = np.zeros((1, 2 * n))
        for i in range(n):
            A[0, 2 * i] = 1       # α_i
            A[0, 2 * i + 1] = -1  # α*_i
        A = matrix(A)
        b = matrix(np.array([0.0]))

        sol = solvers.qp(P, q, G, h, A, b)
        dual_vars = np.array(sol['x']).flatten()
        self.b = sol['y'][0]

        self.alpha = dual_vars[0::2]      # even indices
        self.alpha_star = dual_vars[1::2] # odd indices
        self.X = X
        self.y = y
        self.K = K
        return self

    def predict(self, Xtest):
        kernel_output = self.kernel(Xtest, self.X)
        coeff = self.alpha - self.alpha_star
        return (kernel_output @ coeff + self.get_b()).flatten() # (1, n) instead of (n,)

    def get_alpha(self):
        interleaved = np.empty(2 * len(self.alpha))
        interleaved[0::2] = self.alpha
        interleaved[1::2] = self.alpha_star
        return interleaved.reshape(-1, 2)  # rows: [α_i, α*_i]

    def get_b(self):
        return self.b


# if __name__ == "__main__":
#     plot_sine_regression_demo()
