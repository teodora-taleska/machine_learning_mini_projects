import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR as sklearn_SVR

# Part 1
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
        if A.ndim == 1:
            A = A[np.newaxis, :]
        if B.ndim == 1:
            B = B[np.newaxis, :]

        A_sq = np.sum(A ** 2, axis=1)[:, np.newaxis]
        B_sq = np.sum(B ** 2, axis=1)[np.newaxis, :]

        # ||x-y||² = x² - 2xy + y²
        distances = A_sq - 2 * np.dot(A, B.T) + B_sq

        gamma = 1.0 / (2 * self.sigma ** 2)
        K = np.exp(-gamma * distances)

        if K.shape == (1, 1):
            return K[0, 0]
        elif K.shape[0] == 1 or K.shape[1] == 1:
            return K.ravel()
        else:
            return K


class Polynomial:
    """
        Polynomial kernel implementation.

        ```
            The polynomial kernel is defined as:
            K(x, y) = (c + x^T)^d
        """
    def __init__(self, M):
        self.M = M

    def __call__(self, A, B):
        return (1 + np.dot(A, B.T)) ** self.M

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
        self.alpha = None
        self.alpha_star = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.y = y
        n = X.shape[0]
        K = np.asarray(self.kernel(X, X))
        C = 1 / self.lambda_

        P = np.zeros((2 * n, 2 * n))
        for i in range(n):
            for j in range(n):
                kij = K[i, j]
                P[2 * i, 2 * j] = kij  # α_i α_j term
                P[2 * i + 1, 2 * j + 1] = kij  # α_i^* α_j^* term
                P[2 * i, 2 * j + 1] = -kij  # -α_i α_j^* term
                P[2 * i + 1, 2 * j] = -kij  # -α_i^* α_j term
        P = P + 1e-8 * np.eye(2 * n)  # Stabilize P
        P = matrix(P)

        q = np.zeros(2 * n)
        for i in range(n):
            q[2 * i] = self.epsilon - y[i] # y_i - ε ... coef for α_i
            q[2 * i + 1] = self.epsilon + y[i]  # y_i + ε ... coef for α*_i
        q = matrix(q)

        G = np.vstack([-np.eye(2 * n), np.eye(2 * n)])
        h = np.hstack([np.zeros(2 * n), np.ones(2 * n) * C])
        G, h = matrix(G), matrix(h)

        A = np.zeros((1, 2 * n))
        for i in range(n):
            A[0, 2 * i] = 1 # α_i term
            A[0, 2 * i + 1] = -1 # α*_i term
        A = matrix(A)
        b = matrix(np.array([0.0])) # b = 0 .. the RHS of the constraint

        sol = solvers.qp(P, q, G, h, A, b)
        dual_vars = np.array(sol['x']).flatten()
        self.b = sol['y'][0]

        C = 1 / self.lambda_
        self.alpha = np.clip(dual_vars[0::2], 0, C)      # Clip to [0, C]
        self.alpha_star = np.clip(dual_vars[1::2], 0, C) # Clip to [0, C]
        self.X = X
        return self

    def predict(self, Xtest):
        kernel_output = self.kernel(Xtest, self.X)
        coeff = self.alpha - self.alpha_star
        return (kernel_output @ coeff + self.get_b()).flatten()

    def get_alpha(self):
        interleaved = np.empty(2 * len(self.alpha))
        interleaved[0::2] = self.alpha
        interleaved[1::2] = self.alpha_star
        return interleaved.reshape(-1, 2)  # rows: [α_i, α*_i]

    def get_b(self):
        return self.b

    def get_support_vectors(self, tol=1e-3):
        """
        Returns all support vectors (points with non-zero alpha or alpha*).
        Includes:
          1. Points on the margin (|f(x_i) - y_i| = ε ± tol).
          2. Points outside the ε-tube (|f(x_i) - y_i| > ε).
        """
        y_pred = self.predict(self.X)
        residuals = np.abs(y_pred - self.y)

        # Support vectors => points with non-zero alpha/alpha* AND error >= ε - tol
        sv_mask = ((self.alpha > tol) | (self.alpha_star > tol)) & (residuals >= self.epsilon - tol)

        return {
            'support_vectors': self.X[sv_mask],
            'target_values': self.y[sv_mask],
            'alpha': self.alpha[sv_mask],
            'alpha_star': self.alpha_star[sv_mask],
            'indices': np.where(sv_mask)[0]
        }

def plot_sine_regression_demo():
    df = pd.read_csv("sine.csv")
    X = df["x"].values.reshape(-1, 1)
    y = df["y"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rbf_kernel = RBF(sigma=1.0)
    poly_kernel = Polynomial(M=5)

    X_grid = np.linspace(min(X.ravel()) - 1, max(X.ravel()) + 1, 500).reshape(-1, 1)
    X_grid_scaled = scaler.transform(X_grid)

    models = {
        "KRR with RBF Kernel": {
            "model": KernelizedRidgeRegression(kernel=rbf_kernel, lambda_=0.1),
            "X_grid": X_grid,
            "X_used": X
        },
        "KRR with Polynomial Kernel": {
            "model": KernelizedRidgeRegression(kernel=poly_kernel, lambda_=0.001),
            "X_grid": X_grid_scaled,
            "X_used": X_scaled
        },
        "SVR with RBF Kernel": {
            "model": SVR(kernel=rbf_kernel, lambda_=0.01, epsilon=0.5),
            "X_grid": X_grid,
            "X_used": X
        },
        "SVR with Polynomial Kernel": {
            "model": SVR(kernel=poly_kernel, lambda_=0.01, epsilon=0.5),
            "X_grid": X_grid_scaled,
            "X_used": X_scaled
        }
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i, (title, entry) in enumerate(models.items()):
        model = entry["model"]
        X_used = entry["X_used"]
        X_grid_used = entry["X_grid"]

        model.fit(X_used, y)
        y_pred = model.predict(X_grid_used)

        axes[i].scatter(X_used, y, color="black", label="Data")
        axes[i].plot(X_grid_used, y_pred, color="blue", label="Prediction")

        if isinstance(model, SVR):
            y_pred_upper = y_pred + model.epsilon
            y_pred_lower = y_pred - model.epsilon
            axes[i].plot(X_grid_used, y_pred_upper, color="blue", linestyle="--", label="ε-tube")
            axes[i].plot(X_grid_used, y_pred_lower, color="blue", linestyle="--")

            sv_info = model.get_support_vectors()

            axes[i].scatter(
                sv_info['support_vectors'],
                sv_info['target_values'],
                facecolors='none',
                edgecolors='red',
                s=100,
                linewidths=2,
                label=f"SVs ({len(sv_info['indices'])})"
            )

        axes[i].set_title(title)
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].legend()
        axes[i].grid(True)

    fig.tight_layout()
    plt.savefig("visualizations/sine_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_sine_regression_demo_sklearn():
    df = pd.read_csv("sine.csv")
    X = df["x"].values.reshape(-1, 1)
    y = df["y"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_grid = np.linspace(min(X.ravel()) - 1, max(X.ravel()) + 1, 500).reshape(-1, 1)
    X_grid_scaled = scaler.transform(X_grid)

    models = {
        "KRR with RBF Kernel": {
            "model": KernelRidge(alpha=0.1, kernel='rbf', gamma=1.0),  # gamma = 1/(2*sigma^2)
            "X_grid": X_grid,
            "X_used": X
        },
        "KRR with Polynomial Kernel": {
            "model": KernelRidge(alpha=0.001, kernel='poly', degree=5, coef0=1),
            "X_grid": X_grid_scaled,
            "X_used": X_scaled
        },
        "SVR with RBF Kernel": {
            "model": sklearn_SVR(kernel='rbf', C=100, epsilon=0.5, gamma=1.0),  # C = 1/lambda
            "X_grid": X_grid,
            "X_used": X
        },
        "SVR with Polynomial Kernel": {
            "model": sklearn_SVR(kernel='poly', C=100, epsilon=0.5, degree=5, coef0=1),
            "X_grid": X_grid_scaled,
            "X_used": X_scaled
        }
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i, (title, entry) in enumerate(models.items()):
        model = entry["model"]
        X_used = entry["X_used"]
        X_grid_used = entry["X_grid"]

        model.fit(X_used, y)
        y_pred = model.predict(X_grid_used)

        axes[i].scatter(X_used, y, color="black", label="Data")
        axes[i].plot(X_grid_used, y_pred, color="blue", label="Prediction")

        if isinstance(model, sklearn_SVR):
            y_pred_upper = y_pred + model.epsilon
            y_pred_lower = y_pred - model.epsilon
            axes[i].plot(X_grid_used, y_pred_upper, color="blue", linestyle="--", label="ε-tube")
            axes[i].plot(X_grid_used, y_pred_lower, color="blue", linestyle="--")

            axes[i].scatter(
                X_used[model.support_],
                y[model.support_],
                facecolors='none',
                edgecolors='red',
                s=100,
                linewidths=2,
                label=f"SVs ({len(model.support_)})"
            )

        axes[i].set_title(title)
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].legend()
        axes[i].grid(True)

    fig.tight_layout()
    plt.savefig("visualizations/sine_plot_sklearn.png", dpi=300, bbox_inches='tight')
    plt.show()



# Part 2
def load_housing_data():
    df = pd.read_csv('housing2r.csv')
    X = df.drop(columns=['y']).values
    y = df['y'].values
    return X, y


def nested_cv_with_plots(model_class, kernel_class, X, y, outer_cv=5, inner_cv=5,
                         param_grid={}, model_name=''):
    epsilon = 0.5 if model_name == 'SVR' else None
    kernel_params = param_grid['kernel_param']
    lambdas = param_grid['lambda']

    results = {
        'kernel_params': kernel_params,
        'mse_lambda_1': [],
        'mse_best_lambda': [],
        'best_lambdas_per_kernel': [],
        'support_vectors_lambda_1': [],
        'support_vectors_best_lambda': []
    }

    # Outer loop over kernel parameters
    for kernel_param in kernel_params:
        outer_mse_fixed = []
        outer_mse_best = []
        sv_counts_fixed = []
        sv_counts_best = []
        best_lambdas = []

        # Outer CV loop
        outer_kf = KFold(n_splits=outer_cv, shuffle=True, random_state=42)

        for train_idx, test_idx in outer_kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # --- 1. Evaluate with lambda=1 on test set ---
            if model_name == 'SVR':
                model_fixed = model_class(kernel=kernel_class(kernel_param), lambda_=1.0, epsilon=epsilon)
            else:
                model_fixed = model_class(kernel=kernel_class(kernel_param), lambda_=1.0)

            model_fixed.fit(X_train, y_train)
            y_pred = model_fixed.predict(X_test)
            outer_mse_fixed.append(mean_squared_error(y_test, y_pred))

            if model_name == 'SVR':
                sv_info = model_fixed.get_support_vectors()
                sv_counts_fixed.append(len(sv_info['indices']))

            # --- 2. Inner CV to find best lambda ---
            best_inner_mse = np.inf
            best_lambda = None

            for lam in lambdas:
                if model_name == 'SVR':
                    model_cv = model_class(kernel=kernel_class(kernel_param), lambda_=lam, epsilon=epsilon)
                else:
                    model_cv = model_class(kernel=kernel_class(kernel_param), lambda_=lam)

                inner_mses = []
                inner_kf = KFold(n_splits=inner_cv, shuffle=True, random_state=42)

                for inner_train_idx, inner_val_idx in inner_kf.split(X_train):
                    X_inner_train, X_inner_val = X_train[inner_train_idx], X_train[inner_val_idx]
                    y_inner_train, y_inner_val = y_train[inner_train_idx], y_train[inner_val_idx]

                    model_cv.fit(X_inner_train, y_inner_train)
                    y_pred = model_cv.predict(X_inner_val)
                    inner_mses.append(mean_squared_error(y_inner_val, y_pred))

                avg_inner_mse = np.mean(inner_mses)
                if avg_inner_mse < best_inner_mse:
                    best_inner_mse = avg_inner_mse
                    best_lambda = lam

            best_lambdas.append(round(best_lambda, 2))

            # --- 3. Evaluate best lambda on test set ---
            if model_name == 'SVR':
                model_best = model_class(kernel=kernel_class(kernel_param), lambda_=best_lambda, epsilon=epsilon)
            else:
                model_best = model_class(kernel=kernel_class(kernel_param), lambda_=best_lambda)

            model_best.fit(X_train, y_train)
            y_pred = model_best.predict(X_test)
            outer_mse_best.append(mean_squared_error(y_test, y_pred))

            if model_name == 'SVR':
                sv_info = model_best.get_support_vectors()
                sv_counts_best.append(len(sv_info['indices']))

        results['mse_lambda_1'].append(np.mean(outer_mse_fixed))
        results['mse_best_lambda'].append(np.mean(outer_mse_best))
        results['best_lambdas_per_kernel'].append(np.mean(best_lambdas))

        if model_name == 'SVR':
            results['support_vectors_lambda_1'].append(np.mean(sv_counts_fixed))
            results['support_vectors_best_lambda'].append(np.mean(sv_counts_best))
        else:
            results['support_vectors_lambda_1'].append(None)
            results['support_vectors_best_lambda'].append(None)

    return results


def analyze_housing_data():
    lambda_values = [0.001, 0.01, 0.1, 0.5, 1.0]
    sigma_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    degree_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    param_grid_rbf = {'kernel_param': sigma_values, 'lambda': lambda_values}
    param_grid_poly = {'kernel_param': degree_values, 'lambda': lambda_values}

    X, y = load_housing_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    results_rbf_KRR = nested_cv_with_plots(KernelizedRidgeRegression, lambda s: RBF(sigma=s), X_train, y_train,
                                            param_grid=param_grid_rbf,
                                            model_name='KRR')

    results_poly_KRR = nested_cv_with_plots(KernelizedRidgeRegression, lambda d: Polynomial(M=d), X_train, y_train,
                                            param_grid=param_grid_poly,
                                            model_name='KRR')

    results_rbf_SVR = nested_cv_with_plots(SVR, lambda s: RBF(sigma=s), X_train, y_train,
                                           param_grid=param_grid_rbf,
                                           model_name='SVR')

    results_poly_SVR = nested_cv_with_plots(SVR, lambda d: Polynomial(M=d), X_train, y_train,
                                            param_grid=param_grid_poly,
                                            model_name='SVR')

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(results_rbf_KRR['kernel_params'], results_rbf_KRR['mse_lambda_1'], marker='o', label='KRR RBF λ=1')
    axes[0, 0].plot(results_rbf_KRR['kernel_params'], results_rbf_KRR['mse_best_lambda'], marker='s',
                    label=f'KRR RBF best λ')
    axes[0, 0].set_xlabel('sigma')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('KRR with RBF Kernel')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(results_poly_KRR['kernel_params'], results_poly_KRR['mse_lambda_1'], marker='o', label='KRR Polynomial λ=1')
    axes[0, 1].plot(results_poly_KRR['kernel_params'], results_poly_KRR['mse_best_lambda'], marker='s',
                    label=f'KRR Poly best λ')
    axes[0, 1].set_xlabel('degree')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('KRR with Polynomial Kernel')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(
        results_rbf_SVR['kernel_params'], results_rbf_SVR['mse_lambda_1'],
        marker='o', label=f'SVR RBF λ=1 (Avg SVs={np.mean(results_rbf_SVR["support_vectors_lambda_1"]):.1f})'
    )
    axes[1, 0].plot(
        results_rbf_SVR['kernel_params'], results_rbf_SVR['mse_best_lambda'],
        marker='s', label=f'SVR RBF best λ (Avg SVs={np.mean(results_rbf_SVR["support_vectors_best_lambda"]):.1f})',
    )
    axes[1, 0].set_xlabel('sigma')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('SVR with RBF Kernel')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(
        results_poly_SVR['kernel_params'], results_poly_SVR['mse_lambda_1'],
        marker='o', label=f'SVR RBF λ=1 (Avg SVs={np.mean(results_poly_SVR["support_vectors_lambda_1"]):.1f})'
    )
    axes[1, 1].plot(
        results_poly_SVR['kernel_params'], results_poly_SVR['mse_best_lambda'],
        marker='s', label=f'SVR RBF best λ (Avg SVs={np.mean(results_poly_SVR["support_vectors_best_lambda"]):.1f})',
    )
    axes[1, 1].set_xlabel('degree')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('SVR with Polynomial Kernel')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    for i, param in enumerate(results_rbf_SVR['kernel_params']):
        axes[1, 0].annotate(f'{results_rbf_SVR["support_vectors_lambda_1"][i]}',
                            (param, results_rbf_SVR['mse_lambda_1'][i]),
                            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='blue')
        axes[1, 0].annotate(f'{results_rbf_SVR["support_vectors_best_lambda"][i]}',
                            (param, results_rbf_SVR['mse_best_lambda'][i]),
                            textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8, color='orange')

    for i, param in enumerate(results_poly_SVR['kernel_params']):
        axes[1, 1].annotate(f'{results_poly_SVR["support_vectors_lambda_1"][i]}',
                            (param, results_poly_SVR['mse_lambda_1'][i]),
                            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='blue')
        axes[1, 1].annotate(f'{results_poly_SVR["support_vectors_best_lambda"][i]}',
                            (param, results_poly_SVR['mse_best_lambda'][i]),
                            textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8, color='orange')


    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    plt.savefig('visualizations/compare_KRR_SVR_kernels_individual.png')
    plt.show()


if __name__ == "__main__":
    # plot_sine_regression_demo()
    # plot_sine_regression_demo_sklearn()
    analyze_housing_data()
