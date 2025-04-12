import numpy as np
import unittest
from sklearn.linear_model import LogisticRegression
from scipy.optimize import check_grad, minimize, fmin_l_bfgs_b

class MultinomialLogReg:
    """Multinomial Logistic Regression classifier with L-BFGS-B optimization.

        Parameters:
            max_iter : int, default=1000
                Maximum number of iterations for the optimizer
            tol : float, default=1e-6
                Convergence tolerance for the optimizer
            lb : float, optional
                Lower bound for coefficients (for regularization)
            ub : float, optional
                Upper bound for coefficients (for regularization)
    """

    def __init__(self, max_iter=1000, tol=1e-6, lb=None, ub=None):
        self.max_iter = max_iter
        self.tol = tol
        self.lb = lb
        self.ub = ub
        self.coef_ = None        # weights, shape (m-1, k)
        self.intercept_ = None   # bias terms, shape (m-1,)
        self.classes_ = None

    def _softmax(self, logits):
        """Compute softmax probabilities numerically stable.

            Args:
                logits : np.ndarray
                    Input logits (shape: [n_samples, n_classes])

            Returns:
                np.ndarray of probabilities (shape: [n_samples, n_classes])
        """
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _log_likelihood(self, beta, X, y):
        """Compute the log-likelihood of the parameters.

            Args:
                beta : np.ndarray
                    Flattened array containing both weights and biases
                X : np.ndarray
                    Feature matrix (shape: [n_samples, n_features])
                y : np.ndarray
                    Target labels (shape: [n_samples])

            Returns:
                float: The log-likelihood of the current parameters
        """
        m = len(self.classes_)
        k = X.shape[1]

        W = beta[: (m - 1) * k].reshape((m - 1, k))        # weights
        b = beta[(m - 1) * k:]                             # biases

        logits = X @ W.T + b  # shape (n, m-1)
        logits_full = np.column_stack([logits, np.zeros(X.shape[0])])  # add zero column for reference class
        probs = self._softmax(logits_full)

        return np.sum(np.log(probs[np.arange(len(y)), y]))

    def _gradient(self, beta, X, y):
        """Compute the gradient of the log-likelihood.

            Args:
                beta : np.ndarray
                    Flattened parameter vector
                X : np.ndarray
                    Feature matrix
                y : np.ndarray
                    Target labels

            Returns:
                np.ndarray: Flattened gradient vector
        """
        m = len(self.classes_)
        k = X.shape[1]

        W = beta[: (m - 1) * k].reshape((m - 1, k))
        b = beta[(m - 1) * k:]

        logits = X @ W.T + b
        probs = self._softmax(np.column_stack([logits, np.zeros(X.shape[0])]))

        grad_W = np.zeros_like(W)
        grad_b = np.zeros_like(b)

        for j in range(m - 1):
            indicator = (y == self.classes_[j]).astype(float)
            grad_W[j] = X.T @ (indicator - probs[:, j])
            grad_b[j] = np.sum(indicator - probs[:, j])

        return np.concatenate([grad_W.ravel(), grad_b])

    def build(self, X, y):
        """Fit the model to training data using L-BFGS-B optimization.

            Args:
               X : np.ndarray
                    Training features (shape: [n_samples, n_features])
               y : np.ndarray
                    Training labels (shape: [n_samples])

            Returns:
                self: The fitted model
        """
        self.classes_ = np.unique(y)
        n, k = X.shape
        m = len(self.classes_)

        beta_init = np.zeros((m - 1) * k + (m - 1))

        bounds = [(self.lb, self.ub)] * len(beta_init) if (self.lb is not None or self.ub is not None) else None

        beta_opt, _, _ = fmin_l_bfgs_b(
            func=lambda b: -self._log_likelihood(b, X, y),
            x0=beta_init,
            fprime=lambda b: -self._gradient(b, X, y),
            bounds=bounds,
            maxiter=self.max_iter,
        )

        self.coef_ = beta_opt[: (m - 1) * k].reshape((m - 1, k))
        self.intercept_ = beta_opt[(m - 1) * k:]

        # numeric gradient check
        beta_test = np.random.randn((m - 1) * k + (m - 1))
        grad_error = check_grad(
            # check grad computes the numerical gradient and compares it to the analytical gradient
            lambda b: -self._log_likelihood(b, X, y),
            lambda b: -self._gradient(b, X, y),
            beta_test,
        )

        if grad_error < 1e-5:
            print(f"✅ Excellent gradient (error = {grad_error:.2e})")
        elif grad_error < 1e-3:
            print(f"⚠️ Acceptable gradient (error = {grad_error:.2e}), but could be improved")
        else:
            print(f"❌ Critical gradient error (error = {grad_error:.2e}) - Implementation is wrong!")

        return self

    def predict(self, X):
        """Predict class probabilities for new samples.

            Args:
                X : np.ndarray
                    Feature matrix (shape: [n_samples, n_features])

            Returns:
                np.ndarray: Probability matrix (shape: [n_samples, n_classes])
        """
        logits = X @ self.coef_.T + self.intercept_
        logits_full = np.column_stack([logits, np.zeros(X.shape[0])])
        return self._softmax(logits_full)


class OrdinalLogReg:
    def __init__(self, epsilon=1e-8):
        """
            Initialize ordinal logistic regression model.

            Parameters:
            - epsilon: small constant to avoid numerical issues (default: 1e-8)
        """
        self.epsilon = epsilon
        self.coef_ = None
        self.deltas_ = None
        self.thresholds_ = None
        self.n_classes = None

    def _logistic_cdf(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _compute_thresholds(self, deltas):
        """Convert deltas to ordered thresholds.
            Handles cases for:
            - 1 class: [-inf, inf] (degenerate case)
            - 2 classes: [-inf, 0, inf] (standard logistic regression)
            - n classes: [-inf, 0, delta1, delta1+delta2, ..., inf]
        """
        thresholds = np.zeros(self.n_classes + 1)
        thresholds[0] = -np.inf

        if self.n_classes == 1:
            thresholds[1] = np.inf
        elif self.n_classes == 2:
            thresholds[1] = 0
            thresholds[2] = np.inf
        else:
            thresholds[1] = 0
            thresholds[2:-1] = np.cumsum(deltas)
            thresholds[-1] = np.inf

        return thresholds

    def _compute_probs(self, X, coef, thresholds):
        """Compute class probabilities for given parameters."""
        u = X @ coef
        probs = np.zeros((X.shape[0], self.n_classes))

        # special handling for binary case
        if self.n_classes == 2:
            # P(Y=2) = σ(βX)
            probs[:, 1] = self._logistic_cdf(u)
            probs[:, 0] = 1 - probs[:, 1]
        else:
            # Multi-class ordinal case
            cdf_probs = np.zeros((X.shape[0], self.n_classes + 1))
            cdf_probs[:, 0] = 0  # P(Y ≤ 0) = 0
            cdf_probs[:, -1] = 1  # P(Y ≤ K) = 1
            for j in range(1, self.n_classes):
                cdf_probs[:, j] = self._logistic_cdf(thresholds[j] - u)

            # P(Y = j) = P(Y ≤ j) - P(Y ≤ j-1)
            for j in range(self.n_classes):
                probs[:, j] = cdf_probs[:, j + 1] - cdf_probs[:, j]

        # for numerical stability
        probs = np.clip(probs, self.epsilon, 1 - self.epsilon)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def _negative_log_likelihood(self, params, X, y):
        """Compute negative log-likelihood for optimization."""
        n_features = X.shape[1]
        coef = params[:n_features]
        deltas = params[n_features:]

        thresholds = self._compute_thresholds(deltas)

        probs = self._compute_probs(X, coef, thresholds) # Get P(Y=j|X) for all j

        sample_probs = probs[np.arange(len(y)), y - 1]  # y is 1-based |  get P(Y=y_i|X_i) for each sample

        log_likelihood = np.sum(np.log(sample_probs))
        return -log_likelihood

    def build(self, X, y, init_coef=None, init_deltas=None, maxiter=1000):
        """
        Fit the ordinal logistic regression model using maximum likelihood estimation.

            Parameters:
            - X: input features (n_samples, n_features)
            - y: target labels (1-based, shape n_samples)
            - init_coef: initial coefficients (optional)
            - init_deltas: initial deltas (optional)
            - maxiter: maximum number of iterations for optimization

            Returns:
            - self: fitted model
        """

        # standardising X for better numerical stability
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        n_samples, n_features = X.shape

        unique_classes = np.unique(y)
        self.n_classes = len(unique_classes)

        if init_coef is None:
            init_coef = np.zeros(X.shape[1])
        if init_deltas is None:
            # Smaller initial deltas work better - we need (n_classes - 2) deltas (since t1=0 and tn=inf)
            init_deltas = np.ones(self.n_classes - 2) * 0.1

        initial_params = np.concatenate([init_coef, init_deltas])

        # def. bounds - coefficients are unbounded, deltas must be positive
        bounds = [(None, None) for _ in range(n_features)] + \
                 [(self.epsilon, None) for _ in range(self.n_classes - 2)]

        result = minimize(
            fun=self._negative_log_likelihood,
            x0=initial_params,
            args=(X, y),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter}
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        self.coef_ = result.x[:n_features]
        self.deltas_ = result.x[n_features:]
        self.thresholds_ = self._compute_thresholds(self.deltas_)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

            Parameters:
            - X: input features (n_samples, n_features)

            Returns:
            - probs: array of shape (n_samples, n_classes) with class probabilities
        """
        if self.coef_ is None or self.thresholds_ is None:
            raise RuntimeError("Model has not been fitted yet. Call build() first.")

        return self._compute_probs(X, self.coef_, self.thresholds_)

    def predict(self, X):
        """
        Predict class labels for input samples (returns class with highest probability).

            Parameters:
            - X: input features (n_samples, n_features)

            Returns:
            - y_pred: array of shape (n_samples,) with predicted class labels (1-based)
        """
        probs = self.predict_proba(X)
        return probs

class MyTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]])
        self.y_binary = np.array([0, 0, 1, 1, 1])
        self.y_multiclass = np.array([0, 0, 1, 1, 2])  # 3 classes (0, 1, 2)
        self.y_ordinal = np.array([1, 1, 2, 3, 4])  # 4 ordinal classes (1 < 2 < 3 < 4)

    def test_binary_predictions(self):
        our_model = MultinomialLogReg(max_iter=1000, tol=1e-6)
        our_model.build(self.X, self.y_binary)
        print('Our model coef:',our_model.coef_)
        our_probs = our_model.predict(self.X)

        sklearn_model = LogisticRegression(
            # multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            tol=1e-6,
            penalty=None  # disable regularization for fair comparison
        )
        sklearn_model.fit(self.X, self.y_binary)
        sklearn_probs = sklearn_model.predict_proba(self.X)
        print('Sklearn coef', sklearn_model.coef_)

        np.testing.assert_allclose(
            our_probs,
            sklearn_probs,
            atol=1e-4,
            rtol=1e-4
        )
        print(our_probs)
        print(sklearn_probs)

    def test_multiclass_predictions(self):
        """Test multiclass classification (3 classes)."""
        our_model = MultinomialLogReg(max_iter=1000, tol=1e-6)
        our_model.build(self.X, self.y_multiclass)
        print('Our model coef:', our_model.coef_)
        our_probs = our_model.predict(self.X)

        sklearn_model = LogisticRegression(
            # multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            tol=1e-6,
            penalty=None  # disable regularization for fair comparison
        )
        sklearn_model.fit(self.X, self.y_multiclass)
        sklearn_probs = sklearn_model.predict_proba(self.X)
        print('Sklearn coef', sklearn_model.coef_)

        np.testing.assert_allclose(
            our_probs,
            sklearn_probs,
            atol=1e-4,
            rtol=1e-4
        )
        print(our_probs)
        print(sklearn_probs)

    def test_ord_binary(self):
        model = OrdinalLogReg()
        model.build(self.X, self.y_binary)
        probs = model.predict_proba(self.X)

        print("\nBinary case comparison:")
        print("Our probs:", probs[:, 1])

        self.assertTrue((probs <= 1).all())
        self.assertTrue((probs >= 0).all())
        np.testing.assert_almost_equal(probs.sum(axis=1), 1)

    def test_ord_against_sklearn(self):
        model = OrdinalLogReg()
        model.build(self.X, self.y_ordinal)
        probs = model.predict_proba(self.X)

        print("\nFour level case comparison:")
        print("probs:", probs[:, 1])
        self.assertTrue((probs <= 1).all())
        self.assertTrue((probs >= 0).all())
        np.testing.assert_almost_equal(probs.sum(axis=1), 1)

    def test_threshold_ordering(self):
        model = OrdinalLogReg()
        model.build(self.X, self.y_ordinal)

        # Thresholds should be: -inf, 0, t2, t3, ..., inf
        # With t2 < t3 < ...
        thresholds = model.thresholds_
        self.assertEqual(thresholds[0], -np.inf)
        self.assertEqual(thresholds[1], 0)
        self.assertTrue((np.diff(thresholds[1:-1]) > 0).all())  # Strictly increasing
        self.assertEqual(thresholds[-1], np.inf)

        print("\nThresholds:", thresholds)



if __name__ == "__main__":
    unittest.main()