import numpy as np
import csv
from abc import ABC, abstractmethod
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

# TODO: Report

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(a):
    return (a > 0).astype(float)

def apply_activation(z, func_name):
    if func_name == 'sigmoid':
        return sigmoid(z)
    elif func_name == 'relu':
        return relu(z)
    else:
        raise ValueError(f"Unknown activation function '{func_name}'")

def activation_derivative(a, func_name):
    if func_name == 'sigmoid':
        return sigmoid_derivative(a)
    elif func_name == 'relu':
        return relu_derivative(a)
    else:
        raise ValueError(f"Unknown activation function '{func_name}'")


class ANNBase(ABC):
    def __init__(self, units=[], lambda_=0., activations=None, initialization='he'):
        """
        Base class for ANN implementations.

        Args:
            units: List of integers specifying number of units in each hidden layer
            lambda_: Regularization strength
            activations: List of activation functions for each layer
            initialization: Weight initialization method ('he' or 'small')
        """
        self.units = units
        self.lambda_ = lambda_
        self.activations = activations if activations is not None else ['sigmoid'] * len(units)
        self.initialization = initialization
        self.weights_ = None
        self.n_outputs_ = None

    def _initialize_weights_small(self, n_features, n_outputs):
        """Initialize weights with small random values."""
        layer_sizes = [n_features] + self.units + [n_outputs]
        weights = []

        for i in range(len(layer_sizes) - 1):
            # Add 1 for bias term
            w = np.random.randn(layer_sizes[i] + 1, layer_sizes[i + 1]) * 0.01
            weights.append(w)

        return weights

    def _initialize_weights_he(self, n_features, n_outputs):
        """Initialize weights using He initialization."""
        layer_sizes = [n_features] + self.units + [n_outputs]
        weights = []

        for i in range(len(layer_sizes) - 1):
            # He initialization: scale by sqrt(2/n_input)
            std_dev = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i] + 1, layer_sizes[i + 1]) * std_dev
            weights.append(w)

        return weights

    def _initialize_weights(self, n_features, n_outputs):
        """Initialize weights based on selected method."""
        if self.initialization == 'he':
            return self._initialize_weights_he(n_features, n_outputs)
        return self._initialize_weights_small(n_features, n_outputs)

    def _add_bias(self, X):
        """Add bias term to input matrix."""
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _forward_pass(self, X):
        activations = [self._add_bias(X)]
        zs = []

        for i, w in enumerate(self.weights_[:-1]):
            z = activations[-1] @ w
            zs.append(z)
            a = apply_activation(z, self.activations[i])
            activations.append(self._add_bias(a))  # Add bias for next layer

        z = activations[-1] @ self.weights_[-1]
        zs.append(z)

        return activations, zs

    def _backpropagate_helper(self, delta, l, activations):
        """Generic backpropagation step for hidden layers."""
        a = activations[l + 1][:, 1:]  # remove bias from activation
        d_act = activation_derivative(a, self.activations[l])
        return (delta @ self.weights_[l + 1][1:].T) * d_act

    @abstractmethod
    def _backward_pass(self, y, activations, zs):
        pass

    @abstractmethod
    def fit(self, X, y, learning_rate=0.1, epochs=100000, verbose=False):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def _compute_cost(self, y, output_activation, activations):
        pass

    def weights(self):
        """Return the learned weights."""
        return self.weights_

    def gradient_check(self, X, y, epsilon=1e-5):
        """Verify gradients using numerical approximation."""
        original_weights = [w.copy() for w in self.weights_]
        flat_weights = np.concatenate([w.flatten() for w in original_weights])

        # compute analytical gradient
        activations, zs = self._forward_pass(X)
        grads = self._backward_pass(y, activations, zs)
        flat_grad = np.concatenate([g.flatten() for g in grads])

        # compute numerical gradient
        num_grad = np.zeros_like(flat_weights)

        for i in range(len(flat_weights)):
            # Perturb parameter
            theta_plus = [w.copy() for w in original_weights]
            theta_minus = [w.copy() for w in original_weights]

            pos = i
            for mat_idx in range(len(theta_plus)):
                size = theta_plus[mat_idx].size
                if pos < size:
                    # Unflatten the index
                    idx = np.unravel_index(pos, theta_plus[mat_idx].shape)
                    theta_plus[mat_idx][idx] += epsilon
                    theta_minus[mat_idx][idx] -= epsilon
                    break
                pos -= size

            # Compute cost at both points
            self.weights_ = theta_plus
            _, zs_plus = self._forward_pass(X)
            if isinstance(self, ANNClassification):
                output_plus = self._softmax(zs_plus[-1])
            else:
                output_plus = zs_plus[-1].flatten()
            cost_plus = self._compute_cost(y, output_plus, _)

            self.weights_ = theta_minus
            _, zs_minus = self._forward_pass(X)
            if isinstance(self, ANNClassification):
                output_minus = self._softmax(zs_minus[-1])
            else:
                output_minus = zs_minus[-1].flatten()
            cost_minus = self._compute_cost(y, output_minus, _)

            num_grad[i] = (cost_plus - cost_minus) / (2 * epsilon)

            # Restore original weights
            self.weights_ = [w.copy() for w in original_weights]

        numerator = np.linalg.norm(flat_grad - num_grad)
        denominator = np.linalg.norm(flat_grad) + np.linalg.norm(num_grad)
        difference = numerator / denominator
        print(f"{difference:.10f}")

        return difference < 1e-7, difference


class ANNClassification(ANNBase):
    """ANN for classification tasks."""

    def _softmax(self, z):
        """Softmax activation function for output layer."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # max val in each row of z for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True) # normalize to sum to 1

    def _compute_cost(self, y, output_activation, activations):
        """Compute the cost function for classification."""
        m = y.shape[0]
        y_one_hot = np.eye(self.n_outputs_)[y]

        cost = -np.sum(y_one_hot * np.log(output_activation + 1e-15)) / m # cross-entropy loss

        # l2 regularization
        reg_cost = 0
        for w in self.weights_:
            reg_cost += np.sum(w[1:] ** 2)  # exclude bias weights
        reg_cost = (self.lambda_ / (2 * m)) * reg_cost

        return cost + reg_cost

    def _backward_pass(self, y, activations, zs):
        """Backpropagation for classification."""
        m = y.shape[0]
        y_one_hot = np.eye(self.n_outputs_)[y]
        grads = [np.zeros_like(w) for w in self.weights_] # empty list to store the gradients for each layer’s weight matrix

        output_activation = self._softmax(zs[-1])  # output layer error
        delta = (output_activation - y_one_hot) / m
        grads[-1] = activations[-1].T @ delta

        for l in range(len(self.weights_) - 2, -1, -1):
            delta = self._backpropagate_helper(delta, l, activations)
            grads[l] = activations[l].T @ delta

        # L2 regularization gradient
        for i in range(len(self.weights_)):
            grads[i][1:] += (self.lambda_ / m) * self.weights_[i][1:]

        return grads

    def fit(self, X, y, learning_rate=0.1, epochs=100000, verbose=False):
        """Train the classification network."""
        self.n_outputs_ = len(np.unique(y))

        self.weights_ = self._initialize_weights(X.shape[1], self.n_outputs_)

        for epoch in range(epochs):

            # Forward pass
            activations, zs = self._forward_pass(X)
            output_activation = self._softmax(zs[-1])

            # Compute cost
            cost = self._compute_cost(y, output_activation, activations)

            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.6f}")

            # Backward pass
            grads = self._backward_pass(y, activations, zs)

            # Update weights
            for i in range(len(self.weights_)):
                self.weights_[i] -= learning_rate * grads[i]

        return self

    def predict(self, X):
        """Predict class probabilities."""
        _, zs = self._forward_pass(X)
        return self._softmax(zs[-1])


class ANNRegression(ANNBase):
    """ANN for regression tasks."""

    def _compute_cost(self, y, output_activation, activations):
        """Compute the cost function for regression."""
        m = y.shape[0]

        # MSE
        cost = np.sum((output_activation.flatten() - y) ** 2) / (2 * m)

        # l2 reg
        reg_cost = 0
        for w in self.weights_:
            reg_cost += np.sum(w[1:] ** 2)  # exclude bias weights
        reg_cost = (self.lambda_ / (2 * m)) * reg_cost

        return cost + reg_cost

    def _backward_pass(self, y, activations, zs):
        m = y.shape[0]
        grads = [np.zeros_like(w) for w in self.weights_]

        output_activation = zs[-1].flatten()
        delta = (output_activation - y).reshape(-1, 1) / m
        grads[-1] = activations[-1].T @ delta

        for l in range(len(self.weights_) - 2, -1, -1):
            delta = self._backpropagate_helper(delta, l, activations)
            grads[l] = activations[l].T @ delta

        # add regularization
        for i in range(len(self.weights_)):
            grads[i][1:] += (self.lambda_ / m) * self.weights_[i][1:]

        return grads

    def fit(self, X, y, learning_rate=0.1, epochs=100000, verbose=False):
        """Train the regression network."""
        # For regression, we have 1 output
        self.n_outputs_ = 1

        self.weights_ = self._initialize_weights(X.shape[1], self.n_outputs_)

        for epoch in range(epochs):
            # Forward pass
            activations, zs = self._forward_pass(X)
            output_activation = zs[-1].flatten()

            # Compute cost
            cost = self._compute_cost(y, output_activation, activations)

            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.6f}")

            # Backward pass
            grads = self._backward_pass(y, activations, zs)

            # Update weights
            for i in range(len(self.weights_)):
                self.weights_[i] -= learning_rate * grads[i]

        return self

    def predict(self, X):
        """Predict continuous values."""
        _, zs = self._forward_pass(X)
        return zs[-1].flatten()

# Data loading
def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))
    legend = content[0][1:]
    data = content[1:]
    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])
    return legend, X, y

def doughnut():
    legend, X, y = read_tab("doughnut.tab", {"C1": 0, "C2": 1})
    return X, y

def squares():
    legend, X, y = read_tab("squares.tab", {"C1": 0, "C2": 1})
    return X, y


def fit_data(X, y, model, test_size=0.3, random_state=42, cost_fn=None):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cost = cost_fn(y_test, y_pred) if cost_fn else None
    return y_pred, y_test, cost


def classification_comparison_test(data):
    print("Comparing custom ANN classification with sklearn MLPClassifier...")
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf_custom = ANNClassification(units=[10, 10], lambda_=0.01, activations=["relu", "relu"])
    model_custom = clf_custom.fit(X_train, y_train)
    probs_custom = model_custom.predict(X_test)
    preds_custom = np.argmax(probs_custom, axis=1)

    clf_sklearn = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu',
                                 alpha=0.01, max_iter=3000, random_state=42)
    clf_sklearn.fit(X_train, y_train)
    preds_sklearn = clf_sklearn.predict(X_test)

    acc_custom = accuracy_score(y_test, preds_custom)
    acc_sklearn = accuracy_score(y_test, preds_sklearn)
    agreement = np.mean(preds_custom == preds_sklearn) * 100 # percentage of agreement

    print(f"Custom ANN Accuracy: {acc_custom:.4f}")
    print(f"Sklearn MLP Accuracy: {acc_sklearn:.4f}")
    print(f"Prediction Agreement: {agreement:.2f}%")


    if agreement > 90:
        print("✅ PASS: Predictions are sufficiently aligned.")
    else:
        print("❌ FAIL: Predictions diverge significantly.")



def regression_comparison_test(data):
    print("Comparing custom ANN regression with sklearn MLPRegressor...")
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    my_ann = ANNRegression(units=[10, 10], lambda_=0.01, activations=["relu", "relu"])
    my_ann.fit(X_train, y_train)
    y_pred_custom = my_ann.predict(X_test)

    sk_ann = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', alpha=0.01,
                          max_iter=3000, random_state=42)
    sk_ann.fit(X_train, y_train)
    y_pred_sklearn = sk_ann.predict(X_test)

    mse_custom = mean_squared_error(y_test, y_pred_custom)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

    print("Custom ANN MSE:", mse_custom)
    print("Sklearn ANN MSE:", mse_sklearn)

    abs_diff = np.abs(y_pred_custom - y_pred_sklearn)
    mae = np.mean(abs_diff)
    max_diff = np.max(abs_diff)
    within_threshold = np.mean(abs_diff < 1.0) * 100 # 1.0 unit tolerance

    print("Mean Absolute Difference:", mae)
    print("Max Absolute Difference:", max_diff)
    print(f"Predictions within 1.0 of each other: {within_threshold:.2f}%")

    # Pass/Fail Criteria
    mse_close = np.abs(mse_custom - mse_sklearn) < 5
    similarity_ok = within_threshold > 90
    print("MSE close:", mse_close)
    print("Similarity OK:", similarity_ok)

    print("\nTEST RESULT:")
    if mse_close and similarity_ok:
        print("✅ PASS: Custom ANN behaves similarly to sklearn MLPRegressor.")
    else:
        print("❌ FAIL: Custom ANN differs too much from sklearn.")


# Main block
if __name__ == "__main__":
    fitter = ANNClassification(units=[3, 4], lambda_=0)
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    y = np.array([0, 1, 2])
    model = fitter.fit(X, y)
    predictions = model.predict(X)
    np.testing.assert_almost_equal(predictions,
                                   [[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]], decimal=3)

    success, diff = model.gradient_check(X, y)
    print(f"Gradient check passed: {success}, Difference: {diff}")

    X_squares, y_squares = squares()
    X_doughnut, y_doughnut = doughnut()

    # fit the model on squares.tab
    reg_model = ANNRegression(units=[10, 10], lambda_=0.01, activations=["relu", "relu"])
    y_pred_squares, y_test_squares, cost_sq = fit_data(X_squares, y_squares, reg_model, test_size=0.3, random_state=42, cost_fn=mean_squared_error)
    print("Training on squares.tab (MSE):", cost_sq)

    # fit the model on doughnut.tab
    clas_model = ANNClassification(units=[10, 10], lambda_=0.01, activations=["relu", "relu"])
    y_pred_doughnut, y_test_doughnut, cost_dn = fit_data(X_doughnut, y_doughnut, clas_model, test_size=0.3, random_state=42, cost_fn=accuracy_score)
    print("Training on doughnut.tab (accuracy):", cost_dn)

    # Compare with sklearn
    classification_comparison_test(doughnut())
    regression_comparison_test(doughnut())


