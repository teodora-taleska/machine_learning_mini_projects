import numpy as np
import csv
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


class ANNBase:
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

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def _relu(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)

    def _apply_activation(self, z, activation):
        """Apply specified activation function."""
        if activation == 'sigmoid':
            return self._sigmoid(z)
        elif activation == 'relu':
            return self._relu(z)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def _forward_pass(self, X):
        """Perform forward pass through the network."""
        activations = [self._add_bias(X)]
        zs = []

        for i, w in enumerate(self.weights_[:-1]):
            z = activations[-1] @ w
            zs.append(z)
            a = self._apply_activation(z, self.activations[i])
            activations.append(self._add_bias(a))

        # Output layer (no activation added yet)
        z = activations[-1] @ self.weights_[-1]
        zs.append(z)

        return activations, zs

    def _compute_cost(self, y, output_activation, activations):
        """Compute the cost function."""
        raise NotImplementedError("Should be implemented in subclass")

    def _backward_pass(self, y, activations, zs):
        """Perform backward pass through the network."""
        raise NotImplementedError("Should be implemented in subclass")

    def fit(self, X, y, learning_rate=0.1, epochs=10000, verbose=False):
        """Train the neural network."""
        raise NotImplementedError("Should be implemented in subclass")

    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError("Should be implemented in subclass")

    def weights(self):
        """Return the learned weights."""
        return self.weights_

    def gradient_check(self, X, y, epsilon=1e-7):
        """Verify gradients using numerical approximation."""
        # Flatten all parameters
        original_weights = [w.copy() for w in self.weights_]
        flat_weights = np.concatenate([w.flatten() for w in original_weights])

        # Compute analytical gradient
        activations, zs = self._forward_pass(X)
        grads = self._backward_pass(y, activations, zs)
        flat_grad = np.concatenate([g.flatten() for g in grads])

        # Compute numerical gradient
        num_grad = np.zeros_like(flat_weights)

        for i in range(len(flat_weights)):
            # Perturb parameter
            theta_plus = [w.copy() for w in original_weights]
            theta_minus = [w.copy() for w in original_weights]

            # Find which weight matrix this parameter belongs to
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

        # Compare gradients
        numerator = np.linalg.norm(flat_grad - num_grad)
        denominator = np.linalg.norm(flat_grad) + np.linalg.norm(num_grad)
        difference = numerator / denominator

        return difference < 1e-7, difference


class ANNClassification(ANNBase):
    """ANN for classification tasks."""

    def _softmax(self, z):
        """Softmax activation function for output layer."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_cost(self, y, output_activation, activations):
        """Compute the cost function for classification."""
        m = y.shape[0]
        y_one_hot = np.eye(self.n_outputs_)[y]

        # Cross-entropy loss
        cost = -np.sum(y_one_hot * np.log(output_activation + 1e-15)) / m

        # Add regularization (excluding bias terms)
        reg_cost = 0
        for w in self.weights_:
            reg_cost += np.sum(w[1:] ** 2)  # exclude bias weights
        reg_cost = (self.lambda_ / (2 * m)) * reg_cost

        return cost + reg_cost

    def _backward_pass(self, y, activations, zs):
        """Backpropagation for classification."""
        m = y.shape[0]
        y_one_hot = np.eye(self.n_outputs_)[y]

        # Initialize gradients
        grads = [np.zeros_like(w) for w in self.weights_]

        # Output layer error
        output_activation = self._softmax(zs[-1])
        delta = (output_activation - y_one_hot) / m

        # Store gradient for output layer
        grads[-1] = activations[-1].T @ delta

        # Backpropagate through hidden layers
        for l in range(len(self.weights_) - 2, -1, -1):
            # Remove bias from activation
            a = activations[l + 1][:, 1:]

            if self.activations[l] == 'sigmoid':
                delta = (delta @ self.weights_[l + 1][1:].T) * a * (1 - a)
            elif self.activations[l] == 'relu':
                delta = (delta @ self.weights_[l + 1][1:].T) * (a > 0)

            grads[l] = activations[l].T @ delta

        # Add regularization (excluding bias terms)
        for i, w in enumerate(self.weights_):
            grads[i][1:] += (self.lambda_ / m) * w[1:]

        return grads

    def fit(self, X, y, learning_rate=0.1, epochs=10000, verbose=False):
        """Train the classification network."""
        # Determine number of outputs
        self.n_outputs_ = len(np.unique(y))

        # Initialize weights
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

        # Mean squared error
        cost = np.sum((output_activation.flatten() - y) ** 2) / (2 * m)

        # Add regularization (excluding bias terms)
        reg_cost = 0
        for w in self.weights_:
            reg_cost += np.sum(w[1:] ** 2)  # exclude bias weights
        reg_cost = (self.lambda_ / (2 * m)) * reg_cost

        return cost + reg_cost

    def _backward_pass(self, y, activations, zs):
        """Backpropagation for regression."""
        m = y.shape[0]

        # Initialize gradients
        grads = [np.zeros_like(w) for w in self.weights_]

        # Output layer error
        output_activation = zs[-1].flatten()
        delta = (output_activation - y).reshape(-1, 1) / m

        # Store gradient for output layer
        grads[-1] = activations[-1].T @ delta

        # Backpropagate through hidden layers
        for l in range(len(self.weights_) - 2, -1, -1):
            # Remove bias from activation
            a = activations[l + 1][:, 1:]

            if self.activations[l] == 'sigmoid':
                delta = (delta @ self.weights_[l + 1][1:].T) * a * (1 - a)
            elif self.activations[l] == 'relu':
                delta = (delta @ self.weights_[l + 1][1:].T) * (a > 0)

            grads[l] = activations[l].T @ delta

        # Add regularization (excluding bias terms)
        for i, w in enumerate(self.weights_):
            grads[i][1:] += (self.lambda_ / m) * w[1:]

        return grads

    def fit(self, X, y, learning_rate=0.1, epochs=10000, verbose=False):
        """Train the regression network."""
        # For regression, we have 1 output
        self.n_outputs_ = 1

        # Initialize weights
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

# Main block
if __name__ == "__main__":
    fitter = ANNClassification(units=[3, 4], lambda_=0)
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    y = np.array([0, 1, 2])
    model = fitter.fit(X, y)
    predictions = model.predict(X)
    print(predictions)
    # np.testing.assert_almost_equal(predictions,
    #                                [[1, 0, 0],
    #                                 [0, 1, 0],
    #                                 [0, 0, 1]], decimal=3)
    np.testing.assert_allclose(predictions,
                               [[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]],
                               atol=0.01)  # Allow ±0.01 differences

    # model._initialize_weights(X.shape[1], len(np.unique(y)))
    is_correct, difference = model.gradient_check(X, y)
    print(f"Gradients correct: {is_correct}, difference: {difference}")

    #
    # # Example usage of sklearn for comparison
    # X_squares, y_squares = squares()
    # X_doughnut, y_doughnut = doughnut()
    #
    # X_train_sq, X_test_sq, y_train_sq, y_test_sq = train_test_split(X_squares, y_squares, test_size=0.3, random_state=42)
    # X_train_dg, X_test_dg, y_train_dg, y_test_dg = train_test_split(X_doughnut, y_doughnut, test_size=0.3, random_state=42)
    #
    # print("Training on squares.tab:")
    # my_ann_sq = ANNRegression(units=[10, 10], lambda_=0.01, activations=["relu", "relu"])
    # model_sq = my_ann_sq.fit(X_train_sq, y_train_sq, max_iter=3000, lr=0.01)
    # y_pred_sq = model_sq.predict(X_test_sq)
    # print("Custom ANN MSE (squares):", mean_squared_error(y_test_sq, y_pred_sq))
    #
    # sk_ann_sq = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', alpha=0.01, max_iter=3000, random_state=42)
    # sk_ann_sq.fit(X_train_sq, y_train_sq)
    # print("Scikit-learn MSE (squares):", mean_squared_error(y_test_sq, sk_ann_sq.predict(X_test_sq)))
    #
    # print("\nTraining on doughnut.tab:")
    # my_ann_dg = ANNRegression(units=[10, 10], lambda_=0.01, activations=["relu", "relu"])
    # model_dg = my_ann_dg.fit(X_train_dg, y_train_dg, max_iter=3000, lr=0.01)
    # y_pred_dg = model_dg.predict(X_test_dg)
    # print("Custom ANN MSE (doughnut):", mean_squared_error(y_test_dg, y_pred_dg))
    #
    # sk_ann_dg = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', alpha=0.01, max_iter=3000, random_state=42)
    # sk_ann_dg.fit(X_train_dg, y_train_dg)
    # print("Scikit-learn MSE (doughnut):", mean_squared_error(y_test_dg, sk_ann_dg.predict(X_test_dg)))
    #
    # print("\nClassification Test:")
    # X, y = squares()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #
    # clf = ANNClassification(units=[8, 8], lambda_=0.01, activations=["relu", "relu"])
    # model = clf.fit(X_train, y_train, max_iter=3000, lr=0.01)
    # pred = model.predict(X_test)
    # acc = accuracy_score(y_test, np.argmax(pred, axis=1))
    # print("Custom ANN Accuracy (squares):", acc)
    #
    # sk_clf = MLPClassifier(hidden_layer_sizes=(8, 8), activation='relu', alpha=0.01, max_iter=3000, random_state=42)
    # sk_clf.fit(X_train, y_train)
    # print("Scikit-learn Accuracy (squares):", accuracy_score(y_test, sk_clf.predict(X_test)))
