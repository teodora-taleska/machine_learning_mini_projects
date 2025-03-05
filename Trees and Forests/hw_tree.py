import csv
import numpy as np
import random
from sklearn.metrics import accuracy_score
import time


def all_columns(X, rand=None):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    num_features = int(np.sqrt(X.shape[1]))
    return rand.sample(range(X.shape[1]), num_features)


class Tree:
    """
    Tree - a flexible classification tree with the following attributes:

    1. `rand`: a random generator for reproducibility, of type `random.Random`.
    2. `get_candidate_columns`: a function that returns a list of column indices
       considered for a split (needed for the random forests).
    3. `min_samples`: the minimum number of samples, where a node is still split further.

    Use the Gini impurity to select the best splits.
    """

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        self.rand = rand if rand is not None else random.Random()
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples
        self.root = None

    def gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    # Optimizing the split function...
    def best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in self.get_candidate_columns(X, self.rand):
            sorted_indices = np.argsort(X[:, feature])  # Sort feature values once to save redundant comparisons
            X_sorted, y_sorted = X[sorted_indices, feature], y[sorted_indices]

            for i in range(1, len(y_sorted)):  # Only check n-1 possible splits
                if y_sorted[i] == y_sorted[i - 1]:  # Skip redundant splits - duplicate values to avoid unnecessary calculations
                    continue

                threshold = (X_sorted[i] + X_sorted[i - 1]) / 2  # Midpoint threshold

                left_size, right_size = i, len(y_sorted) - i
                # utilization of numpy operations instead of looping over all unique values..
                gini_left = 1 - np.sum((np.bincount(y_sorted[:i], minlength=2) / left_size) ** 2)
                gini_right = 1 - np.sum((np.bincount(y_sorted[i:], minlength=2) / right_size) ** 2)

                weighted_gini = (left_size * gini_left + right_size * gini_right) / len(y_sorted)

                if weighted_gini < best_gini:
                    best_gini, best_feature, best_threshold = weighted_gini, feature, threshold

        return best_feature, best_threshold
    # best built time: 0.71s (optimized from 6.85s)

    def build(self, X, y):
        self.root = self._build_tree(X, y)
        return TreeModel(self.root)

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples:
            return {'prediction': np.argmax(np.bincount(y))}

        feature, threshold = self.best_split(X, y)
        if feature is None:
            return {'prediction': np.argmax(np.bincount(y))}

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

class TreeModel:
    def __init__(self, root):
        self.root = root

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if 'prediction' in node:
            return node['prediction']
        feature, threshold, left, right = node['feature'], node['threshold'], node['left'], node['right']
        if x[feature] <= threshold:
            return self._predict_single(x, left)
        else:
            return self._predict_single(x, right)

# ---------------------------------------------------------------

class RandomForest:
    """
    RandomForest, with attributes:

    1. `rand`: a random generator.
    2. `n`: number of bootstrap samples. The RandomForest should use an instance
       of `Tree` internally. Build full trees (`min_samples=2`). For each split,
       consider random (square root of the number of input variables) variables.
    """

    def __init__(self, rand=None, n=50):
        self.n = n
        self.rand = rand if rand is not None else random.Random()
        self.trees = []

    def build(self, X, y):
        self.trees = []
        for _ in range(self.n):
            bootstrap_indices = self.rand.choices(range(len(y)), k=len(y))
            X_sample, y_sample = X[bootstrap_indices], y[bootstrap_indices]
            tree = Tree(rand=self.rand, get_candidate_columns=random_sqrt_columns, min_samples=2)
            self.trees.append(tree.build(X_sample, y_sample))
        return RFModel(self.trees)


class RFModel:
    def __init__(self, trees):
        self.trees = trees

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.round(predictions.mean(axis=0)).astype(int)

    def importance(self, X, y):
        """
           Calculate permutation importance for each feature.
        """
        base_accuracy = np.mean(self.predict(X) == y)
        importances = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i]) # permute/shuffle the i-th column
            shuffled_accuracy = np.mean(self.predict(X_permuted) == y)
            importances[i] = base_accuracy - shuffled_accuracy

        return importances


def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def tki():
    legend, Xt, yt = read_tab("./tki-train.tab", {"Bcr-abl": 1, "Wild type": 0})
    _, Xv, yv = read_tab("./tki-test.tab", {"Bcr-abl": 1, "Wild type": 0})
    return (Xt, yt), (Xv, yv), legend


def compute_misclassification(y_true, y_pred):
    misclassification_rate = 1 - accuracy_score(y_true, y_pred)
    std_error = np.std(y_true != y_pred) / np.sqrt(len(y_true))
    return misclassification_rate, std_error


def hw_tree_full(train, test):
    """
    Build a decision tree with min_samples=2 and compute misclassification rates and standard errors.
    """
    X_train, y_train = train
    X_test, y_test = test

    start_time = time.time()
    tree = Tree(rand=random.Random(), min_samples=2)
    tree_model = tree.build(X_train, y_train)
    build_time = time.time() - start_time
    print(f"Tree built in {build_time:.2f} seconds")

    train_results = compute_misclassification(y_train, tree_model.predict(X_train))
    test_results = compute_misclassification(y_test, tree_model.predict(X_test))

    return train_results, test_results


def hw_randomforests(train, test):
    """
    Build a random forest with 100 trees and min_samples=2, and compute misclassification rates and standard errors.
    """
    X_train, y_train = train
    X_test, y_test = test

    start_time = time.time()
    rf = RandomForest(n=100)
    model = rf.build(X_train, y_train)
    build_time = time.time() - start_time
    print(f"RF built in {build_time:.2f} seconds")

    train_results = compute_misclassification(y_train, model.predict(X_train))
    test_results = compute_misclassification(y_test, model.predict(X_test))

    return train_results, test_results


if __name__ == "__main__":
    learn, test, legend = tki()

    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))



