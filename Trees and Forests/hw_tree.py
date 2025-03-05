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

        for feature in self.get_candidate_columns(X):
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

    def build(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples:
            return {'prediction': np.argmax(np.bincount(y))}

        feature, threshold = self.best_split(X, y)
        if feature is None:
            return {'prediction': np.argmax(np.bincount(y))}

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        left_subtree = self.build(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.build(X[right_indices], y[right_indices], depth + 1)

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
        self.rand = rand
        self.rftree = Tree(...)  # initialize the tree properly

    def build(self, X, y):
        # ...
        return RFModel(...)  # return an object that can do prediction


class RFModel:

    def __init__(self, *args, **kwargs):
        # ...
        pass

    def predict(self, X):
        # ...
        predictions = np.ones(len(X))  # dummy output
        return predictions

    def importance(self):
        imps = np.zeros(self.X.shape[1])
        # ...
        return imps


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


def hw_tree_full(train, test):
    """
    In function hw_tree_full, build a tree with min_samples=2.
    Return misclassification rates and standard errors when using training and
    testing data as test sets.
    """
    X_train, y_train = train
    X_test, y_test = test

    start_time = time.time()

    # Build the tree with min_samples=2
    tree = Tree(rand=random.Random(), min_samples=2)
    tree_model = TreeModel(tree.build(X_train, y_train))

    end_time = time.time()
    build_time = end_time - start_time

    print(f"Tree built in {build_time:.2f} seconds")

    # Predict on training and testing data
    y_train_pred = tree_model.predict(X_train)
    y_test_pred = tree_model.predict(X_test)

    # Calculate misclassification rates
    train_misclassification_rate = 1 - accuracy_score(y_train, y_train_pred)
    test_misclassification_rate = 1 - accuracy_score(y_test, y_test_pred)

    # Calculate standard errors
    train_misclassification_error = y_train != y_train_pred
    test_misclassification_error = y_test != y_test_pred
    train_std_error = np.std(train_misclassification_error) / np.sqrt(len(y_train))
    test_std_error = np.std(test_misclassification_error) / np.sqrt(len(y_test))

    return (train_misclassification_rate, train_std_error), (test_misclassification_rate, test_std_error)


def hw_randomforests(*args, **kwargs):
    """
    In function hw_randomforest, use random forests with n=100 trees with min_samples=2.
    Return misclassification rates and standard errors when using training and
    testing data as test sets.
    """
    pass

if __name__ == "__main__":
    learn, test, legend = tki()

    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))



