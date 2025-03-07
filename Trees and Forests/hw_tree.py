import csv
import numpy as np
import random
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def all_columns(X, rand=None):
    columns = list(range(X.shape[1]))
    if rand is not None:
        np.random.shuffle(columns)
    return columns


def random_sqrt_columns(X, rand):
    num_features = int(np.sqrt(X.shape[1]))
    selected_columns = rand.sample(range(X.shape[1]), num_features)
    rand.shuffle(selected_columns)
    return selected_columns


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
        self.features_used = set()

    def gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    # Optimizing the split function...
    def best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in self.get_candidate_columns(X, self.rand):
            self.features_used.add(feature)
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
        return TreeModel(self.root, self.features_used)

    def _build_tree(self, X, y, depth=0):
        if len(y) == 0:
            print(f"Empty y at depth {depth}!")
            return {'prediction': None}
        if len(set(y)) == 1 or len(y) < self.min_samples:
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
    def __init__(self, root, features_used):
        self.root = root
        self.features_used = features_used

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
        self.oob_indices = []
        self.features_used = []

    def build(self, X, y):
        self.trees = []
        self.oob_indices = []
        self.features_used = []
        for _ in range(self.n):
            bootstrap_indices = self.rand.choices(range(len(y)), k=len(y))
            oob_indices = list(set(range(len(y))) - set(bootstrap_indices))
            self.oob_indices.append(oob_indices)
            X_sample, y_sample = X[bootstrap_indices], y[bootstrap_indices]
            tree = Tree(rand=self.rand, get_candidate_columns=random_sqrt_columns, min_samples=2)
            self.trees.append(tree.build(X_sample, y_sample))
            self.features_used.append(tree.features_used)
        print(f"Random forest built with {self.n} trees.")
        return RFModel(self.trees, X, y, self.oob_indices, self.features_used)


class RFModel:
    def __init__(self, trees, X_train, y_train, oob_indices, features_used=None):
        self.trees = trees
        self.X_train = X_train
        self.y_train = y_train
        self.oob_indices = oob_indices
        self.features_used = features_used

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.round(predictions.mean(axis=0)).astype(int)

    def importance(self):
        """
        Calculate permutation importance for each feature using OOB samples.
        """
        start_time = time.time()
        n_features = self.X_train.shape[1]
        importances = np.zeros(n_features)

        # Precompute base accuracies for each tree using its OOB samples
        base_accuracies = []
        for tree, oob_idx in zip(self.trees, self.oob_indices):
            if len(oob_idx) == 0:
                base_accuracies.append(None)
            else:
                X_oob = self.X_train[oob_idx]
                y_oob = self.y_train[oob_idx]
                base_accuracies.append(accuracy_score(y_oob, tree.predict(X_oob)))

        # Compute permutation importance
        for i in range(n_features):
            feature_importance = 0

            for tree, oob_idx, features, base_acc in zip(self.trees, self.oob_indices, self.features_used,
                                                         base_accuracies):
                if len(oob_idx) == 0 or i not in features or base_acc is None:
                    continue

                X_oob = self.X_train[oob_idx].copy()
                y_oob = self.y_train[oob_idx]

                np.random.shuffle(X_oob[:, i])  # Permute feature i
                shuffled_accuracy = accuracy_score(y_oob, tree.predict(X_oob))

                feature_importance += base_acc - shuffled_accuracy

            importances[i] = feature_importance

        execution_time = time.time() - start_time
        print(f"Optimized permutation importance calculation took {execution_time:.4f} seconds.")

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

    # importances = model.importance()
    #
    # plt.figure(figsize=(10, 6))
    # indices = np.arange(X_train.shape[1])
    # plt.bar(indices, importances, width=0.4, label='Feature Importance')
    # plt.xlabel('Feature Index')
    # plt.ylabel('Importance')
    # plt.title('Feature Importance in Random Forest')
    # plt.legend()
    # plt.show()

    return train_results, test_results



def plot_variable_importance(X, y, feature_names):

    # Train the Random Forest Model
    rf = RandomForest(n=100)
    model = rf.build(X, y)
    importances = model.importance()

    # Train 100 Non-Random Trees and Collect Root Features
    root_features = np.zeros(X.shape[1])
    for i in range(100):
        print('tree', i)
        rand_indices = np.random.permutation(len(y))
        X_shuffled, y_shuffled = X[rand_indices], y[rand_indices]
        tree = Tree()
        tree_model = tree.build(X_shuffled, y_shuffled)
        if tree_model.root and 'feature' in tree_model.root:
            root_features[tree_model.root['feature']] += 1

    # Normalize Root Feature Counts
    root_features /= 100  # Convert to frequency

    plt.figure(figsize=(12, 6))
    indices = np.arange(X.shape[1])

    plt.bar(indices - 0.2, importances, width=0.4, label='Random Forest Importance')
    plt.bar(indices + 0.2, root_features, width=0.4, label='Root Features (100 Trees)')

    plt.xlabel('Feature Index')
    plt.ylabel('Importance / Frequency')
    plt.title('Feature Importance in Random Forest & Root Features in Trees')

    plt.xticks(indices, feature_names, rotation=45, ha='right')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))

    plt.legend()
    plt.tight_layout()

    plt.savefig('visualizations/variable_importance.png')

    plt.show()




if __name__ == "__main__":
    learn, test, legend = tki()

    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))

    print('variable importance', plot_variable_importance(learn[0], learn[1], legend))



