import csv
import numpy as np
import random
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import combinations


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

    def best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in self.get_candidate_columns(X, self.rand):
            sorted_indices = np.argsort(X[:, feature])  # sort feature values once to save redundant comparisons
            X_sorted, y_sorted = X[sorted_indices, feature], y[sorted_indices]

            for i in range(1, len(y_sorted)):  # only check n-1 possible splits
                if y_sorted[i] == y_sorted[i - 1]:  # skip redundant splits/duplicate values to avoid unnecessary calculations
                    continue

                threshold = (X_sorted[i] + X_sorted[i - 1]) / 2  # get midpoint threshold

                left_size, right_size = i, len(y_sorted) - i
                gini_left = 1 - np.sum((np.bincount(y_sorted[:i], minlength=2) / left_size) ** 2)
                gini_right = 1 - np.sum((np.bincount(y_sorted[i:], minlength=2) / right_size) ** 2)

                weighted_gini = (left_size * gini_left + right_size * gini_right) / len(y_sorted)

                if weighted_gini < best_gini:
                    best_gini, best_feature, best_threshold = weighted_gini, feature, threshold

        return best_feature, best_threshold
    # best built time: 0.71s (optimized from 6.85s)

    def build(self, X, y):
        self.features_used.clear()  # reset for new tree
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

        self.features_used.add(feature) # adding only features considered for split

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

# ----------------------------------------------------------------------------------------------- #

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

        # precompute base accuracies for each tree using its OOB samples
        base_accuracies = np.array([
            np.mean(self.y_train[oob_idx] == tree.predict(self.X_train[oob_idx])) # accuracy score
            if len(oob_idx) > 0 else None
            for tree, oob_idx in zip(self.trees, self.oob_indices)
        ])

        for i in range(n_features):
            feature_importance = 0

            for tree, oob_idx, features, base_acc in zip(self.trees, self.oob_indices, self.features_used,
                                                         base_accuracies):
                if len(oob_idx) == 0 or i not in features or base_acc is None:
                    continue

                X_oob = self.X_train[oob_idx].copy()
                y_oob = self.y_train[oob_idx]

                np.random.shuffle(X_oob[:, i])  # permute feature i
                y_pred = tree.predict(X_oob)
                if None in y_pred or None in y_oob:
                    continue
                shuffled_accuracy = np.mean(y_oob == y_pred)

                feature_importance += base_acc - shuffled_accuracy

            importances[i] = feature_importance

        execution_time = time.time() - start_time
        print(f"⌛ Permutation importance calculation took {execution_time:.4f} seconds.")

        return importances

    def importance3(self):
        """Efficiently calculate permutation importance for combinations of 3 features using only features present in each tree."""
        start_time = time.time()
        importance3_scores = {}

        base_accuracies = np.array([
            np.mean(self.y_train[oob_idx] == tree.predict(self.X_train[oob_idx])) # accuracy score
            if len(oob_idx) > 0 else None
            for tree, oob_idx in zip(self.trees, self.oob_indices)
        ])

        valid_trees = [
            (tree, oob_idx, features, base_acc)
            for tree, oob_idx, features, base_acc in
            zip(self.trees, self.oob_indices, self.features_used, base_accuracies)
            if len(oob_idx) > 0 and base_acc is not None and len(features) >= 3
        ]

        # for progress tracking
        total_iterations = sum(len(list(combinations(features, 3))) for _, _, features, _ in valid_trees)
        progress = tqdm(total=total_iterations, desc="Calculating Feature Importance")

        for tree, oob_idx, features, base_acc in valid_trees:
            X_oob, y_oob = self.X_train[oob_idx], self.y_train[oob_idx]
            feature_combos = list(combinations(features, 3))

            # vectorized shuffling
            X_oob_shuffled = X_oob.copy()
            shuffled_accuracies = np.zeros(len(feature_combos))

            for i, combo in enumerate(feature_combos):
                X_oob_shuffled[:, combo] = X_oob[:, combo][np.random.permutation(len(X_oob))]
                shuffled_accuracies[i] = np.mean(y_oob == tree.predict(X_oob_shuffled))

            importance_diffs = base_acc - shuffled_accuracies
            for combo, imp_diff in zip(feature_combos, importance_diffs):
                importance3_scores[combo] = importance3_scores.get(combo, 0) + imp_diff

            progress.update(len(feature_combos))

        progress.close()
        execution_time = time.time() - start_time
        print(f"⌛ 3-way permutation importance calculation took {execution_time:.4f} seconds.")

        sorted_importances = sorted(importance3_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_importances

    def importance3_structure(self):
        """
        Identify the best combination of 3 variables by exploring the structure of trees in the forest.
        """
        start_time = time.time()
        combo_counts = {}

        for tree_features in self.features_used:
            if len(tree_features) < 3:
                continue

            feature_combos = list(combinations(tree_features, 3))
            for combo in feature_combos:
                if combo in combo_counts:
                    combo_counts[combo] += 1
                else:
                    combo_counts[combo] = 1

        best_combo = max(combo_counts, key=combo_counts.get)
        execution_time = time.time() - start_time
        print(f"⌛ Importance calculation based on tree structure took {execution_time:.4f} seconds.")

        return best_combo

# ----------------------------------------------------------------------------------------------- #

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
    misclassification_rate = 1 - np.mean(y_true == y_pred)
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
    print(f"⌛ Tree built in {build_time:.2f} seconds")

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
    print(f"⌛ RF with 100 estimations built in {build_time:.2f} seconds")

    train_results = compute_misclassification(y_train, model.predict(X_train))
    test_results = compute_misclassification(y_test, model.predict(X_test))

    return train_results, test_results



def plot_variable_importance(X, y, feature_names):

    print("Plot variable importance -------------------")
    print("Training random forest...")
    start_time = time.time()
    rf = RandomForest(n=100)
    model = rf.build(X, y)
    build_time = time.time() - start_time
    print(f"⌛ RF with 100 estimations built in {build_time:.2f} seconds")

    importances = model.importance()

    # train 100 non-random trees on randomized data and collect root features
    print('Collecting root features...')
    root_features = np.zeros(X.shape[1])
    for i in range(100):
        print('Tree #', i)
        rand_indices = np.random.permutation(len(y))
        X_shuffled, y_shuffled = X[rand_indices], y[rand_indices]
        tree = Tree()
        tree_model = tree.build(X_shuffled, y_shuffled)
        if tree_model.root and 'feature' in tree_model.root:
            root_features[tree_model.root['feature']] += 1

    root_features /= 100  # convert to frequency

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


def plot_misclassification_vs_trees(train, test, tree_counts=None):
    """
    Plots misclassification rates against the number of trees in the Random Forest.
    """
    if tree_counts is None:
        tree_counts = [1, 5, 10, 20, 50, 100, 150]
    train_errors = []
    test_errors = []
    test_std_errors = []

    for n in tree_counts:
        rf = RandomForest(n=n)
        model = rf.build(train[0], train[1])

        train_mis, _ = compute_misclassification(train[1], model.predict(train[0]))
        test_mis, test_std = compute_misclassification(test[1], model.predict(test[0]))

        train_errors.append(train_mis)
        test_errors.append(test_mis)
        test_std_errors.append(test_std)

    print(f"Train Errors: {train_errors}")
    print(f"Test Errors: {test_errors}")
    print(f"Test Std Errors: {test_std_errors}")

    plt.figure(figsize=(8, 5))
    plt.plot(tree_counts, test_errors, marker='o', linestyle='-', color='b', label="Test Misclassification")
    plt.fill_between(tree_counts,
                     np.array(test_errors) - np.array(test_std_errors),
                     np.array(test_errors) + np.array(test_std_errors),
                     color='b', alpha=0.2, label="Test Error ± Std Err")

    plt.xlabel("Number of Trees")
    plt.ylabel("Misclassification Rate")
    plt.title("Misclassification Rate vs. Number of Trees")
    plt.legend()

    save_path = "visualizations/misclassification_vs_trees.png"
    plt.savefig(save_path)
    plt.show()

    print(f"Plot saved to {save_path}")


def train_and_compare(learn, test):
    """
    Train a RandomForest with 1000 trees, extract feature importance,
    and compare the performance of single Tree models with the top features.
    """

    X_train, y_train = learn
    X_test, y_test = test

    print("Training RandomForest...")
    rand_forest = RandomForest(n=1000)
    rf_model = rand_forest.build(X_train, y_train)

    print("Computing feature importance...")
    importance = rf_model.importance()
    importance3 = rf_model.importance3()
    importance3_structure = rf_model.importance3_structure()

    top3_importance = np.argsort(importance)[-3:]
    a, b, c = importance3[0][0]
    top1_importance3 = [a, b, c]
    top1_importance3_structure = list(importance3_structure)

    print("Top 3 features (single importance):", top3_importance)
    print("Top feature combination (importance3):", top1_importance3)
    print("Top feature combination (importance3_structure):", top1_importance3_structure)

    combs = [top3_importance, top1_importance3, top1_importance3_structure]
    results = {}

    for c in combs:
        print(f"Training Tree with features {c}...")
        X_train_feature = X_train[:, c]
        X_test_feature = X_test[:, c]

        tree = Tree()
        tree_model = tree.build(X_train_feature, y_train)

        y_pred = tree_model.predict(X_test_feature)
        accuracy = np.mean(y_pred == y_test)

        results[tuple(c)] = accuracy
        print(f"Accuracy with feature {c}: {accuracy:.4f}")

    return results


if __name__ == "__main__":
    learn, test, legend = tki()

    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))

    # print('variable importance', plot_variable_importance(learn[0], learn[1], legend))
    # print('misclassification vs trees', plot_misclassification_vs_trees(learn, test))

    print('train and compare', train_and_compare(learn, test))


