import csv
import numpy as np
import random


def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    # select random columns of size sqrt(X.shape[1]) with replacement
    c = rand.choice(range(X.shape[1]), size=int(np.sqrt(X.shape[1])), replace=True)
    return c


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
        self.rand = rand or random.Random(0)
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples
        self.root = None

    def build(self, X, y):
        # returns the model as an object (that can do prediction)
        self.root = self._build_tree(X, y)
        return TreeModel(self.root)

    def _build_tree(self, X, y):
        if len(y) < self.min_samples or len(set(y)) == 1:
            return np.bincount(y).argmax()

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return np.bincount(y).argmax()

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices

        left_subtree = self._build_tree(X[left_indices], y[left_indices])
        right_subtree = self._build_tree(X[right_indices], y[right_indices])

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        best_feature, best_threshold, best_gini = None, None, float('inf')
        for feature in self.get_candidate_columns(X, self.rand):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gini = self._gini_impurity(y[left_mask], y[right_mask])
                if gini < best_gini:
                    best_gini, best_feature, best_threshold = gini, feature, threshold
        return best_feature, best_threshold

    def _gini_impurity(self, left_y, right_y):
        def gini(y):
            p = np.bincount(y, minlength=2) / len(y)
            return 1.0 - np.sum(p ** 2)

        left_size, right_size = len(left_y), len(right_y)
        return (left_size * gini(left_y) + right_size * gini(right_y)) / (left_size + right_size)

class TreeModel:
    def __init__(self, root):
        self.root = root

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        return self._predict_single(x, left if x[feature] <= threshold else right)

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
    legend, Xt, yt = read_tab("tki-train.tab", {"Bcr-abl": 1, "Wild type": 0})
    _, Xv, yv = read_tab("tki-test.tab", {"Bcr-abl": 1, "Wild type": 0})
    return (Xt, yt), (Xv, yv), legend


def hw_tree_full(*args, **kwargs):
    """
    In function hw_tree_full, build a tree with min_samples=2.
    Return misclassification rates and standard errors when using training and
    testing data as test sets.
    """
    pass


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



