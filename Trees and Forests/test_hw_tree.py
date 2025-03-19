import random
import unittest

import numpy as np
import time
from hw_tree import Tree, RandomForest, hw_tree_full, hw_randomforests, all_columns
from scipy.stats import norm


def random_feature(X, rand):
    return [rand.choice(list(range(X.shape[1])))]


class HWTreeTests(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])
        self.y = np.array([0, 0, 1, 1])
        self.train = self.X[:3], self.y[:3]
        self.test = self.X[3:], self.y[3:]

    def test_call_tree(self):
        t = Tree(rand=random.Random(1),
                 get_candidate_columns=random_feature,
                 min_samples=2)
        p = t.build(self.X, self.y)
        pred = p.predict(self.X)
        np.testing.assert_equal(pred, self.y)

    def test_call_randomforest(self):
        rf = RandomForest(rand=random.Random(0),
                          n=20)
        p = rf.build(self.X, self.y)
        pred = p.predict(self.X)
        np.testing.assert_equal(pred, self.y)

    def test_call_importance(self):
        rf = RandomForest(rand=random.Random(0),
                          n=20)
        p = rf.build(np.tile(self.X, (2, 1)),
                     np.tile(self.y, 2))
        imp = p.importance()
        self.assertTrue(len(imp), self.X.shape[1])
        self.assertGreater(imp[0], imp[1])

    def test_signature_hw_tree_full(self):
        (train, train_un), (test, test_un) = hw_tree_full(self.train, self.test)
        self.assertIsInstance(train, float)
        self.assertIsInstance(test, float)
        self.assertIsInstance(train_un, float)
        self.assertIsInstance(test_un, float)

    def test_signature_hw_randomforests(self):
        (train, train_un), (test, test_un) = hw_randomforests(self.train, self.test)
        self.assertIsInstance(train, float)
        self.assertIsInstance(test, float)
        self.assertIsInstance(train_un, float)
        self.assertIsInstance(test_un, float)




class MyTests(unittest.TestCase):

    def setUp(self):
        # Simple dataset for testing
        self.X_train = np.array([[2, 3], [4, 1], [5, 2], [6, 5], [8, 7]])
        self.y_train = np.array([0, 1, 1, 1, 0])
        self.X_test = np.array([[3, 2], [7, 6]])
        self.y_test = np.array([0, 0])

        # Initialize Tree
        self.tree = Tree(rand=random.Random(42), min_samples=2)
        self.tree_model = self.tree.build(self.X_train, self.y_train)

        # Initialize RandomForest
        self.forest = RandomForest(rand=random.Random(42), n=10)
        self.rf_model = self.forest.build(self.X_train, self.y_train)

    def test_feature_importance_runtime(self):
        start_time = time.time()
        importances = self.rf_model.importance()
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Feature importance function took {execution_time:.4f} seconds.")

        self.assertIsInstance(importances, np.ndarray)
        self.assertEqual(len(importances), self.X_train.shape[1])

    ### Tests for Decision Tree ###

    def test_tree_structure(self):
        # Ensure tree has a valid structure
        self.assertIsInstance(self.tree.root, dict)
        self.assertIn("feature", self.tree.root)
        self.assertIn("threshold", self.tree.root)
        self.assertIn("left", self.tree.root)
        self.assertIn("right", self.tree.root)

    def test_tree_prediction(self):
        predictions = self.tree_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(all(p in [0, 1] for p in predictions))  # Ensure binary output

    def test_best_split(self):
        feature, threshold = self.tree.best_split(self.X_train, self.y_train)
        self.assertIsNotNone(feature)
        self.assertIsNotNone(threshold)
        self.assertGreaterEqual(feature, 0)
        self.assertLess(feature, self.X_train.shape[1])

    ### Tests for RandomForest ###

    def test_random_forest_training(self):
        # Check if the RandomForest model is properly trained (i.e., has trees)
        self.assertGreater(len(self.rf_model.trees), 0)
        self.assertEqual(len(self.rf_model.trees), 10)  # We set n=10

    def test_random_forest_prediction(self):
        predictions = self.rf_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(all(p in [0, 1] for p in predictions))  # Ensure binary output

    def test_random_forest_misclassification_rate(self):
        def compute_misclassification_rate(model, X_test, y_test, confidence=0.95):
            y_pred = model.predict(X_test)
            error_rate = np.mean(y_pred != y_test)
            n = len(y_test)
            z = norm.ppf(1 - (1 - confidence) / 2)  # 95% CI z-score
            margin = z * np.sqrt((error_rate * (1 - error_rate)) / n)
            return error_rate, (max(0, error_rate - margin), min(1, error_rate + margin))

        # Compute for RandomForest
        error_rf, ci_rf = compute_misclassification_rate(self.rf_model, self.X_test, self.y_test)
        self.assertGreaterEqual(error_rf, 0)
        self.assertLessEqual(error_rf, 1)
        self.assertGreaterEqual(ci_rf[0], 0)
        self.assertLessEqual(ci_rf[1], 1)

    def test_random_forest_feature_importance(self):
        importances = self.rf_model.importance()
        self.assertEqual(len(importances), self.X_train.shape[1])  # Should match feature count
        self.assertTrue(np.all(importances >= 0))  # Importance should be non-negative

    ### Edge Case Tests ###

    # def test_empty_dataset(self):
    #     empty_X = np.empty((0, self.X_train.shape[1]))
    #     empty_y = np.empty((0,))
    #     with self.assertRaises(ValueError):
    #         self.forest.build(empty_X, empty_y)

    def test_single_sample(self):
        single_X = np.array([[5, 3]])
        single_y = np.array([1])
        rf_model = self.forest.build(single_X, single_y)
        prediction = rf_model.predict(single_X)
        self.assertEqual(prediction[0], 1)  # Should predict the only available class

    def test_all_same_labels(self):
        same_y = np.array([1, 1, 1, 1, 1])
        rf_model = self.forest.build(self.X_train, same_y)
        prediction = rf_model.predict(self.X_test)
        self.assertTrue(all(p == 1 for p in prediction))  # Should always predict 1


class MyTest(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [1, 10, 5, 100],
            [2, 20, 10, 200],
            [2, 20, 5, 100],
            [3, 30, 15, 300],
            [3, 30, 15, 300],
            [3, 30, 5, 100]
        ])

        self.y = np.array([0, 0, 1, 1, 0, 1])

        self.X_1 = np.array([[2, 20, 10, 200],
                             [2, 20, 10, 200],
                             [2, 20, 10, 200]])
        self.y_1 = np.array([0, 0, 1])

    def test_tree_splits(self):
        t = Tree(rand=None, get_candidate_columns=all_columns)
        tree_model = t.build(self.X, self.y)

        # check first split should be feature_1 value:1
        self.assertEqual(tree_model.root["feature"], 0)
        self.assertEqual(tree_model.root["threshold"], 1.5)

        # check left_subtree is pure and class 0
        left_subtree = tree_model.root["left"]
        self.assertEqual(left_subtree["prediction"], 0)

        # check right_subtree is split by feature_3 value:5
        right_subtree = tree_model.root["right"]
        self.assertEqual(right_subtree["feature"], 2)
        self.assertEqual(right_subtree["threshold"], 7.5)

        # check right_left_subtree is pure and class 1
        right_left_subtree = right_subtree["left"]
        self.assertEqual(right_left_subtree["prediction"], 1)

        # check right_right_subtree
        right_right_subtree = right_subtree["right"]
        self.assertEqual(right_right_subtree["feature"], 0)
        self.assertEqual(right_right_subtree["threshold"], 2.5)

        # check right_right_left subtree is pure and class 0
        right_right_left_subtree = right_right_subtree["left"]
        self.assertEqual(right_right_left_subtree["prediction"], 0)

        # check right_right_right subtree predicts class 0
        right_right_right_subtree = right_right_subtree["right"]
        self.assertEqual(right_right_right_subtree["prediction"], 0)

    def test_tree_edge_case1(self):
        # properly handle when multiple rows have same feature values, but different classes
        t = Tree(rand=None, get_candidate_columns=all_columns)
        tree_model = t.build(self.X_1, self.y_1)
        self.assertEqual(tree_model.root["prediction"], 0)


if __name__ == '__main__':
    unittest.main()
