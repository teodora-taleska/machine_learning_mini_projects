import random
import unittest

import numpy as np
import time
from hw_tree import Tree, RandomForest, hw_tree_full, hw_randomforests
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
            return error_rate, (error_rate - margin, error_rate + margin)

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

    def test_empty_dataset(self):
        empty_X = np.array([])
        empty_y = np.array([])
        with self.assertRaises(ValueError):
            self.forest.build(empty_X, empty_y)

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

if __name__ == '__main__':
    unittest.main()
