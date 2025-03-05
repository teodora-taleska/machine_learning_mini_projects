import random
import unittest

import numpy as np

from hw_tree import Tree, RandomForest, hw_tree_full, hw_randomforests


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

    def test_misclassification_rate(self):
        (train_misclass, train_se), (test_misclass, test_se) = hw_tree_full((self.X_train, self.y_train),
                                                                            (self.X_test, self.y_test))

        self.assertGreaterEqual(train_misclass, 0)
        self.assertLessEqual(train_misclass, 1)
        self.assertGreaterEqual(test_misclass, 0)
        self.assertLessEqual(test_misclass, 1)

        self.assertGreaterEqual(train_se, 0)
        self.assertGreaterEqual(test_se, 0)

    def test_best_split(self):
        feature, threshold = self.tree.best_split(self.X_train, self.y_train)
        self.assertIsNotNone(feature)
        self.assertIsNotNone(threshold)
        self.assertGreaterEqual(feature, 0)
        self.assertLess(feature, self.X_train.shape[1])




if __name__ == "__main__":
    unittest.main()
