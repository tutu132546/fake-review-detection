import sys
from pathlib import Path
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/models'))
from cohort_uncertainty import bootstrap_auc, grouped_auc


class UncertaintyTests(unittest.TestCase):
    def test_weighted_ties_match_explicit_resampling(self):
        labels = np.array([0, 1, 0, 1, 1])
        scores = np.array([1., 2., 2., 3., 1.])
        weights = np.array([2, 3, 1, 0, 2])
        groups = np.unique(scores, return_inverse=True)[1]
        expected = roc_auc_score(np.repeat(labels, weights), np.repeat(scores, weights))
        self.assertAlmostEqual(grouped_auc(groups, labels, weights), expected)

    def test_paired_identical_models_have_zero_difference(self):
        labels = [0, 0, 1, 1]
        scores = np.array([[1, 1], [3, 3], [2, 2], [4, 4]])
        first = bootstrap_auc(labels, scores, 100, 42)
        second = bootstrap_auc(labels, scores, 100, 42)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, 0], first[:, 1])
        self.assertGreater(first[:, 0].std(), 0)

    def test_perfect_ranking_and_all_ties(self):
        result = bootstrap_auc([0, 0, 1, 1], np.array([[0, 1], [0, 1], [1, 1], [1, 1]]), 100)
        np.testing.assert_array_equal(result[:, 0], 1)
        np.testing.assert_array_equal(result[:, 1], .5)

    def test_single_class_is_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap_auc([0, 0], np.ones((2, 1)), 100)


if __name__ == '__main__':
    unittest.main()
