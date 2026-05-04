"""Unit tests for backend/aggregation robust Fed helpers."""

import unittest
import numpy as np

from backend.aggregation import (
    coordinate_trimmed_mean,
    weighted_fedavg,
    multi_krum_indices,
)


class TestWeightedFedavg(unittest.TestCase):
    def test_two_clients_weight_ratio(self):
        w1 = [np.ones((2,)), np.zeros((3,))]
        w2 = [np.zeros((2,)), np.ones((3,))]
        nw = weighted_fedavg([(w1, 2.0), (w2, 8.0)])
        np.testing.assert_allclose(nw[0], np.array([0.2, 0.2]))
        np.testing.assert_allclose(nw[1], np.array([0.8, 0.8, 0.8]))


class TestTrimmedMean(unittest.TestCase):
    def test_removes_obvious_spike_each_coordinate(self):
        x = np.array(
            [
                [10.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 10.0],
            ],
            dtype=np.float64,
        )
        out = coordinate_trimmed_mean(x, beta_per_tail=0.25)
        np.testing.assert_allclose(out, np.array([0.0, 0.0]), rtol=0, atol=1e-6)


class TestMultiKrum(unittest.TestCase):
    def test_picks_most_central_three_of_four(self):
        clean = np.array([[0.0, 0.0], [0.1, 0.05], [-0.05, 0.1], [100.0, -100.0]])
        idx = multi_krum_indices(clean, multi_k=3, neighbor_m=1)
        self.assertEqual(len(idx), 3)
        self.assertNotIn(3, idx)


if __name__ == "__main__":
    unittest.main()
