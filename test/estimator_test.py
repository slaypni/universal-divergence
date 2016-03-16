from __future__ import absolute_import, division, print_function

import sys
import unittest

import numpy as np
from sklearn import mixture

sys.path.append('./src')
from estimator import estimate  # noqa


class EstimatorTestCase(unittest.TestCase):
    def setUp(self):
        obs1 = np.concatenate((np.random.randn(100, 20),
                              10 + np.random.randn(300, 20)))
        g1 = mixture.GMM(n_components=2).fit(obs1)
        self.x1 = g1.sample(500)
        self.y1 = g1.sample(400)

        obs2 = np.concatenate((np.random.randn(200, 20),
                              5 + np.random.randn(200, 20)))
        g2 = mixture.GMM(n_components=2).fit(obs2)
        self.x2 = g2.sample(500)
        self.y2 = g2.sample(400)

    def test_k_estimator(self):
        self.assertAlmostEqual(estimate(self.x1, self.y1, k=1), 0, places=0)
        self.assertAlmostEqual(estimate(self.x2, self.y2, k=1), 0, places=0)
        self.assertNotAlmostEqual(estimate(self.x1, self.y2, k=1), 0, places=0)

    def test_generalized_estimator(self):
        self.assertAlmostEqual(estimate(self.x1, self.y1), 0, places=0)
        self.assertAlmostEqual(estimate(self.x2, self.y2), 0, places=0)
        self.assertNotAlmostEqual(estimate(self.x1, self.y2), 0, places=0)


if __name__ == '__main__':
    unittest.main()
