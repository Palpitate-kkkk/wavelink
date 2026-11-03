import unittest
import numpy as np
from experiments.clock_estimation_demo import fixture
from wavelink.clock_estimation import estimate_clock


class ClockEstimationTests(unittest.TestCase):
    def test_both_drift_signs_with_missing_marker(self):
        for ppm in (-200, 0, 200):
            y, p, t = fixture(ppm, 551001, 20, missing=True)
            result = estimate_clock(y, p, t, radius=32)
            self.assertAlmostEqual(result['ppm'], ppm, delta=1)
            self.assertAlmostEqual(result['offset_samples'], 5, delta=.3)
            self.assertEqual(len(result['anchors']), 8)
            self.assertNotIn(t[4], [a[0] for a in result['anchors']])

    def test_no_signal_refuses_estimate(self):
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            estimate_clock(np.zeros(100), np.ones(5), [10, 30, 50], radius=4)
