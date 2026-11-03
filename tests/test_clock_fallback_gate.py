import unittest
import numpy as np
from experiments.clock_fallback_gate import choose, quality


class ClockFallbackGateTests(unittest.TestCase):
    def test_fixed_window_score_and_routing(self):
        p = np.array([1, 1j, -1])
        x = np.r_[p, p*2j, np.zeros(3)]
        self.assertAlmostEqual(quality(x, p, [0, 3, 6]), 1)
        raw = {'applied_ratio': '1'}
        rejected = {'applied_ratio': None}
        accepted = {'applied_ratio': '5000/5001'}
        self.assertIs(choose(raw, rejected, .3, .3), raw)
        self.assertIs(choose(raw, rejected, .29, .3), rejected)
        self.assertIs(choose(raw, accepted, 1, .3), accepted)
