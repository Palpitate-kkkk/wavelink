import unittest

from experiments.clock_fallback import fallback


class ClockFallbackTests(unittest.TestCase):
    def test_switches_only_on_rejection(self):
        raw = dict(applied_ratio='1', errors=12, erased=0)
        rejected = dict(applied_ratio=None, errors=0, erased=20)
        accepted = dict(applied_ratio='1', errors=17, erased=0)
        self.assertEqual(fallback(raw, rejected), dict(raw, method='fallback', used_raw=True))
        self.assertEqual(fallback(raw, accepted), dict(accepted, method='fallback', used_raw=False))
