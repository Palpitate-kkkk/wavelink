import unittest

from experiments.clock_link import episode


class ClockLinkTests(unittest.TestCase):
    def test_estimated_correction_and_rejection(self):
        for snr in (20, -8):
            raw, estimated, oracle = episode(200, snr, 710101)
            self.assertEqual(oracle['applied_ratio'], '5000/5001')
            self.assertEqual(raw['offered'], estimated['offered'])
            if snr == 20:
                self.assertEqual(estimated['erased'], 0)
                self.assertLess(estimated['errors'], raw['errors'])
            else:
                self.assertEqual(estimated['erased'], estimated['offered'])
                self.assertIsNone(estimated['applied_ratio'])
