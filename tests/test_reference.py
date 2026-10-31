import json
from pathlib import Path
import unittest
import numpy as np
from wavelink.reference import ReferenceTrackedLMMSE, rotate_grid_reference
from wavelink.slot_tracking import SlotTrackedLMMSE
from wavelink.receiver import comb_grid, zero_force
from wavelink import qpsk_demodulate
from experiments.acquisition_ber import packet
from experiments.captured_tracking_ber import grid
from experiments.reference_ber import validate, episode
from experiments.alignment_ber import episode as parent_episode


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.layout = comb_grid(2)
        self.rng = np.random.default_rng(330001)
        self.pilots = np.ones((20, 28), complex)

    def frame(self):
        taps = self.rng.normal(size=17)+1j*self.rng.normal(size=17)
        return np.broadcast_to(np.fft.fft(taps, 64), (20, 64)).copy()

    def test_unitary_inverse_and_input_unchanged(self):
        y = self.frame()
        before = y.copy()
        rotated = rotate_grid_reference(y, 13)
        np.testing.assert_allclose(abs(rotated), abs(y), atol=1e-14)
        np.testing.assert_allclose(rotate_grid_reference(rotated, -13), y, atol=1e-12)
        np.testing.assert_array_equal(before, y)

    def test_zero_shift_reproduces_parent_across_bootstrap_gap_final(self):
        a, b = ReferenceTrackedLMMSE(self.layout), SlotTrackedLMMSE(self.layout)
        for slot in range(150):
            y = None if 80 <= slot < 100 else self.frame()
            p = None if y is None else self.pilots
            np.testing.assert_array_equal(a.step(y, p, final=slot == 149), b.step(y, p, final=slot == 149))
            self.assertEqual(a.last['statistics_consumed'], b.last['statistics_consumed'])
        self.assertEqual(a.events, b.events)
        with self.assertRaises(RuntimeError):
            a.step()

    def test_noiseless_safe_shift_has_correct_sign_and_original_data_decode(self):
        protocol = json.loads(Path('experiments/followup.json').read_text())
        for seed in range(330101, 330121):
            x, p, bits, start, _ = packet(seed, 0, protocol['states']['initial'], 0.)
            if start >= 4:
                break
        self.assertGreaterEqual(start, 4)
        raw = grid(x, start-4, 0.)
        original = raw.copy()
        canonical = rotate_grid_reference(raw, 4)
        np.testing.assert_allclose(canonical, grid(x, start, 0.), atol=1e-11)
        receiver = ReferenceTrackedLMMSE(self.layout)
        h = receiver.step(raw, p, shift=4)
        decoded = qpsk_demodulate(zero_force(raw[:, self.layout.bins(self.layout.data_tones)], h).ravel())
        np.testing.assert_array_equal(decoded, bits.ravel())
        np.testing.assert_array_equal(raw, original)

    def test_skipped_weak_path_remains_in_raw_window_but_outside_rotated_prior(self):
        taps = np.zeros(64, complex)
        taps[0], taps[12] = .01, 1.
        raw = np.broadcast_to(np.fft.fft(taps), (20, 64)).copy()
        before = raw.copy()
        rotated = rotate_grid_reference(raw, 12)
        self.assertAlmostEqual(abs(np.fft.ifft(rotated[0])[52]), .01)
        receiver = ReferenceTrackedLMMSE(self.layout, gate=.25)
        receiver.step(raw, self.pilots, shift=12)
        np.testing.assert_array_equal(raw, before)
        self.assertTrue(receiver.last['captured'])

    def test_shift_validation_does_not_consume_slot(self):
        r = ReferenceTrackedLMMSE(self.layout)
        for shift in (True, -1, 17, .5):
            with self.assertRaises(ValueError):
                r.step(self.frame(), self.pilots, shift=shift)
        with self.assertRaises(ValueError):
            r.step(shift=1)
        self.assertEqual(r.slots, 0)

    def test_frozen_protocol_and_paired_legacy_comparators(self):
        cfg = json.loads(Path('experiments/reference.json').read_text())
        protocol = validate(cfg)
        new, _, _, timing = episode(cfg, protocol, 'gap32', 18, 1, 330101)
        parent_cfg = json.loads(Path('experiments/alignment.json').read_text())
        old, *_ = parent_episode(parent_cfg, protocol, 'gap32', 18, 1, 330101, [8.])
        for a, b in [('raw_ls','raw_ls'), ('physical_ls','ls_8.0'), ('raw_adaptive','raw_adaptive'),
                     ('raw_frozen','raw_frozen'), ('physical_adaptive','aligned_8.0'), ('oracle_adaptive','oracle_adaptive')]:
            self.assertEqual(new[a], old[b])
        self.assertEqual(timing['raw_window_sha256'], timing['reference_window_sha256'])
        self.assertEqual(timing['reference_new_unsafe'], 0)
        for method, values in new.items():
            if method != 'oracle_adaptive':
                self.assertEqual([v[1] for v in values], [v[1] for v in new['raw_ls']])


if __name__ == '__main__':
    unittest.main()
