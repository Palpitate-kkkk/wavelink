import copy
import json
from pathlib import Path
import unittest
import numpy as np
from wavelink import qpsk_demodulate
from wavelink.acquisition import Acquisition, marker_grid
from wavelink.alignment import align_marker
from wavelink.synchronization import rotate_frequency
from wavelink.estimation import TapDomainLS
from wavelink.receiver import comb_grid, zero_force
from experiments.acquisition_ber import packet
from experiments.captured_tracking_ber import grid, episode as parent_episode
from experiments.alignment_ber import episode, select, validate


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(Path('experiments/followup.json').read_text())
        self.cfg = json.loads(Path('experiments/alignment.json').read_text())

    def proposal(self, start, epsilon=0.):
        return Acquisition(start, epsilon, 1., 1., 1., True)

    def example(self):
        for seed in range(320101, 320121):
            result = packet(seed, 0, self.protocol['states']['initial'], 0.)
            if result[3] >= 4:
                return result
        self.fail('unit-test packet with prefix required')

    def test_noiseless_alignment_and_decoding_under_integer_cfo(self):
        x, pilots, bits, start, _ = self.example()
        model = TapDomainLS(comb_grid(2), 17)
        for epsilon in (-1.2, 1.2, 3.2):
            observed = rotate_frequency(x, epsilon, 64)
            p = self.proposal(start-4, epsilon)
            result = align_marker(observed, p, 4.)
            self.assertEqual(result.shift, 4)
            self.assertEqual(result.start, start)
            y = grid(observed, result.start, epsilon)
            h = model.estimate(y, pilots)
            decoded = qpsk_demodulate(zero_force(y[:, model.layout.bins(model.layout.data_tones)], h).ravel())
            np.testing.assert_array_equal(decoded, bits.ravel())

    def test_correct_noiseless_boundary_stays_put(self):
        x, _, _, start, _ = self.example()
        self.assertEqual(align_marker(x, self.proposal(start), 8.).shift, 0)

    def test_only_original_marker_prefix_read(self):
        x, _, _, start, _ = self.example()
        p = self.proposal(start-4)
        original = align_marker(x, p, 4.)
        x[287:] = np.nan
        self.assertEqual(original, align_marker(x, p, 4.))

    def test_weak_first_path_can_be_skipped(self):
        # Controlled leakage/noise-like tail raises the observable threshold.
        # The receiver cannot certify that the first physical path is above it.
        taps = np.zeros(64, complex)
        taps[4], taps[16], taps[17:] = .01, 1., .1
        x = np.zeros(1984, complex)
        x[160:224] = np.fft.ifft(marker_grid()*np.fft.fft(taps), norm='ortho')
        a = align_marker(x, self.proposal(0), 2.)
        self.assertEqual(a.shift, 16)
        self.assertGreater(a.shift, 4)

    def test_no_significant_tap_falls_back_without_rejecting_packet(self):
        a = align_marker(np.zeros(1984), self.proposal(63), 4.)
        self.assertFalse(a.significant_tap)
        self.assertEqual(a.start, 63)
        self.assertEqual(a.shift, 0)

    def test_maximum_shift_still_has_complete_payload(self):
        taps = np.zeros(64, complex)
        taps[16] = 1.
        x = np.zeros(1984, complex)
        x[223:287] = np.fft.ifft(marker_grid()*np.fft.fft(taps), norm='ortho')
        a = align_marker(x, self.proposal(63), 4.)
        self.assertEqual(a.start, 79)
        self.assertEqual(grid(x, a.start, 0.).shape, (20, 64))

    def test_phase_and_scale_invariance(self):
        x, _, _, start, _ = self.example()
        p = self.proposal(start-4)
        a, b = align_marker(x, p, 4.), align_marker(x*(.3-.2j), p, 4.)
        self.assertEqual(a.shift, b.shift)
        self.assertAlmostEqual(b.threshold, a.threshold*.13, places=15)

    def test_invalid_proposal_and_parameter(self):
        for multiple in (True, 0., -1., float('nan'), 1j):
            with self.assertRaises(ValueError):
                align_marker(np.zeros(1984), self.proposal(0), multiple)
        with self.assertRaises(ValueError):
            align_marker(np.zeros(100), self.proposal(0), 4.)
        with self.assertRaises(ValueError):
            align_marker(np.zeros(1984), Acquisition(0, 0., 0., 0., 0., False), 4.)

    def test_development_selection_and_failure_fallback(self):
        rows = [dict(scenario='stationary', ebn0_db=18, receiver='raw_adaptive', window='all', loss=.1)]
        for c, loss in [(2., .11), (4., .1), (8., .1)]:
            rows.append(dict(scenario='stationary', ebn0_db=18, receiver=f'aligned_{c}', window='all', loss=loss))
        chosen, candidates, eligible = select(dict(summary=rows), [2., 4., 8.])
        self.assertEqual(chosen, 4.)
        self.assertTrue(eligible)
        self.assertFalse(candidates[0]['eligible'])
        for r in rows[1:]:
            r['loss'] = .2
        self.assertEqual(select(dict(summary=rows), [2., 4., 8.])[0], 2.)
        self.assertFalse(select(dict(summary=rows), [2., 4., 8.])[2])

    def test_new_seeds_and_weak_first_excluded_from_development(self):
        validate(self.cfg)
        c = copy.deepcopy(self.cfg)
        c['development']['seeds'][0] = 291001
        with self.assertRaises(ValueError):
            validate(c)
        c = copy.deepcopy(self.cfg)
        c['development']['scenarios'].append('weak_first')
        with self.assertRaises(ValueError):
            validate(c)

    def test_full_raw_path_matches_frozen_parent_and_alignment_keeps_mask(self):
        old, _, _, _ = parent_episode(self.protocol, 'gap32', 18, 1, 320101, [.25])
        new, costs, _, timing, _ = episode(self.cfg, self.protocol, 'gap32', 18, 1, 320101, [4.])
        self.assertEqual([tuple(x[:2]) for x in old['gate_0.25']], new['raw_adaptive'])
        self.assertEqual([tuple(x[:2]) for x in old['ls']], new['raw_ls'])
        for key in new:
            if key != 'oracle_adaptive':
                self.assertEqual([x[1] for x in new[key]], [x[1] for x in new['raw_ls']])
        self.assertTrue(all(c['gap_updates'] == 0 for c in costs))
        self.assertEqual(timing[0]['captured'], sum(x[1] == 0 for x in new['raw_ls']))


if __name__ == '__main__':
    unittest.main()
