import copy
import json
from pathlib import Path
import unittest
import numpy as np
from wavelink.acquisition import acquire, acquisition_header, marker_grid
from wavelink.synchronization import rotate_frequency
from wavelink.estimation import TapDomainLS
from wavelink.receiver import comb_grid
from experiments.acquisition_ber import packet, decode, select, apply_threshold, summarize, validate


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(Path('experiments/followup.json').read_text())
        self.model = TapDomainLS(comb_grid(2), 17)

    def test_public_header_is_fixed_and_marker_breaks_repetition(self):
        a, b = acquisition_header(), acquisition_header()
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(a), 224)
        np.testing.assert_array_equal(a[16:48], a[48:80])
        self.assertFalse(np.allclose(a[160:192], a[192:224]))
        # Unit-modulus full-tone marker gives orthogonal cyclic delay templates.
        marker = np.fft.ifft(marker_grid(), norm='ortho')
        matrix = np.array([np.roll(marker, i) for i in range(17)]).T
        np.testing.assert_allclose(matrix.conj().T@matrix, 64*np.eye(17), atol=1e-12)

    def test_noiseless_integer_offsets_and_safe_early_timing_decode(self):
        for state, max_delay in [('initial', 12), ('weak15', 15)]:
            x, p, bits, start, _ = packet(280001, 0, self.protocol['states'][state], 0.)
            for epsilon in (0., .2, 1.2, -1.2, 3.2):
                y = rotate_frequency(x, epsilon, 64)
                result = acquire(y, .1)
                self.assertTrue(result.accepted)
                self.assertLessEqual(max_delay-16, result.start-start)
                self.assertLessEqual(result.start-start, 0)
                self.assertAlmostEqual(result.epsilon, epsilon, places=10)
                np.testing.assert_array_equal(decode(y, p, result.start, result.epsilon, self.model), bits.ravel())

    def test_out_of_range_not_silently_given_true_integer_offset(self):
        x, _, _, _, _ = packet(280001, 0, self.protocol['states']['initial'], 0.)
        result = acquire(rotate_frequency(x, 5.2, 64), .1)
        self.assertGreater(abs(result.epsilon-5.2), 1.)
        self.assertLessEqual(abs(result.epsilon), 5.)

    def test_search_boundaries(self):
        header = acquisition_header()
        for start in (0, 63):
            x = np.zeros(287, complex)
            x[start:start+224] = header
            result = acquire(rotate_frequency(x, -1.2, 64), .1)
            self.assertLessEqual(-16, result.start-start)
            self.assertLessEqual(result.start-start, 0)
            self.assertAlmostEqual(result.epsilon, -1.2, places=10)

    def test_only_fixed_prefix_used_and_future_nan_ignored(self):
        x, _, _, _, _ = packet(280001, 0, self.protocol['states']['initial'], .01)
        before = acquire(x, .1)
        x[287:] = np.nan
        self.assertEqual(before, acquire(x, .1))

    def test_global_phase_and_scale_do_not_change_score(self):
        x, _, _, _, _ = packet(280001, 0, self.protocol['states']['weak15'], .01)
        a, b = acquire(x, .1), acquire(x*(.3-.8j), .1)
        self.assertEqual(a.start, b.start)
        self.assertAlmostEqual(a.epsilon, b.epsilon, places=12)
        self.assertAlmostEqual(a.score, b.score, places=12)

    def test_empty_energy_rejected_even_at_zero_threshold(self):
        result = acquire(np.zeros(287), 0.)
        self.assertFalse(result.accepted)
        self.assertEqual(result.score, 0.)

    def test_invalid_prefix_threshold_and_search(self):
        for samples, threshold, search in [(np.ones(286), .1, 64), (np.full(287, np.nan), .1, 64), (np.ones(287), True, 64), (np.ones(287), float('nan'), 64), (np.ones(287), .1, 0), (np.ones(287), 1.1, 64)]:
            with self.assertRaises(ValueError):
                acquire(samples, threshold, search)

    def test_development_selection_null_gate_and_loss_objective(self):
        signals = [dict(score=.15, bits=100, proposal_errors=5), dict(score=.5, bits=100, proposal_errors=10)]
        nulls = [dict(score=.08)]*100
        threshold, candidates, eligible = select(signals, nulls, [.05, .1, .2])
        self.assertTrue(eligible)
        self.assertEqual(threshold, .1)
        self.assertFalse(candidates[0]['eligible'])
        self.assertAlmostEqual(candidates[1]['mean_offered_bit_loss'], .075)

    def test_selection_tie_higher_threshold_and_failure_retained(self):
        signals = [dict(score=.5, bits=100, proposal_errors=5)]
        self.assertEqual(select(signals, [dict(score=.01)], [.1, .2])[0], .2)
        threshold, _, eligible = select(signals, [dict(score=.9)], [.1, .2])
        self.assertEqual(threshold, .2)
        self.assertFalse(eligible)

    def base_row(self):
        return dict(state='initial', ebn0_db=18, epsilon=1.2, score=.1, bits=100, proposal_errors=20,
                    oracle_errors=2, fractional_errors=50, capture_correct=0, timing_safe=0, strict_timing=0)

    def test_miss_is_erasure_and_wrong_capture_is_not_truth_filtered(self):
        row = self.base_row()
        rejected = apply_threshold([row], .2)[0]
        self.assertEqual(rejected['offered_bit_loss'], 1.)
        self.assertEqual(rejected['accepted_bits'], 0)
        accepted = apply_threshold([row], .05)[0]
        self.assertEqual(accepted['errors'], 20)
        self.assertEqual(accepted['accepted_bits'], 100)
        self.assertEqual(accepted['false_capture'], 1)
        self.assertEqual(accepted['offered_bit_loss'], .2)

    def test_summary_keeps_conditional_ber_and_offered_loss_distinct(self):
        a, b = self.base_row(), self.base_row()
        b['score'] = .5
        r = summarize(apply_threshold([a, b], .2))[0]
        self.assertEqual(r['accepted_ber'], .2)
        self.assertEqual(r['offered_bit_loss'], .6)
        self.assertEqual(r['miss_rate'], .5)
        self.assertEqual(r['false_capture'], .5)
        empty = summarize(apply_threshold([a], .9))[0]
        self.assertEqual(empty['accepted_ber'], '')
        self.assertEqual(empty['offered_bit_loss'], 1.)

    def test_full_true_compensation_restores_zero_cfo_decisions(self):
        x, p, _, start, _ = packet(280001, 0, self.protocol['states']['weak15'], .02)
        reference = decode(x, p, start, 0., self.model)
        for epsilon in (-1.2, 3.2, 5.2):
            np.testing.assert_array_equal(decode(rotate_frequency(x, epsilon, 64), p, start, epsilon, self.model), reference)

    def test_seed_isolation_and_frozen_protocol(self):
        cfg = json.loads(Path('experiments/acquisition.json').read_text())
        validate(cfg)
        other = copy.deepcopy(cfg)
        other['development']['seeds'][0] = 231001
        with self.assertRaises(ValueError):
            validate(other)


if __name__ == '__main__':
    unittest.main()
