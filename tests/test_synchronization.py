import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from wavelink import ofdm_modulate, ofdm_demodulate
from wavelink.receiver import apply_multipath
from wavelink.synchronization import rotate_frequency, estimate_repeated_cfo
from experiments.cfo_ber import training, burst, receive, validate, episode, summarize


class SynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path('experiments/followup.json').read_text())

    def test_chunk_continuity_and_energy(self):
        x = np.random.default_rng(233001).normal(size=135)+1j
        whole = rotate_frequency(x, .2, 64, 500)
        split = np.r_[rotate_frequency(x[:61], .2, 64, 500), rotate_frequency(x[61:], .2, 64, 561)]
        np.testing.assert_array_equal(whole, split)
        np.testing.assert_allclose(abs(whole), abs(x), atol=1e-14)
        np.testing.assert_allclose(rotate_frequency(whole, -.2, 64, 500), x, atol=1e-13)

    def test_repeated_header_energy_period_and_cp(self):
        h = training(233001, 0, 1, 0)
        self.assertEqual(len(h), 144)
        for i in (1, 2, 3):
            np.testing.assert_array_equal(h[16:48], h[16+i*32:48+i*32])
        np.testing.assert_array_equal(h[:16], h[-16:])
        self.assertAlmostEqual(float(np.sum(abs(h[16:])**2)), 128.)

    def test_noiseless_positive_negative_capture_and_cp_edge_multipath(self):
        header = training(233001, 0, 1, 0)
        taps = np.zeros(17, complex)
        taps[0], taps[16] = .2+.3j, 1.-.2j
        y, _ = apply_multipath(header, taps)
        for epsilon in (-.99, -.2, 0., .2, .99):
            observed = rotate_frequency(y, epsilon, 64, 771)
            value, correlation = estimate_repeated_cfo(observed[16:], 32, 64)
            self.assertAlmostEqual(value, epsilon, places=12)
            self.assertAlmostEqual(correlation, 1., places=12)

    def test_out_of_capture_alias_retained(self):
        header = training(233001, 0, 1, 0)
        for epsilon, expected in [(1.2, -.8), (-1.2, .8)]:
            value, _ = estimate_repeated_cfo(rotate_frequency(header, epsilon, 64)[16:], 32, 64)
            self.assertAlmostEqual(value, expected, places=12)

    def test_correlation_invariant_to_unknown_static_phase_and_gain(self):
        h = training(233001, 0, 1, 0)[16:]
        value, score = estimate_repeated_cfo(rotate_frequency(h*(.04-.1j), .17, 64), 32, 64)
        self.assertAlmostEqual(value, .17, places=12)
        self.assertAlmostEqual(score, 1., places=12)

    def test_exact_single_tone_ici_and_energy(self):
        grid = np.zeros((1, 64), complex)
        grid[0, 4] = 1.
        samples = ofdm_modulate(grid, 16)
        epsilon = .2
        received = ofdm_demodulate(rotate_frequency(samples, epsilon, 64), 64, 16)[0]
        coefficient = abs(np.sin(np.pi*epsilon)/(64*np.sin(np.pi*epsilon/64)))
        self.assertAlmostEqual(abs(received[4]), coefficient, places=12)
        self.assertAlmostEqual(float(np.sum(abs(received)**2)), 1., places=12)
        self.assertGreater(float(np.sum(abs(received)**2)-abs(received[4])**2), 0.)

    def test_oracle_restores_shared_zero_cfo_with_receiver_noise(self):
        self.cfg['symbols_per_frame'] = 4
        latent, _, _ = burst(233001, 0, 2, 0, self.cfg['states']['initial'], .02, self.cfg)
        reference = ofdm_demodulate(latent[144:], 64, 16)[1:]
        for epsilon in (0., -.2, 1.2):
            values, _, _ = receive(latent, epsilon, 1280, self.cfg)
            np.testing.assert_allclose(values['oracle'], reference, atol=1e-12)
        self.assertTrue(np.isfinite(latent).all())

    def test_training_estimate_does_not_read_future_payload(self):
        latent, _, _ = burst(233001, 0, 2, 0, self.cfg['states']['initial'], .02, self.cfg)
        altered = latent.copy()
        altered[144:] *= 100-30j
        _, first, _ = receive(latent, .2, 0, self.cfg)
        _, second, _ = receive(altered, .2, 0, self.cfg)
        self.assertEqual(first, second)

    def test_estimated_branch_uses_only_returned_estimate(self):
        latent, _, _ = burst(233001, 0, 2, 0, self.cfg['states']['initial'], .02, self.cfg)
        with patch('experiments.cfo_ber.estimate_repeated_cfo', return_value=(0., .5)):
            values, estimate, _ = receive(latent, .2, 0, self.cfg)
        np.testing.assert_array_equal(values['estimated'], values['uncorrected'])
        self.assertEqual(estimate, 0.)

    def test_undefined_correlation_and_invalid_arguments(self):
        for values, lag, n in [(np.zeros(64), 32, 64), (np.ones(32), 32, 64), (np.ones(64), 0, 64), (np.full(64, np.nan), 32, 64)]:
            with self.assertRaises(ValueError):
                estimate_repeated_cfo(values, lag, n)
        for epsilon, n, offset in [(float('nan'), 64, 0), (True, 64, 0), (.2, 0, 0), (.2, 64, -1), (.2, 64, 1.5)]:
            with self.assertRaises(ValueError):
                rotate_frequency(np.ones(10), epsilon, n, offset)

    def test_short_complete_episode_oracle_equivalence_and_causality(self):
        self.cfg.update(symbols_per_frame=4, episode_frames=16)
        spec = next(x for x in self.cfg['scenarios'] if x['name'] == 'stationary')
        metrics, diagnostics, events, trackers = episode(self.cfg, spec, 18, 0, 233001, 64)
        self.assertEqual(len(metrics), 12)
        self.assertEqual(len(diagnostics), (64+16)*4)
        for epsilon in (0., .05, .2, 1.2):
            self.assertEqual(metrics[epsilon, 'oracle'], metrics[0., 'oracle'])
        self.assertTrue(all(e['applies_from_frame'] < 16 for e in events))
        self.assertTrue(all(t.last_update_after_frame < 15 for t in trackers.values()))

    def test_frozen_design_and_seed_isolation(self):
        audit = json.loads(Path('experiments/cfo.json').read_text())
        validate(audit)
        for key, value in [('seeds', list(range(201001, 201017))), ('offsets', [0., .1, .2, 1.2])]:
            other = copy.deepcopy(audit)
            other[key] = value
            with self.assertRaises(ValueError):
                validate(other)

    def test_episode_se_uses_paired_differences(self):
        rows = [dict(scenario='test', ebn0_db=18, epsilon=.2, branch='estimated', window='post_all', bits=100,
                     errors=e, ber=e/100, paired_minus_oracle=d, paired_minus_uncorrected=-d, paired_minus_zero_oracle=d)
                for e, d in [(2, .01), (4, .03)]]
        r = summarize(rows)[0]
        self.assertEqual(r['episodes'], 2)
        self.assertEqual(r['bits'], 200)
        self.assertAlmostEqual(r['paired_minus_oracle'], .02)
        self.assertAlmostEqual(r['paired_minus_oracle_se'], .01)


if __name__ == '__main__':
    unittest.main()
