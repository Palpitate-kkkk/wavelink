import copy
import json
from pathlib import Path
import unittest
import numpy as np
from wavelink.receiver import comb_grid
from wavelink.tracking import CausalPilotLMMSE
from wavelink.periodic import CachedPeriodicLMMSE
from experiments.runtime_benchmark import orders, summarize, equivalence, replay, validate


class CachedPeriodicTests(unittest.TestCase):
    def compare(self, symbols, window, period, spacing, length, count=200):
        rng = np.random.default_rng(221900)
        layout = comb_grid(spacing)
        cal = (rng.normal(size=(64, symbols, len(layout.pilot_tones))) +
               1j*rng.normal(size=(64, symbols, len(layout.pilot_tones))))
        a = CausalPilotLMMSE(cal, layout, length, window, period)
        b = CachedPeriodicLMMSE(cal, layout, length, window, period)
        for t in range(count):
            y = (rng.normal(size=(symbols, 64)) + 1j*rng.normal(size=(symbols, 64)))*(1 if t < count//2 else 3)
            p = np.ones((symbols, len(layout.pilot_tones)), complex)
            np.testing.assert_allclose(a.estimate(y, p), b.estimate(y, p), atol=1e-10, rtol=1e-10)
            self.assertEqual(a.advance(), b.advance())
            self.assertEqual(a.last_update_after_frame, b.last_update_after_frame)
            np.testing.assert_allclose(a.statistics.powers, b.statistics.powers, atol=1e-10, rtol=1e-10)
            self.assertAlmostEqual(a.statistics.noise_variance, b.statistics.noise_variance, places=11)
        self.assertEqual(a.update_count, b.update_count)
        self.assertEqual(b.fit_frame_visits, 0)
        self.assertEqual(b.feature_frames, window+count)
        return a, b, cal, layout

    def test_sliding_window_long_run_and_power_change(self):
        self.compare(4, 64, 32, 2, 17, 2000)

    def test_odd_symbol_count_shorter_window_and_sparse_grid(self):
        self.compare(5, 16, 7, 4, 13)

    def test_frozen_schedule(self):
        a, b, _, _ = self.compare(3, 8, None, 2, 17)
        self.assertEqual(b.update_count, 0)
        self.assertEqual(a.update_count, 0)

    def test_advance_requires_pending_and_cannot_double_estimate(self):
        _, b, _, layout = self.compare(2, 2, 1, 2, 17, 1)
        with self.assertRaises(RuntimeError):
            b.advance()
        b.estimate(np.zeros((2, 64), complex), np.ones((2, len(layout.pilot_tones))))
        with self.assertRaises(RuntimeError):
            b.estimate(np.zeros((2, 64), complex), np.ones((2, len(layout.pilot_tones))))

    def test_zero_observations_preserve_noise_and_power_floors(self):
        layout = comb_grid(2)
        cal = np.zeros((64, 3, len(layout.pilot_tones)), complex)
        frames = [(np.zeros((3, 64), complex), np.ones((3, len(layout.pilot_tones))), np.zeros((3, 48), int)) for _ in range(70)]
        result = equivalence(cal, frames, layout)
        self.assertTrue(result['passed'])
        self.assertEqual(result['updates'], 2)

    def test_no_final_unused_rebuild_and_replay_same_decisions(self):
        _, _, cal, layout = self.compare(4, 64, 32, 2, 17, 1)
        frames = [(np.zeros((4, 64), complex), np.ones((4, len(layout.pilot_tones))), np.zeros((4, 48), int)) for _ in range(32)]
        a, b = [replay(name, cal, frames, layout) for name in ('periodic', 'cached_periodic')]
        self.assertEqual(a['updates'], 0)
        self.assertEqual(a['output_sha256'], b['output_sha256'])
        self.assertEqual(a['event_sha256'], b['event_sha256'])
        self.assertGreaterEqual(a['total_ms'], a['online_ms'])

    def test_latin_orders_balanced_and_reproducible(self):
        result = orders(100, 3)
        self.assertEqual(result, orders(100, 3))
        for position in range(4):
            self.assertEqual(sorted([row[position] for row in result]), sorted(['periodic', 'cached_periodic', 'innovation', 'followup']*3))

    def test_summary_aggregates_repeats_before_seed_uncertainty(self):
        rows = []
        for name in ('periodic', 'cached_periodic', 'innovation', 'followup'):
            for seed, values in [(1, [1, 2, 99]), (2, [4, 6, 7])]:
                for repeat, value in enumerate(values):
                    rows.append(dict(scenario='test', ebn0_db=6, receiver=name, seed=seed, repeat=repeat,
                                     init_ms=value, online_ms=value, total_ms=2*value, updates=9, bits=100, errors=2))
        result = summarize(rows)[0]
        self.assertEqual(result['online_ms'], 4.)
        self.assertEqual(result['online_ms_seed_se'], 2.)
        self.assertEqual(result['mean_online_ratio_to_cached'], 1.)
        self.assertEqual(result['independent_seed_branches'], 2)
        self.assertEqual(result['replays_per_seed'], 3)

    def test_timing_config_rejects_seed_reuse_and_invalid_cycles(self):
        cfg = json.loads(Path('experiments/runtime.json').read_text())
        validate(cfg)
        for key, value in [('cycles', True), ('cycles', 0), ('seeds', [201001, 221002, 221003, 221004])]:
            other = copy.deepcopy(cfg)
            other[key] = value
            with self.assertRaises(ValueError):
                validate(other)


if __name__ == '__main__':
    unittest.main()
