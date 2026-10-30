import copy
import json
from pathlib import Path
import unittest
import numpy as np
from wavelink.receiver import comb_grid
from wavelink.estimation import TapDomainLS
from wavelink.followup import FollowupPilotLMMSE
from wavelink.slot_tracking import SlotTrackedLMMSE
from experiments.captured_tracking_ber import validate, select, summarize


class SlotTrackingTests(unittest.TestCase):
    def setUp(self):
        self.layout = comb_grid(2)
        self.rng = np.random.default_rng(300101)
        self.fp = TapDomainLS(self.layout, 17).pilot_design
        self.p = np.ones((20, len(self.layout.pilot_tones)), complex)

    def frame(self):
        taps = (self.rng.normal(size=17)+1j*self.rng.normal(size=17))/np.sqrt(34)
        h = np.exp(-2j*np.pi*np.outer(np.arange(64), np.arange(17))/64)@taps
        noise = (self.rng.normal(size=(20, 64))+1j*self.rng.normal(size=(20, 64)))*.01
        return np.broadcast_to(h, (20, 64))+noise

    def boot(self, receiver):
        values = []
        for _ in range(64):
            y = self.frame()
            values.append(y[:, self.layout.bins(self.layout.pilot_tones)])
            receiver.step(y, self.p)
        self.assertEqual(receiver.bootstrap_slot, 63)
        return np.asarray(values)

    def test_all_admitted_matches_frozen_legacy_followup(self):
        r = SlotTrackedLMMSE(self.layout)
        cal = self.boot(r)
        baseline = FollowupPilotLMMSE(cal, self.layout, 17, innovation_threshold=2., refit_delay=64)
        for _ in range(80):
            y = self.frame()
            expected = baseline.estimate(y, self.p)
            actual = r.step(y, self.p)
            np.testing.assert_array_equal(actual, expected)
            baseline.advance()
            np.testing.assert_array_equal(r.tracker.model.mapping, baseline.model.mapping)
        self.assertEqual(r.tracker.update_count, baseline.update_count)

    def test_missing_slots_only_advance_physical_clock(self):
        r = SlotTrackedLMMSE(self.layout)
        self.boot(r)
        mapping = r.tracker.model.mapping.copy()
        history = np.asarray(r.tracker.history).copy()
        for _ in range(100):
            self.assertIsNone(r.step())
        self.assertEqual(r.slots, 164)
        self.assertEqual(r.tracker.frames_observed, 0)
        self.assertEqual(r.consumed, 64)
        np.testing.assert_array_equal(mapping, r.tracker.model.mapping)
        np.testing.assert_array_equal(history, r.tracker.history)
        r.step(self.frame(), self.p)
        self.assertEqual(r.maximum_history_span, 164)

    def test_no_zero_fill_before_bootstrap(self):
        r = SlotTrackedLMMSE(self.layout)
        for _ in range(30):
            r.step()
        for _ in range(63):
            r.step(self.frame(), self.p)
        self.assertIsNone(r.tracker)
        r.step(self.frame(), self.p)
        self.assertEqual(r.bootstrap_slot, 93)
        self.assertEqual(r.last['ready_before'], False)
        self.assertEqual(r.last['ready_after'], True)

    def test_gate_filters_statistics_not_captured_output(self):
        r = SlotTrackedLMMSE(self.layout, gate=.15)
        self.boot(r)
        y = self.frame()
        # Construct pilot observations wholly outside the assumed subspace.
        v = self.rng.normal(size=(20, 28))+1j*self.rng.normal(size=(20, 28))
        v -= v@r.projector.T
        y[:, self.layout.bins(self.layout.pilot_tones)] = v
        expected = r.tracker.model.estimate(y, self.p)
        before = r.tracker.frames_observed
        actual = r.step(y, self.p)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(r.last['statistics_consumed'])
        self.assertEqual(r.tracker.frames_observed, before)
        self.assertEqual(r.gate_rejected, 1)

    def test_current_estimate_uses_prior_when_step_updates(self):
        r = SlotTrackedLMMSE(self.layout)
        self.boot(r)
        r.tracker.threshold = 1e-20
        for _ in range(7):
            r.step(self.frame(), self.p)
        y = self.frame()
        expected = r.tracker.model.estimate(y, self.p)
        actual = r.step(y, self.p)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(r.last['updated'])
        self.assertEqual(r.events[-1]['applies_from_slot'], r.slots)

    def test_followup_waits_for_admitted_observations_across_gap(self):
        r = SlotTrackedLMMSE(self.layout)
        self.boot(r)
        r.tracker.threshold = 1e20
        r.tracker._innovation_score = lambda noise: 3. if r.tracker.frames_observed == 8 else 1.
        for _ in range(8):
            r.step(self.frame(), self.p)
        self.assertEqual(r.tracker.armed_plans, 1)
        for _ in range(100):
            r.step()
        self.assertEqual(r.tracker.followup_count, 0)
        for _ in range(63):
            r.step(self.frame(), self.p)
        self.assertEqual(r.tracker.followup_count, 0)
        r.step(self.frame(), self.p)
        self.assertEqual(r.tracker.followup_count, 1)
        self.assertEqual(r.events[-1]['kind'], 'followup')
        self.assertEqual(r.events[-1]['after_slot'], 235)

    def test_final_never_bootstraps_or_consumes_and_closes(self):
        r = SlotTrackedLMMSE(self.layout)
        for _ in range(63):
            r.step(self.frame(), self.p)
        r.step(self.frame(), self.p, final=True)
        self.assertIsNone(r.tracker)
        self.assertEqual(r.consumed, 63)
        with self.assertRaises(RuntimeError):
            r.step()
        r = SlotTrackedLMMSE(self.layout)
        self.boot(r)
        r.step(self.frame(), self.p, final=True)
        self.assertEqual(r.tracker.frames_observed, 0)
        self.assertIsNone(r.tracker._pending)

    def test_frozen_comparator_never_refits(self):
        r = SlotTrackedLMMSE(self.layout, adaptive=False)
        self.boot(r)
        initial = r.tracker.model.mapping.copy()
        for _ in range(80):
            r.step(self.frame(), self.p)
        np.testing.assert_array_equal(r.tracker.model.mapping, initial)
        self.assertEqual(r.tracker.update_count, 0)
        self.assertEqual(r.tracker.frames_observed, 80)

    def test_future_observations_do_not_change_past_outputs(self):
        a, b = SlotTrackedLMMSE(self.layout), SlotTrackedLMMSE(self.layout)
        for _ in range(80):
            y = self.frame()
            np.testing.assert_array_equal(a.step(y, self.p), b.step(y, self.p))
        snapshot = a.tracker.model.mapping.copy()
        b.step(self.frame()*100, self.p)
        np.testing.assert_array_equal(a.tracker.model.mapping, snapshot)

    def test_invalid_inputs_do_not_consume_slot(self):
        for gate in (True, 0., -1., float('nan'), 1.1):
            with self.assertRaises(ValueError):
                SlotTrackedLMMSE(self.layout, gate=gate)
        r = SlotTrackedLMMSE(self.layout)
        with self.assertRaises(ValueError):
            r.step(np.zeros((4, 64)), self.p)
        with self.assertRaises(ValueError):
            r.step(None, self.p)
        self.assertEqual(r.slots, 0)

    def test_gate_selection_rules_and_failure_are_explicit(self):
        rows = []
        for scenario in ('stationary', 'alias64'):
            for method, loss in [('capture_adaptive', .1), ('gate_0.15', .11), ('gate_0.25', .1), ('gate_0.35', .1)]:
                rows.append(dict(scenario=scenario, ebn0_db=18, receiver=method, window='all', loss=loss))
        chosen, candidates, eligible = select(dict(summary=rows), [.15, .25, .35])
        self.assertEqual(chosen, .35)
        self.assertTrue(eligible)
        self.assertFalse(candidates[0]['eligible'])
        for row in rows:
            if row['receiver'] != 'capture_adaptive':
                row['loss'] = .2
        self.assertFalse(select(dict(summary=rows), [.15, .25, .35])[2])

    def test_config_frozen_and_seed_pools_disjoint(self):
        cfg = json.loads(Path('experiments/captured_tracking.json').read_text())
        validate(cfg)
        other = copy.deepcopy(cfg)
        other['development']['seeds'][0] = 271001
        with self.assertRaises(ValueError):
            validate(other)


if __name__ == '__main__':
    unittest.main()
