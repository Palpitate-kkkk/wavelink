import copy
import csv
import json
import unittest
from pathlib import Path
import numpy as np
from experiments.drift_ber import episode, phases as legacy_phases
from experiments.robustness_ber import onset_schedule, transition_weights, evaluate, validate


class RobustnessTests(unittest.TestCase):
    def test_balanced_schedule_is_reproducible_and_complete(self):
        cfg=json.loads(Path('experiments/robustness.json').read_text())
        a=onset_schedule(cfg)
        self.assertEqual(a,onset_schedule(copy.deepcopy(cfg)))
        self.assertEqual(set(x%32 for x in a.values()),set(range(32)))
        self.assertEqual(set(a.values()),set(range(64,96)))
        self.assertNotEqual(list(a.values()),sorted(a.values()))

    def test_step_and_linear_power_weights(self):
        np.testing.assert_array_equal(transition_weights(8,3,0),[0,0,0,1,1,1,1,1])
        np.testing.assert_array_equal(transition_weights(8,2,4),[0,0,.25,.5,.75,1,1,1])
        self.assertTrue(np.all(np.diff(transition_weights(20,3,10))>=0))
        for start,ramp in [(-1,0),(8,0),(2,-1),(2,7),(True,0)]:
            with self.assertRaises(ValueError):transition_weights(8,start,ramp)

    def test_previous_saved_episode_results_are_preserved(self):
        cfg=json.loads(Path('experiments/drift.json').read_text())
        spec=next(s for s in cfg['scenarios'] if s['name']=='both')
        values,_,_=episode(cfg,spec,6,61001,0)
        with Path('reports/drift/runs.csv').open() as handle:
            stored=[r for r in csv.DictReader(handle)
                    if r['scenario']=='both' and r['ebn0_db']=='6' and r['seed']=='61001']
        self.assertEqual(len(stored),32)
        for row in stored:
            a,b=legacy_phases(cfg)[row['phase']];v=values[row['receiver']][a:b]
            self.assertEqual(int(v[:,3].sum()),int(row['errors']))
            self.assertAlmostEqual(v[:,0].mean(),float(row['ber']),places=15)
            self.assertAlmostEqual(v[:,1].mean(),float(row['ensemble_nmse']),places=15)
            self.assertAlmostEqual(v[:,2].mean(),float(row['predicted_nmse']),places=15)

    def test_explicit_step_matches_legacy_boundary_exactly(self):
        cfg=json.loads(Path('experiments/drift.json').read_text())
        cfg.update(calibration_frames=4,episode_frames=24,switch_after_frames=8,transient_frames=8,symbols_per_frame=4,trace_bin_frames=4)
        cfg['policies']=[dict(name='frozen',window=4,update_every=None),dict(name='periodic_64_32',window=4,update_every=4)]
        spec=cfg['scenarios'][-1]
        original=episode(cfg,spec,6,71001,0)[0]
        explicit=episode(cfg,spec,6,71001,0,transition_weights(24,8,0))[0]
        for name in original:np.testing.assert_array_equal(original[name],explicit[name])
        for weights in [np.zeros(23),np.full(24,np.nan),np.full(24,1.1)]:
            with self.assertRaises(ValueError):episode(cfg,spec,6,71001,0,weights)

    def test_future_ramp_cannot_change_prefix(self):
        cfg=json.loads(Path('experiments/drift.json').read_text())
        cfg.update(calibration_frames=4,episode_frames=24,switch_after_frames=8,transient_frames=8,symbols_per_frame=4,trace_bin_frames=4)
        cfg['policies']=[dict(name='frozen',window=4,update_every=None),dict(name='rolling_16',window=4,update_every=1)]
        spec=cfg['scenarios'][-1]
        step=episode(cfg,spec,6,71001,0,transition_weights(24,8,0))[0]
        ramp=episode(cfg,spec,6,71001,0,transition_weights(24,8,8))[0]
        for name in step:np.testing.assert_array_equal(step[name][:8],ramp[name][:8])
        self.assertFalse(np.array_equal(step['frozen'][8:16],ramp['frozen'][8:16]))

    def test_small_complete_evaluation_and_phase_guards(self):
        cfg=json.loads(Path('experiments/robustness.json').read_text())
        cfg.update(seeds=[71001,71002],phase_cycle=2,onset_min=4,pre_frames=4,early_frames=4,
                   late_start_offset=16,late_frames=4,episode_frames=32,trace_bin_frames=2,
                   calibration_frames=4,symbols_per_frame=4,ebn0_db=[6])
        cfg['policies']=[dict(name='frozen',window=4,update_every=None),dict(name='periodic_64_32',window=4,update_every=2)]
        cfg['scenarios']=[cfg['scenarios'][0],dict(name='ramp4',change_pdp=True,noise_factor=4.,ramp_frames=4)]
        a=evaluate(cfg);b=evaluate(copy.deepcopy(cfg))
        for index in (0,1,2,4):self.assertEqual(a[index],b[index])
        self.assertEqual({r['onset_phase'] for r in a[4]},{0,1})
        for row in a[1]:
            if row['receiver']=='periodic_64_32':
                self.assertEqual(row['paired_ber_minus_periodic'],0)
                self.assertEqual(row['paired_nmse_minus_periodic'],0)
        cfg['episode_frames']=20
        with self.assertRaises(ValueError):validate(cfg)


if __name__=='__main__':unittest.main()
