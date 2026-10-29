import copy
import json
import unittest
from pathlib import Path
import numpy as np
from wavelink.tracking import CausalPilotLMMSE
from wavelink.estimation import TapDomainLS
from wavelink.receiver import comb_grid
from experiments.drift_ber import episode, evaluate, validate


class CausalTrackingTests(unittest.TestCase):
    def setUp(self):
        self.layout=comb_grid(4)
        rng=np.random.default_rng(573)
        fp=TapDomainLS(self.layout,13).pilot_design
        h=(rng.normal(size=(8,13))+1j*rng.normal(size=(8,13)))/np.sqrt(26)
        self.z=(h@fp.T)[:,None,:]+(rng.normal(size=(8,4,len(fp)))+1j*rng.normal(size=(8,4,len(fp))))*.1
        self.p=np.ones((4,len(fp)),complex)
        self.y=np.zeros((4,64),complex);self.y[:,self.layout.bins(self.layout.pilot_tones)]=self.z[0]

    def test_estimate_then_advance_and_no_current_frame_fit(self):
        tracker=CausalPilotLMMSE(self.z,self.layout,13,4,1)
        initial=tracker.model.mapping.copy()
        expected=tracker.model.estimate(self.y,self.p)
        np.testing.assert_array_equal(tracker.estimate(self.y,self.p),expected)
        np.testing.assert_array_equal(tracker.model.mapping,initial)
        self.assertEqual(tracker.last_update_after_frame,-1)
        with self.assertRaises(RuntimeError):tracker.estimate(self.y,self.p)
        tracker.advance()
        self.assertEqual(tracker.frames_observed,1)
        self.assertEqual(tracker.last_update_after_frame,0)
        self.assertEqual(tracker.update_count,1)
        with self.assertRaises(RuntimeError):tracker.advance()
        self.assertFalse(np.array_equal(tracker.model.mapping,initial))

    def test_common_initial_fit_and_frozen_mapping(self):
        a=CausalPilotLMMSE(self.z,self.layout,13,4,1)
        b=CausalPilotLMMSE(self.z,self.layout,13,8,None)
        np.testing.assert_array_equal(a.model.mapping,b.model.mapping)
        initial=b.model.mapping.copy()
        for _ in range(5):
            b.estimate(self.y,self.p);self.assertFalse(b.advance())
        np.testing.assert_array_equal(b.model.mapping,initial)
        self.assertEqual(b.update_count,0)

    def test_schedule_memory_bound_and_copying(self):
        tracker=CausalPilotLMMSE(self.z,self.layout,13,4,3)
        for t in range(7):
            y=self.y.copy();tracker.estimate(y,self.p)
            saved=tracker._pending.copy();y[:]=1e5
            np.testing.assert_array_equal(saved,tracker._pending)
            updated=tracker.advance()
            self.assertEqual(updated,(t+1)%3==0)
            self.assertLessEqual(len(tracker.history),4)
        self.assertEqual(tracker.update_count,2)
        self.assertEqual(tracker.fit_frame_visits,8)
        self.assertEqual(tracker.last_update_after_frame,5)

    def test_future_suffix_cannot_change_prefix(self):
        a=CausalPilotLMMSE(self.z,self.layout,13,4,1)
        b=CausalPilotLMMSE(self.z.copy(),self.layout,13,4,1)
        rng=np.random.default_rng(41)
        frames=[self.y+rng.normal(size=self.y.shape)*.1 for _ in range(6)]
        altered=[x.copy() for x in frames];altered[4:]=[x+100 for x in altered[4:]]
        for t,(x,y) in enumerate(zip(frames,altered)):
            left=a.estimate(x,self.p);right=b.estimate(y,self.p)
            if t<4:np.testing.assert_array_equal(left,right)
            a.advance();b.advance()
        self.assertFalse(np.array_equal(a.model.mapping,b.model.mapping))

    def test_invalid_parameters_and_state(self):
        for window,stride in [(1,1),(9,1),(True,1),(4,0),(4,1.5)]:
            with self.assertRaises(ValueError):CausalPilotLMMSE(self.z,self.layout,13,window,stride)
        tracker=CausalPilotLMMSE(self.z,self.layout,13,8)
        with self.assertRaises(RuntimeError):tracker.advance()

    def test_episode_controls_and_reproducibility(self):
        cfg=json.loads(Path('experiments/drift.json').read_text())
        cfg.update(calibration_frames=8,episode_frames=16,switch_after_frames=8,transient_frames=4,symbols_per_frame=4,trace_bin_frames=4)
        cfg['policies']=[dict(name='frozen',window=8,update_every=None),dict(name='rolling',window=4,update_every=1)]
        stationary=cfg['scenarios'][0];changed=cfg['scenarios'][-1]
        a=episode(cfg,stationary,18,61001,0)
        b=episode(cfg,changed,18,61001,0)
        c=episode(copy.deepcopy(cfg),stationary,18,61001,0)
        for key in a[0]:
            np.testing.assert_array_equal(a[0][key][:8],b[0][key][:8])
            np.testing.assert_array_equal(a[0][key],c[0][key])
        self.assertFalse(np.array_equal(a[0]['frozen'][8:],b[0]['frozen'][8:]))
        for cost in a[1]:self.assertLess(cost['maximum_training_lead'],0)
        self.assertEqual(a[1][1]['updates'],15)
        cfg.update(seeds=[61001,61002],ebn0_db=[18],scenarios=[changed])
        runs,summary,trace,costs,events=evaluate(cfg)
        for row in summary:
            self.assertEqual(row['episodes'],2)
            if row['receiver']=='frozen':
                self.assertEqual(row['paired_ber_minus_frozen'],0)
                self.assertEqual(row['paired_nmse_minus_frozen'],0)
        self.assertTrue(events)
        cfg['cp_len']=2
        with self.assertRaises(ValueError):validate(cfg)


if __name__=='__main__':unittest.main()
