import copy
import json
import unittest
from pathlib import Path
import numpy as np
from wavelink.tracking import TriggeredPilotLMMSE
from wavelink.statistics import learn_pilot_statistics
from wavelink.estimation import TapDomainLS
from wavelink.receiver import comb_grid
from experiments.triggered_ber import validate, policies_for, evaluate_dataset, select_threshold


class TriggeredTests(unittest.TestCase):
    def setUp(self):
        self.layout=comb_grid(4)
        rng=np.random.default_rng(447)
        fp=TapDomainLS(self.layout,13).pilot_design
        h=(rng.normal(size=(16,13))+1j*rng.normal(size=(16,13)))/np.sqrt(26)
        self.z=(h@fp.T)[:,None,:]+(rng.normal(size=(16,4,len(fp)))+1j*rng.normal(size=(16,4,len(fp))))*.1
        self.p=np.ones((4,len(fp)),complex)

    def received(self,z):
        y=np.zeros((len(z),64),complex);y[:,self.layout.bins(self.layout.pilot_tones)]=z
        return y

    def test_cached_moments_match_batch_fit(self):
        tracker=TriggeredPilotLMMSE(self.z[:8],self.layout,13,window=8,monitor_window=4,check_every=1,cooldown=2,threshold=1e6)
        history=list(self.z[:8])
        for z in self.z[8:]:
            tracker.estimate(self.received(z),self.p);tracker.advance();history.append(z)
            actual=tracker._statistics(tracker.fit_moments)
            expected=learn_pilot_statistics(np.asarray(history[-8:]),self.layout,13)
            self.assertAlmostEqual(actual.noise_variance,expected.noise_variance,places=14)
            np.testing.assert_allclose(actual.raw_powers,expected.raw_powers,atol=1e-13)
            np.testing.assert_allclose(actual.powers,expected.powers,atol=1e-13)
        self.assertEqual(tracker.feature_frames,16)
        self.assertEqual(tracker.monitor_checks,8)
        self.assertEqual(tracker.update_count,0)

    def test_trigger_is_after_scoring_and_cooldown_is_enforced(self):
        z=np.ones((8,4,len(self.layout.pilot_tones)),complex)
        z[:,::2]-=.1;z[:,1::2]+=.1
        tracker=TriggeredPilotLMMSE(z,self.layout,13,window=4,monitor_window=2,check_every=1,cooldown=3,threshold=.5)
        initial=tracker.model.mapping.copy()
        for t in range(10):
            current=z[0]+(t+1)*3
            tracker.estimate(self.received(current),self.p)
            if t==0:np.testing.assert_array_equal(tracker.model.mapping,initial)
            tracker.advance()
        self.assertGreaterEqual(tracker.update_count,2)
        times=[r['after_frame'] for r in tracker.trigger_log]
        self.assertEqual(times[0],0)
        self.assertTrue(all(b-a>=3 for a,b in zip(times,times[1:])))
        self.assertEqual(tracker.fit_frame_visits,0)
        self.assertEqual(tracker.feature_frames,18)
        with self.assertRaises(RuntimeError):tracker.advance()

    def test_future_observations_do_not_change_prefix(self):
        a=TriggeredPilotLMMSE(self.z[:8],self.layout,13,window=8,monitor_window=4,check_every=1,cooldown=2,threshold=.5)
        b=TriggeredPilotLMMSE(self.z[:8],self.layout,13,window=8,monitor_window=4,check_every=1,cooldown=2,threshold=.5)
        for i,z in enumerate(self.z[8:]):
            x=a.estimate(self.received(z),self.p)
            y=b.estimate(self.received(z if i<4 else z+10),self.p)
            if i<4:np.testing.assert_array_equal(x,y)
            a.advance();b.advance()
        self.assertNotEqual(a.trigger_log,b.trigger_log)

    def test_invalid_parameters(self):
        for args in [dict(monitor_window=1),dict(monitor_window=9),dict(check_every=0),dict(cooldown=True),dict(threshold=0),dict(threshold=np.nan)]:
            parameters=dict(window=8,monitor_window=4);parameters.update(args)
            with self.assertRaises(ValueError):TriggeredPilotLMMSE(self.z[:8],self.layout,13,**parameters)

    def test_selection_constraints_and_explicit_fallback(self):
        cfg=json.loads(Path('experiments/triggered.json').read_text())
        summary=[];costs=[]
        ratios={.25:1.02,.5:1.03,.75:1.06,1.:1.02}
        updates={.25:2,.5:.5,.75:.1,1.:.3}
        for scenario in ['stationary','step']:
            summary.append(dict(scenario=scenario,ebn0_db=6,phase='post_all',receiver='periodic_64_32',ensemble_nmse=1.))
            for t in cfg['trigger']['threshold_candidates']:
                name=f'trigger_{t:g}'
                summary.append(dict(scenario=scenario,ebn0_db=6,phase='post_all',receiver=name,ensemble_nmse=ratios[t]))
                costs.append(dict(scenario=scenario,receiver=name,updates=updates[t]))
        chosen=select_threshold(cfg,summary,costs)
        self.assertTrue(chosen['development_constraints_met']);self.assertEqual(chosen['threshold'],1.)
        cfg['selection']['max_stationary_mean_updates']=0
        fallback=select_threshold(cfg,summary,costs)
        self.assertFalse(fallback['development_constraints_met'])
        self.assertIn('no feasible',fallback['reason'])

    def test_disjoint_streams_small_pipeline_and_reproducibility(self):
        cfg=json.loads(Path('experiments/triggered.json').read_text())
        cfg.update(phase_cycle=2,onset_min=4,pre_frames=4,early_frames=4,late_start_offset=16,late_frames=4,
                   episode_frames=32,trace_bin_frames=2,calibration_frames=4,symbols_per_frame=4,ebn0_db=[6])
        cfg['development']['seeds']=[81001,81002];cfg['test']['seeds']=[91001,91002]
        cfg['scenarios']=[cfg['scenarios'][0],cfg['scenarios'][3]]
        cfg['policies']=[dict(name='frozen',window=4,update_every=None),dict(name='periodic_64_32',window=4,update_every=2)]
        cfg['trigger'].update(window=4,monitor_window=2,check_every=1,cooldown=2,threshold_candidates=[.5,1.])
        validate(cfg)
        policies=policies_for(cfg)
        a=evaluate_dataset(cfg,cfg['development'],policies)
        b=evaluate_dataset(copy.deepcopy(cfg),cfg['development'],policies)
        for key in ('runs','summary','trace','schedules','detection','events'):self.assertEqual(a[key],b[key])
        chosen=select_threshold(cfg,a['summary'],a['costs'])
        test=evaluate_dataset(cfg,cfg['test'],policies_for(cfg,chosen['threshold']))
        self.assertTrue(all(r['seed'] not in cfg['development']['seeds'] for r in test['runs']))
        self.assertEqual({r['onset_phase'] for r in test['schedules']},{0,1})
        cfg['test']['seeds'][0]=81001
        with self.assertRaises(ValueError):validate(cfg)


if __name__=='__main__':unittest.main()
