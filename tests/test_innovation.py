import copy
import json
from pathlib import Path
import unittest
from collections import deque
from unittest.mock import patch
import numpy as np
from experiments.density_ber import coupled_frame
from experiments.innovation_ber import validate, episode, evaluate, select
from wavelink.innovation import InnovationTriggeredLMMSE
from wavelink.tracking import TriggeredPilotLMMSE
from wavelink.receiver import comb_grid


class InnovationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('experiments/innovation.json').read_text());self.layout=comb_grid(2)
        self.noise=.02;self.symbols=4;self.cal=[]
        for t in range(64):
            y,p,_,_=coupled_frame(171001,0,1,t,self.cfg['states']['initial'],self.noise,self.layout,16,self.symbols)
            self.cal.append(y[:,self.layout.bins(self.layout.pilot_tones)]/p)
        self.cal=np.asarray(self.cal)

    def frame(self,t,state='initial'):
        return coupled_frame(171001,0,2,t,self.cfg['states'][state],self.noise,self.layout,16,self.symbols)[:2]

    def test_probe_covariance_algebra(self):
        g=InnovationTriggeredLMMSE(self.cal,self.layout,17)
        fp=g.model.pilot_design;p=g.statistics.powers;c=(fp*p)@fp.conj().T;lam=g.statistics.noise_variance/self.symbols
        b=g.innovation_matrix
        np.testing.assert_allclose(b@(c+lam*np.eye(28)),lam*np.eye(28),atol=1e-13)
        expected=float(np.trace(b@(c+lam*np.eye(28))@b.conj().T).real)
        self.assertAlmostEqual(expected,g.innovation_signal+lam*g.innovation_noise_gain,places=12)

    def test_modeled_noise_normalization_and_new_path_sensitivity(self):
        g=InnovationTriggeredLMMSE(self.cal,self.layout,17);rng=np.random.default_rng(18881)
        fp=g.model.pilot_design;p=g.statistics.powers;n=g.statistics.noise_variance
        h=(rng.normal(size=(4096,17))+1j*rng.normal(size=(4096,17)))*np.sqrt(p/2)
        noise=(rng.normal(size=(4096,28))+1j*rng.normal(size=(4096,28)))*np.sqrt(n/(2*self.symbols))
        z=h@fp.T+noise;g.pilot_means=deque(z,maxlen=4096)
        background=g._innovation_score(n)
        self.assertLess(abs(background-1),.04)
        new=(rng.normal(size=4096)+1j*rng.normal(size=4096))*np.sqrt(.05/2)
        g.pilot_means=deque(z+new[:,None]*fp[:,15],maxlen=4096)
        self.assertGreater(g._innovation_score(n),1.5*background)

    def test_disabled_innovation_is_exact_legacy_ablation(self):
        a=InnovationTriggeredLMMSE(self.cal,self.layout,17,innovation_threshold=1e20)
        b=TriggeredPilotLMMSE(self.cal,self.layout,17,threshold=.75)
        for t in range(80):
            y,p=self.frame(t,'initial' if t<16 else 'weak15')
            np.testing.assert_array_equal(a.estimate(y,p),b.estimate(y,p))
            self.assertEqual(a.advance(),b.advance())
            np.testing.assert_array_equal(a.model.mapping,b.model.mapping)
        self.assertEqual(a.update_count,b.update_count)

    def test_shared_cooldown_and_one_effective_rebuild(self):
        g=InnovationTriggeredLMMSE(self.cal,self.layout,17,innovation_threshold=1e-5)
        for t in range(96):
            y,p=self.frame(t);mapping=g.model.mapping.copy()
            expected=g.model.estimate(y,p);np.testing.assert_array_equal(g.estimate(y,p),expected)
            np.testing.assert_array_equal(g.model.mapping,mapping)
            g.advance()
        times=[e['after_frame'] for e in g.trigger_log]
        self.assertEqual(times,[7,39,71]);self.assertEqual(g.update_count,len(times))
        self.assertTrue(any(e['event_kind']=='innovation' for e in g.trigger_log))
        self.assertTrue(all(e['after_frame']==t for e,t in zip(g.trigger_log,times)))
        self.assertEqual(g.innovation_checks,12)

    def test_cached_pilot_means_are_copied_and_bounded(self):
        g=InnovationTriggeredLMMSE(self.cal,self.layout,17)
        y,p=self.frame(0);g.estimate(y,p);expected=g._pending.mean(axis=0).copy();y[:]=1e9;g.advance()
        np.testing.assert_array_equal(g.pilot_means[-1],expected)
        self.assertEqual(len(g.pilot_means),16);self.assertEqual(g.feature_frames,82)
        self.assertEqual(g.fit_frame_visits,0)

    def test_future_suffix_cannot_change_prefix(self):
        a=InnovationTriggeredLMMSE(self.cal,self.layout,17);b=InnovationTriggeredLMMSE(self.cal.copy(),self.layout,17)
        for t in range(32):
            y,p=self.frame(t);other=y.copy()
            if t>=24:other[:]=100
            left=a.estimate(y,p);right=b.estimate(other,p)
            if t<24:np.testing.assert_array_equal(left,right)
            a.advance();b.advance()

    def test_state_and_threshold_validation(self):
        for v in [True,0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):InnovationTriggeredLMMSE(self.cal,self.layout,17,innovation_threshold=v)
        g=InnovationTriggeredLMMSE(self.cal,self.layout,17)
        with self.assertRaises(RuntimeError):g.advance()
        y,p=self.frame(0);g.estimate(y,p)
        with self.assertRaises(RuntimeError):g.estimate(y,p)

    def test_baseline_and_seed_isolation(self):
        validate(self.cfg)
        for seed in [81001,91001,111001,121001,131001,141001,151001,161001]:
            bad=copy.deepcopy(self.cfg);bad['test']['seeds'][0]=seed
            with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(self.cfg);bad['test']['seeds'][0]=bad['development']['seeds'][0]
        with self.assertRaises(ValueError):validate(bad)

    def test_development_gate_and_explicit_failure_fallback(self):
        cfg=self.cfg;rows=[];costs=[]
        for s in cfg['scenarios']:
            for db in cfg['ebn0_db']:
                for name,value in [('legacy17',1.),('innovation_2',.5),('innovation_4',.5)]:
                    rows.append(dict(scenario=s['name'],ebn0_db=db,receiver=name,phase='post_all',ensemble_nmse=value))
                    costs.append(dict(scenario=s['name'],receiver=name,updates=0))
        results=dict(summary=rows,costs=costs)
        self.assertEqual(select(results,cfg)[0],2.);self.assertTrue(select(results,cfg)[2])
        for r in rows:
            if r['scenario']=='weak15' and r['receiver']=='innovation_2':r['ensemble_nmse']=.9
        self.assertEqual(select(results,cfg)[0],4.)
        for r in rows:
            if r['scenario']=='weak15' and r['receiver']=='innovation_4':r['ensemble_nmse']=.9
        self.assertFalse(select(results,cfg)[2]);self.assertEqual(select(results,cfg)[0],4.)

    def test_small_episode_and_causal_events(self):
        cfg=copy.deepcopy(self.cfg);cfg.update(episode_frames=16,symbols_per_frame=2)
        values,events,costs=episode(cfg,cfg['scenarios'][0],6,0,171001,4,[2.])
        self.assertEqual(values['innovation_2'].shape,(16,5))
        for v in values.values():self.assertTrue(np.all(np.isfinite(v)))
        self.assertTrue(all(e['applies_from_frame']==e['after_frame']+1<16 for e in events))
        self.assertTrue(all(c['last_update_after_frame']<15 for c in costs))

    def test_evaluation_schema_and_native_json_counters(self):
        cfg=copy.deepcopy(self.cfg);cfg['scenarios']=cfg['scenarios'][:1]
        def fake(cfg,spec,db,si,seed,start,thresholds):
            names=['oracle_zf','matched','legacy17','recovery13','periodic17']+[f'innovation_{v:g}' for v in thresholds]
            return {n:np.zeros((320,5)) for n in names},[],[]
        with patch('experiments.innovation_ber.episode',side_effect=fake):r=evaluate(cfg,'development',[2.])
        self.assertEqual(len(r['runs']),8*2*6*4)
        self.assertEqual(set(x['phase'] for x in r['runs']),{'pre','post_early','post_late','post_all'})
        json.dumps(dict(oracle_pass=sum(x['oracle_consistent'] for x in r['summary'] if x['receiver']=='oracle_zf')))


if __name__=='__main__':unittest.main()
