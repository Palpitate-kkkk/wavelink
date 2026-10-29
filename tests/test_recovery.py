import copy
import json
from pathlib import Path
import unittest
import numpy as np
from wavelink.guard import SupportGuard
from wavelink.recovery import RecoveringSupportGuard
from wavelink.estimation import TapDomainLS
from wavelink.receiver import comb_grid
from wavelink.statistics import learn_pilot_statistics
from experiments.recovery_ber import validate, audit_modes, select


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(773901);self.layout=comb_grid(4)
        self.fp=TapDomainLS(self.layout,13).pilot_design
        h=(self.rng.normal(size=(64,13))+1j*self.rng.normal(size=(64,13)))/np.sqrt(26)
        self.z=(h@self.fp.T)[:,None,:]+self.noise((64,20,16))
        self.p=np.ones((20,16),complex)

    def noise(self,shape):return .1*(self.rng.normal(size=shape)+1j*self.rng.normal(size=shape))

    def frame(self,outside=False):
        h=np.exp(-2j*np.pi*np.asarray(self.layout.pilot_tones)*15/64) if outside else self.fp[:,0]
        y=np.zeros((20,64),complex)
        y[:,self.layout.bins(self.layout.pilot_tones)]=h[None,:]+self.noise((20,16))
        return y

    def step(self,g,y):
        previous=g.model.mapping.copy();expected=g.model.estimate(y,self.p)
        np.testing.assert_array_equal(g.estimate(y,self.p),expected)
        np.testing.assert_array_equal(g.model.mapping,previous)
        return g.advance()

    def test_release_is_causal_and_refits_observed_history(self):
        g=RecoveringSupportGuard(self.z,self.layout,13,16,release_checks=2)
        for _ in range(8):self.step(g,self.frame(True))
        self.assertTrue(g.alarmed)
        for _ in range(40):
            old_count=len(g.release_log);self.step(g,self.frame())
            if len(g.release_log)>old_count:
                fit=learn_pilot_statistics(np.asarray(g.history),self.layout,13)
                np.testing.assert_array_equal(g.statistics.powers,fit.powers)
                self.assertEqual(g.primary.frames_observed,0)
                self.assertEqual(g.last_update_after_frame,g.frames_observed-1)
                self.assertEqual(g.fit_frame_visits,64)
                break
        else:self.fail('No release after supported history')
        self.assertFalse(g.alarmed);self.assertEqual(len(g.model.powers),13)
        self.step(g,self.frame());self.assertEqual(g.primary.frames_observed,1)
        self.assertEqual(len(g.history),64)

    def test_persistent_outside_does_not_release(self):
        g=RecoveringSupportGuard(self.z,self.layout,13,16,release_checks=2)
        for _ in range(96):self.step(g,self.frame(True))
        self.assertTrue(g.alarmed);self.assertEqual(len(g.release_log),0)
        self.assertEqual(len(g.warning_log),1)

    def test_reentry_after_release_and_effective_event_counts(self):
        g=RecoveringSupportGuard(self.z,self.layout,13,16,release_checks=2)
        for _ in range(8):self.step(g,self.frame(True))
        for _ in range(64):self.step(g,self.frame())
        self.assertFalse(g.alarmed)
        for _ in range(16):self.step(g,self.frame(True))
        self.assertTrue(g.alarmed);self.assertEqual(len(g.warning_log),2)
        self.assertEqual(len(g.release_log),1)
        self.assertEqual(g.update_count,len(g.trigger_log))
        self.assertEqual(len({e['after_frame'] for e in g.trigger_log}),len(g.trigger_log))

    def test_low_checks_reset_when_residual_rises(self):
        g=RecoveringSupportGuard(self.z,self.layout,13,16,release_checks=8)
        for _ in range(8):self.step(g,self.frame(True))
        for _ in range(24):self.step(g,self.frame())
        self.assertGreater(g.low_checks,0);self.assertLess(g.low_checks,8)
        for _ in range(8):self.step(g,self.frame(True))
        self.assertEqual(g.low_checks,0);self.assertTrue(g.alarmed)

    def test_without_release_mapping_matches_frozen_latched_guard_exactly(self):
        a=RecoveringSupportGuard(self.z,self.layout,13,16,release_checks=8,exit_threshold=.001)
        b=SupportGuard(self.z,self.layout,13,16,threshold=4.)
        for t in range(80):
            y=self.frame(16<=t<40)
            np.testing.assert_array_equal(a.estimate(y,self.p),b.estimate(y,self.p))
            self.assertEqual(a.advance(),b.advance())
            np.testing.assert_array_equal(a.model.mapping,b.model.mapping)
        self.assertEqual(len(a.release_log),0)

    def test_future_suffix_cannot_affect_prefix(self):
        a=RecoveringSupportGuard(self.z,self.layout,13,16,release_checks=2)
        b=RecoveringSupportGuard(self.z.copy(),self.layout,13,16,release_checks=2)
        for t in range(48):
            y=self.frame(t<8);other=y.copy()
            if t>=40:other=self.frame(True)
            left=a.estimate(y,self.p);right=b.estimate(other,self.p)
            if t<40:np.testing.assert_array_equal(left,right)
            a.advance();b.advance()

    def test_invalid_confirmation_and_hysteresis(self):
        for kw in [dict(release_checks=0),dict(release_checks=True),dict(release_checks=2.5),dict(exit_threshold=4.),dict(exit_threshold=0),dict(exit_threshold=float('nan'))]:
            with self.assertRaises(ValueError):RecoveringSupportGuard(self.z,self.layout,13,16,**kw)
        g=RecoveringSupportGuard(self.z,self.layout,13,16)
        with self.assertRaises(RuntimeError):g.advance()
        g.estimate(self.frame(),self.p)
        with self.assertRaises(RuntimeError):g.estimate(self.frame(),self.p)

    def test_mode_audit_applies_changes_only_on_next_frame(self):
        logs=[dict(after_frame=1,event_kind='alarm'),dict(after_frame=3,event_kind='release'),dict(after_frame=5,event_kind='alarm')]
        outside=np.array([False,True,True,False,False,True,True])
        active,ts=audit_modes(logs,outside)
        np.testing.assert_array_equal(active,[False,False,True,True,False,False,True])
        self.assertEqual(ts[1]['release_while_outside'],0)
        self.assertEqual(ts[1]['applies_from_frame'],4)
        _,ts=audit_modes([dict(after_frame=2,event_kind='release')],outside)
        self.assertEqual(ts[0]['release_while_outside'],1)

    def test_baseline_hash_and_seed_isolation(self):
        cfg=json.loads(Path('experiments/recovery.json').read_text());validate(cfg)
        old=json.loads(Path('experiments/guard.json').read_text())
        bad=copy.deepcopy(cfg);bad['test']['seeds'][0]=old['test']['seeds'][0]
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(cfg);bad['baseline_metadata_sha256']='0'*64
        with self.assertRaises(ValueError):validate(bad)

    def test_selection_uses_development_gates_and_retains_failure(self):
        cfg=json.loads(Path('experiments/recovery.json').read_text());summary=[];audit=[]
        for s in cfg['scenarios']:
            for db in cfg['ebn0_db']:
                for phase in ('post_all','post_late'):
                    for name,nmse in [('latched',1.),('recovery_4',.25),('recovery_8',.25)]:
                        summary.append(dict(scenario=s['name'],ebn0_db=db,phase=phase,receiver=name,ensemble_nmse=nmse))
                for k in cfg['release_checks']:audit.append(dict(scenario=s['name'],receiver=f'recovery_{k}',supported_only=int(s['name']=='stationary'),alarms=0,release_while_outside=0))
        data=dict(summary=summary,mode_audit=audit)
        self.assertEqual(select(data,cfg)[0],4);self.assertTrue(select(data,cfg)[2])
        for r in audit:
            if r['receiver']=='recovery_4' and r['scenario']=='persistent15':r['release_while_outside']=1
        self.assertEqual(select(data,cfg)[0],8)
        for r in audit:r['alarms']=1
        self.assertFalse(select(data,cfg)[2]);self.assertEqual(select(data,cfg)[0],8)


if __name__=='__main__':unittest.main()
