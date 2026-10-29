import copy
import json
from pathlib import Path
import unittest
import numpy as np
from wavelink.guard import SupportGuard
from wavelink.tracking import TriggeredPilotLMMSE, make_tracker
from wavelink.estimation import TapDomainLS, TapDomainLMMSE
from wavelink.receiver import comb_grid
from experiments.guard_ber import validate, select
from experiments.drift_ber import episode


class SupportGuardTests(unittest.TestCase):
    def setUp(self):
        self.layout=comb_grid(4);self.rng=np.random.default_rng(73151)
        self.fp=TapDomainLS(self.layout,13).pilot_design
        h=(self.rng.normal(size=(64,13))+1j*self.rng.normal(size=(64,13)))/np.sqrt(26)
        self.z=(h@self.fp.T)[:,None,:]+self.noise((64,20,16))
        self.p=np.ones((20,16),complex)

    def noise(self,shape):return .1*(self.rng.normal(size=shape)+1j*self.rng.normal(size=shape))

    def received(self,z):
        y=np.zeros((20,64),complex);y[:,self.layout.bins(self.layout.pilot_tones)]=z
        return y

    def outside(self):
        f=np.exp(-2j*np.pi*np.asarray(self.layout.pilot_tones)*15/64)
        return self.received(f[None,:]+self.noise((20,16)))

    def test_residual_annihilates_supported_channel_and_detects_delay15(self):
        g=SupportGuard(self.z,self.layout,13,16)
        np.testing.assert_allclose(g.residual_basis.conj().T@self.fp,0,atol=2e-14)
        e,n=g._feature(np.tile(self.fp@np.ones(13),(20,1)))
        self.assertLess(e,1e-24);self.assertEqual(n,0)
        z=self.outside()[:,self.layout.bins(self.layout.pilot_tones)]
        self.assertGreater(g._feature(z)[0],.1)

    def test_supported_noise_normalization(self):
        g=SupportGuard(self.z,self.layout,13,16)
        moments=np.array([g._feature(self.z[i%64]+self.noise((20,16))) for i in range(512)])
        ratio=moments[:,0].mean()/(3*moments[:,1].mean()/20)
        self.assertLess(abs(ratio-1),.15)

    def test_alarm_is_next_frame_only_and_latched_with_cp_prior(self):
        g=SupportGuard(self.z,self.layout,13,16,threshold=2.)
        with self.assertRaises(RuntimeError):g.advance()
        for t in range(8):
            y=self.outside();mapping=g.model.mapping.copy()
            expected=g.model.estimate(y,self.p)
            np.testing.assert_array_equal(g.estimate(y,self.p),expected)
            np.testing.assert_array_equal(g.model.mapping,mapping)
            with self.assertRaises(RuntimeError):g.estimate(y,self.p)
            g.advance()
            if t<7:self.assertFalse(g.alarmed)
        self.assertTrue(g.alarmed);self.assertEqual(g.alarm_after_frame,7)
        self.assertEqual(g.last_update_after_frame,7);self.assertEqual(len(g.model.powers),17)
        np.testing.assert_allclose(g.model.powers,1/17)
        for _ in range(40):g.estimate(self.received(self.z[0]),self.p);g.advance()
        self.assertTrue(g.alarmed);self.assertEqual(len(g.warning_log),1)
        self.assertEqual(len(g.history),64)
        self.assertTrue(all(x['after_frame']<g.frames_observed for x in g.trigger_log))
        self.assertEqual(g.update_count,len(g.trigger_log))

    def test_no_alarm_matches_original_trigger_exactly(self):
        g=SupportGuard(self.z,self.layout,13,16,threshold=1e20)
        a=TriggeredPilotLMMSE(self.z,self.layout,13,threshold=.75)
        for t in range(48):
            y=self.received(self.z[t])
            np.testing.assert_array_equal(g.estimate(y,self.p),a.estimate(y,self.p))
            self.assertEqual(g.advance(),a.advance())
            np.testing.assert_array_equal(g.model.mapping,a.model.mapping)
        self.assertFalse(g.alarmed);self.assertEqual(g.update_count,a.update_count)

    def test_future_suffix_cannot_change_prefix_and_input_is_copied(self):
        a=SupportGuard(self.z,self.layout,13,16);b=SupportGuard(self.z,self.layout,13,16)
        for t in range(16):
            y=self.outside();other=y.copy()
            if t>=12:other[:]=0
            left=a.estimate(y,self.p);right=b.estimate(other,self.p)
            saved=a._pending.copy();y[:]=1e6
            np.testing.assert_array_equal(a._pending,saved)
            if t<12:np.testing.assert_array_equal(left,right)
            a.advance();b.advance()

    def test_pilot_nullspace_is_an_explicit_detection_limit(self):
        # 17 taps, 16 pilots: at least one nonzero channel is invisible to
        # this grid. A guard cannot certify all possible CP-covered channels.
        fp=np.exp(-2j*np.pi*np.outer(self.layout.pilot_tones,np.arange(17))/64)
        _,_,vh=np.linalg.svd(fp,full_matrices=True)
        h=vh[-1].conj();np.testing.assert_allclose(fp@h,0,atol=2e-14)
        fd=np.exp(-2j*np.pi*np.outer(self.layout.data_tones,np.arange(17))/64)
        self.assertGreater(np.linalg.norm(fd@h),.1)

    def test_regularized_fallback_is_finite_when_pilot_problem_is_underdetermined(self):
        model=TapDomainLMMSE(self.layout,np.ones(17)/17,.02)
        self.assertTrue(np.all(np.isfinite(model.mapping)))
        fp=model.pilot_design;fd=model.data_design
        expected=(fd@fp.conj().T)@np.linalg.inv(fp@fp.conj().T+17*.02*np.eye(16))
        np.testing.assert_allclose(model.mapping,expected,rtol=1e-10,atol=1e-10)

    def test_parameters_and_state_validation(self):
        for kwargs in [dict(threshold=0),dict(threshold=float('nan')),dict(check_every=True),dict(monitor_window=65),dict(refresh_every=0),dict(fallback_power=-1)]:
            with self.assertRaises(ValueError):SupportGuard(self.z,self.layout,13,16,**kwargs)
        with self.assertRaises(ValueError):SupportGuard(self.z,self.layout,16,16)
        with self.assertRaises(ValueError):SupportGuard(self.z,self.layout,13,12)
        g=SupportGuard(self.z,self.layout,13,16)
        with self.assertRaises(ValueError):g.estimate(np.zeros((2,64)),np.ones((2,16)))

    def test_new_protocol_and_seed_isolation(self):
        cfg=json.loads(Path('experiments/guard.json').read_text());validate(cfg)
        bad=copy.deepcopy(cfg);bad['test']['seeds'][0]=bad['development']['seeds'][0]
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(cfg);bad['states']['delay16'][16]=1.
        with self.assertRaises(ValueError):validate(bad)

    def test_selection_rule_and_explicit_failure_fallback(self):
        cfg=json.loads(Path('experiments/guard.json').read_text())
        rows=[];detection=[]
        for spec in cfg['scenarios']:
            outside=any(np.any(np.asarray(cfg['states'][c['state']])[13:]>0) for c in spec['changes'])
            for db in cfg['ebn0_db']:
                rows.append(dict(scenario=spec['name'],ebn0_db=db,phase='post_all',receiver='triggered',ensemble_nmse=1.))
                for t in cfg['thresholds']:
                    rows.append(dict(scenario=spec['name'],ebn0_db=db,phase='post_all',receiver=f'guard_{t:g}',ensemble_nmse=.2 if outside else 1.))
                    detection.append(dict(receiver=f'guard_{t:g}',supported_only=int(not outside),any_alarm=int(outside)))
        data=dict(summary=rows,detection=detection)
        self.assertEqual(select(data,cfg)[0],2.)
        self.assertTrue(select(data,cfg)[2])
        for d in detection:d['any_alarm']=1
        self.assertEqual(select(data,cfg)[0],4.)
        self.assertFalse(select(data,cfg)[2])

    def test_optional_factory_preserves_old_experiment_exactly(self):
        cfg=json.loads(Path('experiments/drift.json').read_text())
        cfg.update(calibration_frames=4,episode_frames=16,switch_after_frames=4,transient_frames=4,symbols_per_frame=4,trace_bin_frames=4)
        cfg['policies']=[dict(name='frozen',window=4,update_every=None)]
        args=(cfg,cfg['scenarios'][0],6,131001,0)
        old=episode(*args)[0];new=episode(*args,tracker_factory=make_tracker)[0]
        for name in old:np.testing.assert_array_equal(old[name],new[name])


if __name__=='__main__':unittest.main()
