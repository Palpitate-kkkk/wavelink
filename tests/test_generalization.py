import copy
import json
import unittest
from pathlib import Path
import numpy as np
from experiments.generalization_ber import frozen_protocol, validate, timeline, update_response, subspace_floor, evaluate
from experiments.drift_ber import episode
from wavelink.estimation import TapDomainLS, ensemble_nmse
from wavelink.receiver import comb_grid


class GeneralizationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('experiments/generalization.json').read_text())

    def test_original_policy_is_locked_and_new_seeds_are_disjoint(self):
        parent,lock=validate(self.cfg)
        self.assertEqual(lock['threshold'],.75)
        self.assertTrue(set(self.cfg['seeds']).isdisjoint(parent['development']['seeds']+parent['test']['seeds']))
        bad=copy.deepcopy(self.cfg);bad['source_selection_sha256']='0'*64
        with self.assertRaises(ValueError):frozen_protocol(bad)
        bad=copy.deepcopy(self.cfg);bad['seeds'][0]=parent['test']['seeds'][0]
        with self.assertRaises(ValueError):validate(bad)

    def test_piecewise_statistics_and_repeated_returns(self):
        spec=next(s for s in self.cfg['scenarios'] if s['name']=='repeat24')
        p,n=timeline(self.cfg,spec,71)
        np.testing.assert_array_equal(p[:71],np.tile(self.cfg['states']['initial'],(71,1)))
        np.testing.assert_array_equal(p[71],self.cfg['states']['seen_final'])
        np.testing.assert_array_equal(p[95],self.cfg['states']['initial'])
        self.assertEqual(n[70],1);self.assertEqual(n[71],4);self.assertEqual(n[95],1)
        np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-15)
        outside=next(s for s in self.cfg['scenarios'] if s['name']=='outside_support')
        p,_=timeline(self.cfg,outside,71)
        self.assertEqual(np.flatnonzero(p[71])[-1],15)
        self.assertLessEqual(15,self.cfg['cp_len'])
        self.assertGreater(15,self.cfg['assumed_length']-1)

    def test_change_responses_cannot_reuse_updates_or_count_next_state(self):
        times=[15,23,30,60]
        self.assertEqual(update_response(times,10,24)['first_update_delay'],6)
        self.assertEqual(update_response(times,24,48)['first_update_delay'],7)
        # Update after frame 23 takes effect at 24, outside interval [0,24).
        self.assertEqual(update_response([23],0,24)['no_effective_update'],1)
        self.assertEqual(update_response([],0,96)['missed_within_64'],1)
        self.assertEqual(update_response([70],0,96)['missed_within_64'],1)
        self.assertEqual(update_response([70],0,96)['no_effective_update'],0)

    def test_oracle_subspace_floor_and_supported_states(self):
        layout=comb_grid(4);m=TapDomainLS(layout,13)
        for name in ['initial','seen_final','new_paths']:
            self.assertLess(subspace_floor(layout,13,self.cfg['states'][name]),1e-24)
        p=self.cfg['states']['outside_support']
        floor=subspace_floor(layout,13,p)
        self.assertGreater(floor,.01)
        self.assertGreaterEqual(sum(ensemble_nmse(m.mapping,layout,p,0))+1e-12,floor)
        fd=m.data_design
        true=np.exp(-2j*np.pi*np.outer(layout.data_tones,np.arange(16))/64)
        residual=true-fd@(np.linalg.pinv(fd)@true)
        np.testing.assert_allclose(fd.conj().T@residual,0,atol=1e-12)

    def test_explicit_statistics_match_existing_step_and_validate_inputs(self):
        cfg=json.loads(Path('experiments/drift.json').read_text())
        cfg.update(calibration_frames=4,episode_frames=24,switch_after_frames=8,transient_frames=8,symbols_per_frame=4,trace_bin_frames=4)
        cfg['policies']=[dict(name='frozen',window=4,update_every=None)]
        spec=cfg['scenarios'][-1]
        old=episode(cfg,spec,6,111001,0)[0]
        from experiments.drift_ber import validate as original_validate
        initial,changed=original_validate(cfg)
        p=np.tile(initial,(24,1));p[8:]=changed
        n=np.r_[np.ones(8),np.full(16,4.)]
        explicit=episode(cfg,spec,6,111001,0,statistics=(p,n))[0]
        for name in old:np.testing.assert_array_equal(old[name],explicit[name])
        with self.assertRaises(ValueError):episode(cfg,spec,6,111001,0,statistics=(2*p,n))
        with self.assertRaises(ValueError):episode(cfg,spec,6,111001,0,statistics=(p,-n))
        with self.assertRaises(ValueError):episode(cfg,spec,6,111001,0,np.zeros(24),(p,n))

    def test_long_true_channel_preserves_noiseless_ofdm_model(self):
        from experiments.learned_ber import simulate_frame
        from wavelink import qpsk_modulate, qpsk_demodulate
        from wavelink.receiver import zero_force
        layout=comb_grid(4)
        powers=np.asarray(self.cfg['states']['outside_support'])
        y,p,bits,truth=simulate_frame(np.random.default_rng(115731),powers,0.,layout,16,8)
        data=qpsk_modulate(bits.ravel()).reshape(len(y),-1)
        yd=y[:,layout.bins(layout.data_tones)]
        np.testing.assert_allclose(yd,data*truth,atol=1e-12)
        decoded=qpsk_demodulate(zero_force(yd,truth).ravel())
        np.testing.assert_array_equal(decoded,bits.ravel())

    def test_fixed_scenario_shapes_and_schedule_bounds(self):
        validate(self.cfg)
        for spec in self.cfg['scenarios']:
            p,n=timeline(self.cfg,spec,95)
            self.assertEqual(p.shape,(448,16));self.assertEqual(n.shape,(448,))
            self.assertTrue(np.all(n>0))
        bad=copy.deepcopy(self.cfg);bad['states']['outside_support'][-1]=.8
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(self.cfg);bad['scenarios'][-1]['changes'][1]['offset']=0
        with self.assertRaises(ValueError):validate(bad)


if __name__=='__main__':unittest.main()
