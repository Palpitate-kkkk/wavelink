import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from experiments.density_ber import validate, effective_db, coupled_frame, geometry, run_episode, evaluate, budget_noise
from wavelink.receiver import comb_grid, pilot_noise_variance
from wavelink.estimation import TapDomainLS
from wavelink import qpsk_modulate


class DensityTests(unittest.TestCase):
    def setUp(self):self.cfg=json.loads(Path('experiments/density.json').read_text())

    def test_common_waveform_bits_and_pilots_are_exactly_coupled(self):
        sparse,dense=comb_grid(4),comb_grid(2)
        for frame in [0,1,19]:
            a=coupled_frame(161001,0,2,frame,self.cfg['states']['delay15'],.02,sparse,16,4)
            b=coupled_frame(161001,0,2,frame,self.cfg['states']['delay15'],.02,dense,16,4)
            np.testing.assert_array_equal(a[0],b[0])
            di=[sparse.data_tones.index(k) for k in dense.data_tones]
            pi=[dense.pilot_tones.index(k) for k in sparse.pilot_tones]
            np.testing.assert_array_equal(a[2].reshape(4,-1,2)[:,di,:],b[2].reshape(4,-1,2))
            np.testing.assert_array_equal(a[1],b[1][:,pi])
            np.testing.assert_array_equal(a[3][di],b[3])

    def test_noiseless_full_length_dense_identification(self):
        layout=comb_grid(2);model=TapDomainLS(layout,17)
        y,p,bits,truth=coupled_frame(161002,1,2,1,self.cfg['states']['delay16'],0.,layout,16,4)
        estimates=model.estimate(y,p)
        np.testing.assert_allclose(estimates,np.tile(truth,(4,1)),atol=1e-12)
        data=qpsk_modulate(bits.ravel()).reshape(4,-1)
        np.testing.assert_allclose(y[:,layout.bins(layout.data_tones)],data*truth,atol=1e-12)
        with self.assertRaises(ValueError):TapDomainLS(comb_grid(4),17)

    def test_geometry_separates_identifiability_and_conditioning(self):
        a,b=geometry(4),geometry(2)
        self.assertEqual(a['full_rank'],16);self.assertEqual(a['full_nullity'],1)
        self.assertFalse(a['full_identifiable']);self.assertEqual(a['full_condition'],'')
        self.assertEqual(b['full_rank'],17);self.assertEqual(b['full_nullity'],0)
        self.assertTrue(b['full_identifiable']);self.assertGreater(b['full_condition'],1.)
        self.assertEqual(a['bits_per_ofdm_symbol'],72);self.assertEqual(b['bits_per_ofdm_symbol'],48)
        self.assertEqual(a['expected_energy_per_symbol'],b['expected_energy_per_symbol'])

    def test_noise_and_eb_budgets_differ_by_exact_resource_cost(self):
        a,b=comb_grid(4),comb_grid(2)
        for db in [6,18]:
            fixed_eb=pilot_noise_variance(effective_db(db,b,16,'fixed_eb'),b,16)
            common_noise=pilot_noise_variance(effective_db(db,b,16,'fixed_noise'),b,16)
            reference=pilot_noise_variance(db,a,16)
            self.assertAlmostEqual(fixed_eb/reference,1.5)
            self.assertAlmostEqual(common_noise,reference)
            self.assertEqual(budget_noise(db,a,16,'fixed_noise'),budget_noise(db,b,16,'fixed_noise'))
            left=coupled_frame(161001,0,2,0,self.cfg['states']['initial'],budget_noise(db,a,16,'fixed_noise'),a,16,4)
            right=coupled_frame(161001,0,2,0,self.cfg['states']['initial'],budget_noise(db,b,16,'fixed_noise'),b,16,4)
            np.testing.assert_array_equal(left[0],right[0])
            self.assertAlmostEqual(effective_db(db,b,16,'fixed_noise')-db,10*np.log10(1.5))
            self.assertEqual(effective_db(db,a,16,'fixed_noise'),db)
        with self.assertRaises(ValueError):effective_db(6,b,16,'wrong')

    def test_common_oracle_errors_match_across_grids_at_fixed_noise(self):
        cfg=copy.deepcopy(self.cfg);cfg.update(episode_frames=16,symbols_per_frame=2)
        spec=next(s for s in cfg['scenarios'] if s['name']=='persistent15')
        a=run_episode(cfg,spec,6,0,161001,4,4,'fixed_noise')
        b=run_episode(cfg,spec,6,0,161001,4,2,'fixed_noise')
        np.testing.assert_array_equal(a[0]['oracle_zf'][:,5],b[0]['oracle_zf'][:,5])
        self.assertEqual(a[0]['recovery13'].shape,(16,6))
        self.assertIn('full17',b[0]);self.assertNotIn('full17',a[0])
        for result in [a,b]:
            for values in result[0].values():self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(all(c['last_update_after_frame']<15 for c in result[2]))
            self.assertTrue(all(e['applies_from_frame']==e['after_frame']+1 for e in result[1]))

    def test_sparse_budget_cases_are_exact_duplicate_controls(self):
        cfg=copy.deepcopy(self.cfg);cfg.update(episode_frames=16,symbols_per_frame=2)
        spec=cfg['scenarios'][0]
        a=run_episode(cfg,spec,6,0,161001,4,4,'fixed_eb')[0]
        b=run_episode(cfg,spec,6,0,161001,4,4,'fixed_noise')[0]
        for method in a:np.testing.assert_array_equal(a[method],b[method])

    def test_stream_parts_and_roles_are_separated(self):
        layout=comb_grid(4);args=(self.cfg['states']['initial'],.02,layout,16,4)
        a=coupled_frame(161001,0,1,0,*args)
        b=coupled_frame(161001,0,2,0,*args)
        self.assertFalse(np.array_equal(a[0],b[0]))
        repeat=coupled_frame(161001,0,1,0,*args)
        for x,y in zip(a,repeat):np.testing.assert_array_equal(x,y)

    def test_factorial_evaluation_keeps_phase_and_comparison_keys(self):
        cfg=copy.deepcopy(self.cfg);cfg['scenarios']=cfg['scenarios'][:1]
        def fake(cfg,spec,db,si,seed,start,spacing,budget):
            names=['oracle_zf','matched','recovery13']+(['full17'] if spacing==2 else [])
            return {n:np.zeros((320,6)) for n in names},[],[],np.zeros(320,bool)
        with patch('experiments.density_ber.run_episode',side_effect=fake):result=evaluate(cfg)
        # Regression: NumPy booleans must not break final JSON counters.
        json.dumps(dict(oracle_pass=sum(r['oracle_consistent'] for r in result['summary'] if r['receiver']=='oracle_zf')))
        self.assertEqual(len(result['runs']),32*2*2*7*4)
        self.assertEqual(len(result['summary']),2*2*7*4)
        self.assertEqual(len(result['paired']),2*2*4*2)
        self.assertEqual(set(r['phase'] for r in result['runs']),{'pre','post_early','post_late','post_all'})
        self.assertEqual(set(r['onset_phase'] for r in result['runs']),set(range(32)))

    def test_frozen_protocol_and_prior_seed_isolation(self):
        validate(self.cfg)
        for seed in [81001,91001,111001,121001,131001,141001,151001]:
            bad=copy.deepcopy(self.cfg);bad['seeds'][0]=seed
            with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(self.cfg);bad['release_checks']=4
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(self.cfg);bad['baseline_metadata_sha256']='0'*64
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(self.cfg);bad['states']['delay15'][15]=.8
        with self.assertRaises(ValueError):validate(bad)


if __name__=='__main__':unittest.main()
