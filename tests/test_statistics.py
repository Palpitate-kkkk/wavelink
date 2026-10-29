import copy
import json
import unittest
from pathlib import Path
import numpy as np
from wavelink.estimation import TapDomainLS
from wavelink.statistics import learn_pilot_statistics
from wavelink.receiver import comb_grid
from experiments.learned_ber import evaluate, calibrate, rng_for, simulate_frame


class PilotStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.layout=comb_grid(4)
        self.fp=TapDomainLS(self.layout,13).pilot_design

    def test_noiseless_powers_without_true_normalization(self):
        powers=np.zeros(13);powers[[0,4,12]]=[2,.6,.4]
        # Each frame has exact tap powers; overall power is deliberately 3.
        rng=np.random.default_rng(96)
        h=np.sqrt(powers)*np.exp(1j*rng.uniform(0,2*np.pi,(12,13)))
        z=np.repeat((h@self.fp.T)[:,None,:],6,axis=1)
        fit=learn_pilot_statistics(z,self.layout,13)
        np.testing.assert_allclose(fit.powers,powers,atol=1e-11)
        self.assertAlmostEqual(fit.powers.sum(),3,places=10)
        self.assertEqual(fit.noise_variance,1e-12)

    def test_gaussian_noise_and_noise_debiasing(self):
        rng=np.random.default_rng(337)
        count,symbols=6000,6
        powers=np.zeros(13);powers[[0,4,12]]=[.8,.12,.08]
        n0=.4
        h=(rng.normal(size=(count,13))+1j*rng.normal(size=(count,13)))*np.sqrt(powers/2)
        z=(h@self.fp.T)[:,None,:]+(rng.normal(size=(count,symbols,len(self.fp)))+1j*rng.normal(size=(count,symbols,len(self.fp))))*np.sqrt(n0/2)
        fit=learn_pilot_statistics(z,self.layout,13)
        self.assertLess(abs(fit.noise_variance/n0-1),.015)
        np.testing.assert_allclose(fit.raw_powers,powers,atol=.025)
        # Clipping has a positive finite-sample bias at genuinely empty delays.
        self.assertTrue(np.all(fit.powers>0))
        self.assertEqual(fit.frames,count)
        self.assertEqual(fit.symbols,symbols)

    def test_noise_cancels_only_static_channel(self):
        z=np.zeros((3,4,len(self.fp)),complex)
        z[:,1::2]=self.fp[:,0]
        fit=learn_pilot_statistics(z,self.layout,13)
        self.assertAlmostEqual(fit.noise_variance,.5)

    def test_invalid_samples_and_geometry(self):
        for z in [np.zeros((1,4,len(self.fp))),np.zeros((3,1,len(self.fp))),np.zeros((3,4,2)),np.zeros((3,4)),np.full((3,4,len(self.fp)),np.nan)]:
            with self.assertRaises(ValueError):learn_pilot_statistics(z,self.layout,13)
        with self.assertRaises(ValueError):learn_pilot_statistics(np.zeros((3,4,len(self.fp))),self.layout,17)

    def test_training_prefix_and_test_stream_independence(self):
        cfg=json.loads(Path('experiments/learned.json').read_text())
        cfg.update(calibration_frames=[4,8],symbols_per_frame=4)
        p=np.ones(13)/13
        a=calibrate(cfg,self.layout,p,.1,51001,0,4,0)
        cfg['calibration_frames']=[4]
        b=calibrate(cfg,self.layout,p,.1,51001,0,4,0)
        np.testing.assert_array_equal(a[4].powers,b[4].powers)
        self.assertEqual(a[4].noise_variance,b[4].noise_variance)
        train=rng_for(51001,0,4,0,1).normal(size=16)
        test=rng_for(51001,0,4,0,2).normal(size=16)
        self.assertFalse(np.array_equal(train,test))
        # Test frames cannot depend on how many calibration frames were read.
        x=simulate_frame(rng_for(51001,0,4,0,2),p,.1,self.layout,16,4)
        calibrate(cfg,self.layout,p,.1,51001,0,4,0)
        y=simulate_frame(rng_for(51001,0,4,0,2),p,.1,self.layout,16,4)
        for left,right in zip(x,y):np.testing.assert_array_equal(left,right)

    def test_small_experiment_reproducible_and_conditional_theory(self):
        cfg=json.loads(Path('experiments/learned.json').read_text())
        cfg.update(calibration_frames=[4],frames_per_seed=6,symbols_per_frame=4,seeds=[51001,51002],pilot_spacings=[4],ebn0_db=[6])
        a=evaluate(cfg)
        self.assertEqual(a,evaluate(copy.deepcopy(cfg)))
        self.assertEqual(len(a[2]),4)
        for row in a[0]:
            if row['receiver']=='lmmse_matched':
                self.assertEqual(row['paired_nmse_minus_matched'],0)
        cfg['cp_len']=2
        with self.assertRaises(ValueError):evaluate(cfg)


if __name__=='__main__':unittest.main()
