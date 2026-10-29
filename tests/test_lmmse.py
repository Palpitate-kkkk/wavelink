import copy
import json
import unittest
from pathlib import Path
import numpy as np
from wavelink.estimation import TapDomainLS, TapDomainLMMSE, ensemble_nmse
from wavelink.receiver import comb_grid
from experiments.lmmse_ber import evaluate, models, profile


class LMMSETests(unittest.TestCase):
    def setUp(self):
        self.layout = comb_grid(4)
        self.p = np.zeros(13); self.p[[0,4,12]] = [.8,.12,.08]

    def test_matches_posterior_covariance(self):
        m = TapDomainLMMSE(self.layout,self.p,.2)
        fp,fd = m.pilot_design,m.data_design
        r = np.diag(self.p)
        posterior = r-r@fp.conj().T@np.linalg.solve(fp@r@fp.conj().T+.2*np.eye(len(fp)),fp@r)
        predicted = np.trace(fd@posterior@fd.conj().T).real/len(fd)
        self.assertAlmostEqual(sum(ensemble_nmse(m.mapping,self.layout,self.p,.2)),predicted,places=12)
        for factor in [.1,10]:
            other = TapDomainLMMSE(self.layout,self.p,.2*factor)
            self.assertGreater(sum(ensemble_nmse(other.mapping,self.layout,self.p,.2)),predicted)

    def test_uniform_prior_equals_ridge_and_low_noise_ls(self):
        length = 13; noise = .1
        m = TapDomainLMMSE(self.layout,np.ones(length)/length,noise)
        fp,fd = m.pilot_design,m.data_design
        ridge = fd@np.linalg.solve(fp.conj().T@fp+noise*length*np.eye(length),fp.conj().T)
        np.testing.assert_allclose(m.mapping,ridge,atol=1e-13)
        small = TapDomainLMMSE(self.layout,np.ones(length)/length,1e-8)
        np.testing.assert_allclose(small.mapping,TapDomainLS(self.layout,length).mapping,atol=2e-5)
        scaled = TapDomainLMMSE(self.layout,3*self.p,3*noise)
        np.testing.assert_allclose(scaled.mapping,TapDomainLMMSE(self.layout,self.p,noise).mapping,atol=1e-13)

    def test_ensemble_prediction_with_wrong_pdp_and_noise(self):
        rng=np.random.default_rng(972)
        m=TapDomainLMMSE(self.layout,[.25]*4,.5)
        samples=12000; noise=.08
        taps=(rng.normal(size=(samples,13))+1j*rng.normal(size=(samples,13)))*np.sqrt(self.p/2)
        fp=np.exp(-2j*np.pi*np.outer(self.layout.pilot_tones,np.arange(13))/64)
        fd=np.exp(-2j*np.pi*np.outer(self.layout.data_tones,np.arange(13))/64)
        z=taps@fp.T+(rng.normal(size=(samples,len(fp)))+1j*rng.normal(size=(samples,len(fp))))*np.sqrt(noise/2)
        error=np.mean(abs(z@m.mapping.T-taps@fd.T)**2,axis=1)
        predicted=sum(ensemble_nmse(m.mapping,self.layout,self.p,noise))
        self.assertLess(abs(error.mean()-predicted),6*error.std(ddof=1)/np.sqrt(samples))

    def test_estimate_does_not_need_truth(self):
        m=TapDomainLMMSE(self.layout,self.p,.1)
        rng=np.random.default_rng(44)
        y=rng.normal(size=(3,64))+1j*rng.normal(size=(3,64))
        pilots=np.exp(1j*rng.uniform(0,2*np.pi,(3,len(self.layout.pilot_tones))))
        np.testing.assert_allclose(m.estimate(y,pilots),(y[:,self.layout.bins(self.layout.pilot_tones)]/pilots)@m.mapping.T)
        with self.assertRaises(ValueError): m.estimate(y,2*pilots)

    def test_invalid_priors_and_regularized_long_support(self):
        for p in [[],[0,0],[-1,2],[np.nan],[[1,2]],[1+1j],np.ones(65)]:
            with self.assertRaises(ValueError): TapDomainLMMSE(self.layout,p,.1)
        for noise in [0,-1,np.inf,np.nan]:
            with self.assertRaises(ValueError): TapDomainLMMSE(self.layout,self.p,noise)
        long = TapDomainLMMSE(self.layout,np.ones(64)/64,.1)
        self.assertTrue(np.all(np.isfinite(long.mapping)))
        with self.assertRaises(ValueError): ensemble_nmse(long.mapping,self.layout,self.p,-1)
        with self.assertRaises(ValueError): ensemble_nmse(np.zeros((2,2)),self.layout,self.p,.1)

    def test_small_benchmark_reproducibility_and_optimum(self):
        cfg=json.loads(Path('experiments/lmmse.json').read_text())
        cfg.update(frames_per_seed=4,symbols_per_frame=3,seeds=[41001],pilot_spacings=[4],ebn0_db=[6])
        a=evaluate(cfg); b=evaluate(copy.deepcopy(cfg))
        self.assertEqual(a,b)
        for spec in cfg['channels']:
            p=profile(spec)
            fitted=models(self.layout,p,.1,cfg)
            optimum=sum(ensemble_nmse(fitted['lmmse_matched_n1'].mapping,self.layout,p,.1))
            for m in fitted.values():
                self.assertGreaterEqual(sum(ensemble_nmse(m.mapping,self.layout,p,.1))+1e-13,optimum)
        cfg['cp_len']=2
        with self.assertRaises(ValueError): evaluate(cfg)


if __name__=='__main__':
    unittest.main()
