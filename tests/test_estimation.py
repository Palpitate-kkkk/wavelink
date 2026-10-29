import unittest
import numpy as np
from wavelink import qpsk_modulate
from wavelink.channel import add_awgn
from wavelink.receiver import comb_grid, ResourceGrid
from wavelink.estimation import TapDomainLS


class TapLSTests(unittest.TestCase):
    def setUp(self):
        self.layout = comb_grid(4)
        self.rng = np.random.default_rng(562)
        self.p = qpsk_modulate(self.rng.integers(0,2,16*10*2)).reshape(10,16)
        self.grid = self.layout.pack(np.ones((10,36)),self.p)

    def test_exact_matched_and_overlong_noiseless(self):
        for length in (4,8,13):
            taps = np.array([.8+.6j,.3-.2j,0,.1+.1j])
            h = np.fft.fft(taps,64)
            model = TapDomainLS(self.layout,length)
            got = model.estimate(self.grid*h,self.p)
            expected = h[self.layout.bins(self.layout.data_tones)]
            np.testing.assert_allclose(got,np.broadcast_to(expected,got.shape),atol=1e-12)
            self.assertLess(model.expected_nmse(h,0)[0],1e-24)

    def test_long_channel_match_and_underfit_bias(self):
        taps=np.zeros(13,complex);taps[[0,4,12]]=[.8+.6j,.3-.2j,.2+.25j]
        h=np.fft.fft(taps,64)
        for length in (2,4,8):
            model=TapDomainLS(self.layout,length)
            estimate=model.estimate(self.grid*h,self.p)
            hd=h[self.layout.bins(self.layout.data_tones)]
            observed=np.sum(abs(estimate-hd)**2)/(len(estimate)*np.sum(abs(hd)**2))
            predicted=model.expected_nmse(h,0)[0]
            self.assertGreater(predicted,.01)
            self.assertAlmostEqual(observed,predicted,places=12)
        self.assertLess(TapDomainLS(self.layout,13).expected_nmse(h,0)[0],1e-24)

    def test_empirical_bias_variance_prediction(self):
        trials=6000
        p=qpsk_modulate(self.rng.integers(0,2,trials*16*2)).reshape(trials,16)
        grid=self.layout.pack(np.ones((trials,36)),p)
        h=np.fft.fft([1,0,0,.2j],64)
        y=add_awgn(grid*h,.15,self.rng)
        hd=h[self.layout.bins(self.layout.data_tones)]
        for length in (2,4,13):
            model=TapDomainLS(self.layout,length)
            got=model.estimate(y,p)
            observed=np.sum(abs(got-hd)**2)/(trials*np.sum(abs(hd)**2))
            expected=sum(model.expected_nmse(h,.15))
            self.assertAlmostEqual(observed,expected,delta=.025*expected)

    def test_noise_cost_of_overlong_model(self):
        short=TapDomainLS(self.layout,4)
        long=TapDomainLS(self.layout,13)
        self.assertGreater(long.noise_gain,short.noise_gain)
        np.testing.assert_allclose(short.pilot_design[:,0],1)
        self.assertEqual(short.rank,4)

    def test_reject_unidentifiable_and_nonunit_pilots(self):
        for length in (0,17,1.5,True):
            with self.assertRaises(ValueError):
                TapDomainLS(self.layout,length)
        # A nearly contiguous half-band pilot geometry is ill-conditioned.
        layout=ResourceGrid(64,tuple(range(-31,0)),tuple(range(-31,-2))+(-1,))
        with self.assertRaises(ValueError):
            TapDomainLS(layout,30)
        model=TapDomainLS(self.layout,4)
        with self.assertRaises(ValueError):
            model.estimate(self.grid,self.p*2)
        with self.assertRaises(ValueError):
            model.estimate(self.grid[:,:63],self.p)
        with self.assertRaises(ValueError):
            model.expected_nmse(np.zeros(64),.1)

    def test_extension_keeps_existing_baselines_identical(self):
        from experiments.multipath_ber import run
        from experiments.tapls_ber import evaluate
        cfg=dict(channels=[dict(name='flat',delays=[0],gains=[[.6,.8]])],pilot_spacings=[4],
                 cp_lengths=[16],ebn0_db=[6],seeds=[101],frames_per_seed=4,symbols_per_frame=10)
        old_rows,_=run(cfg)
        extended=dict(cfg,tap_lengths=[2,4])
        rows,summary,_=evaluate(extended)
        self.assertEqual(old_rows,[r for r in rows if not r['receiver'].startswith('tap_ls')])
        self.assertTrue(all(r['nmse_consistent'] for r in summary if r['nmse_consistent']!=''))
        with self.assertRaises(ValueError):
            evaluate(dict(extended,channels=[dict(name='late',delays=[0,20],gains=[[1,0],[.1,0]])]))
