import unittest
import numpy as np
from wavelink import qpsk_modulate, qpsk_demodulate, ofdm_modulate, ofdm_demodulate
from wavelink.receiver import ResourceGrid, comb_grid, apply_multipath, ls_channel_estimate, zero_force, pilot_noise_variance
from wavelink.channel import add_awgn


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(810)
        self.layout = comb_grid(4)
        self.bits = self.rng.integers(0, 2, 30*len(self.layout.data_tones)*2)
        self.data = qpsk_modulate(self.bits).reshape(30, -1)
        self.pilots = qpsk_modulate(self.rng.integers(0, 2, 30*len(self.layout.pilot_tones)*2)).reshape(30, -1)
        self.grid = self.layout.pack(self.data, self.pilots)

    def test_grid_and_nulls(self):
        self.assertEqual(len(self.layout.active_tones), 52)
        self.assertEqual(len(self.layout.pilot_tones), 16)
        self.assertEqual(len(self.layout.data_tones), 36)
        nulls = sorted(set(range(64))-set(self.layout.bins(self.layout.active_tones)))
        self.assertTrue(np.all(self.grid[:, nulls] == 0))
        np.testing.assert_allclose(np.sum(abs(self.grid)**2, axis=1), 52)
        with self.assertRaises(ValueError):
            ResourceGrid(64, (-2, -1, 1, 2), (-2, 1))
        with self.assertRaises(ValueError):
            comb_grid(0)

    def test_flat_channel_ls_exact(self):
        h = .4+.7j
        est = ls_channel_estimate(self.grid*h, self.pilots, self.layout)
        np.testing.assert_allclose(est, h, atol=1e-14)
        rx = self.grid[:, self.layout.bins(self.layout.data_tones)]*h
        np.testing.assert_array_equal(qpsk_demodulate(zero_force(rx, est).ravel()), self.bits)

    def test_interpolation_signed_frequency(self):
        tones = np.asarray(self.layout.active_tones)
        channel = np.ones(64, complex)
        channel[self.layout.bins(tones)] = 1+.01*tones + 1j*(.5-.02*tones)
        est = ls_channel_estimate(self.grid*channel, self.pilots, self.layout)
        np.testing.assert_allclose(est, np.broadcast_to(channel[self.layout.bins(self.layout.data_tones)], est.shape), atol=1e-14)

    def test_streaming_convolution(self):
        x = self.rng.normal(size=137)+1j*self.rng.normal(size=137)
        taps = np.array([1, 0, .2j, -.1])
        full, _ = apply_multipath(x, taps)
        first, state = apply_multipath(x[:2], taps)
        second, state = apply_multipath(x[2:41], taps, state)
        last, _ = apply_multipath(x[41:], taps, state)
        np.testing.assert_allclose(full, np.convolve(x, taps)[:len(x)])
        np.testing.assert_allclose(np.r_[first, second, last], full)
        same, state = apply_multipath(x, [1])
        np.testing.assert_array_equal(same, x)
        self.assertEqual(state.size, 0)

    def test_cp_boundary_and_oracle(self):
        taps = np.array([1, 0, .2j, .3])
        for cp in (3, 2):
            tx = ofdm_modulate(self.grid, cp)
            rx, _ = apply_multipath(tx, taps)
            got = ofdm_demodulate(rx, 64, cp)
            expected = self.grid*np.fft.fft(taps, 64)
            error = np.mean(abs(got[1:]-expected[1:])**2)
            if cp == 3:
                self.assertLess(error, 1e-25)
                eq = zero_force(got[:, self.layout.bins(self.layout.data_tones)], np.fft.fft(taps,64)[self.layout.bins(self.layout.data_tones)])
                np.testing.assert_array_equal(qpsk_demodulate(eq.ravel()), self.bits)
            else:
                self.assertGreater(error, 1e-4)

    def test_ls_pilot_noise_and_interpolation_error(self):
        # Independent complex pilot noise: LS pilot MSE equals N0 for unit pilots.
        p = qpsk_modulate(self.rng.integers(0,2,400000))
        y = add_awgn(p*(.7+.2j), .1, self.rng)
        self.assertAlmostEqual(float(np.mean(abs(y/p-(.7+.2j))**2)), .1, delta=.001)
        taps = np.r_[1, np.zeros(11), .4j]
        h = np.fft.fft(taps, 64)
        est = ls_channel_estimate(self.grid*h, self.pilots, self.layout)
        self.assertGreater(np.mean(abs(est-h[self.layout.bins(self.layout.data_tones)])**2), .01)

    def test_energy_accounting(self):
        self.assertAlmostEqual(pilot_noise_variance(0,self.layout,16), 52/72*1.25)
        data = qpsk_modulate(self.rng.integers(0,2,4000*36*2)).reshape(4000,36)
        pilots = qpsk_modulate(self.rng.integers(0,2,4000*16*2)).reshape(4000,16)
        tx = ofdm_modulate(self.layout.pack(data,pilots),16)
        empirical_eb = np.sum(abs(tx)**2)/(4000*36*2)
        self.assertAlmostEqual(empirical_eb, pilot_noise_variance(0,self.layout,16), delta=.005)

    def test_null_and_invalid_inputs(self):
        out = zero_force([1,1], [0,1])
        np.testing.assert_array_equal(out, [0,1])
        with self.assertRaises(ValueError):
            apply_multipath([1], [0])
        with self.assertRaises(ValueError):
            apply_multipath([1], [1,.2], history=[])
        with self.assertRaises(ValueError):
            ls_channel_estimate(self.grid, self.pilots*0, self.layout)

    def test_benchmark_accounting_and_determinism(self):
        from experiments.multipath_ber import run, diagnostics
        cfg = dict(channels=[dict(name='flat',delays=[0],gains=[[0.6,0.8]])],
                   pilot_spacings=[4], cp_lengths=[16], ebn0_db=[6], seeds=[101],
                   frames_per_seed=4, symbols_per_frame=10)
        rows, summary = run(cfg)
        self.assertEqual(rows, run(cfg)[0])
        self.assertEqual(len(summary), 3)
        self.assertTrue(all(r['bits']==4*10*36*2 for r in summary))
        self.assertTrue(summary[1]['oracle_consistent'])
        self.assertLess(diagnostics(cfg)[0]['noiseless_ls_data_nmse'], 1e-25)
