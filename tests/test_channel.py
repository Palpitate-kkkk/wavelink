import unittest
import numpy as np
from wavelink.channel import add_awgn, qpsk_noise_variance, qpsk_ber, wilson_interval
from wavelink import ofdm_demodulate


class ChannelTests(unittest.TestCase):
    def test_complex_noise_statistics_and_fft(self):
        noise = add_awgn(np.zeros(800000), .4, np.random.default_rng(123))
        self.assertLess(abs(noise.mean()), .003)
        self.assertAlmostEqual(np.var(noise.real), .2, delta=.002)
        self.assertAlmostEqual(np.var(noise.imag), .2, delta=.002)
        self.assertLess(abs(np.mean(noise.real*noise.imag)), .002)
        self.assertLess(abs(np.mean(noise[:-1]*noise[1:].conj())), .003)
        spectrum = ofdm_demodulate(noise, 64, 16)
        self.assertAlmostEqual(float(np.mean(abs(spectrum)**2)), .4, delta=.003)

    def test_reproducibility_and_zero_noise(self):
        x = np.ones(100, dtype=complex)
        np.testing.assert_array_equal(add_awgn(x, .3, np.random.default_rng(8)), add_awgn(x, .3, np.random.default_rng(8)))
        np.testing.assert_array_equal(add_awgn(x, 0, np.random.default_rng(8)), x)

    def test_energy_conventions(self):
        self.assertAlmostEqual(qpsk_noise_variance(0), .5)
        self.assertAlmostEqual(qpsk_noise_variance(0, convention='transmitted'), .625)
        self.assertAlmostEqual(qpsk_noise_variance(0, cp_len=0, convention='transmitted'), .5)
        self.assertAlmostEqual(qpsk_noise_variance(10), .05)
        self.assertAlmostEqual(qpsk_ber(0), .07864960352514257)

    def test_interval_boundaries(self):
        lo, hi = wilson_interval(0, 1000)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, 0)
        self.assertLess(hi, .004)
        lo, hi = wilson_interval(1000, 1000)
        self.assertGreater(lo, .996)
        self.assertAlmostEqual(hi, 1)
        lo, hi = wilson_interval(50, 100)
        self.assertAlmostEqual(lo+hi, 1)

    def test_invalid_arguments(self):
        for variance in (-1, float('nan'), float('inf')):
            with self.subTest(variance=variance), self.assertRaises(ValueError):
                add_awgn([0], variance, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            qpsk_noise_variance(0, convention='unknown')
        with self.assertRaises(ValueError):
            wilson_interval(1, 0)
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)

    def test_fixed_budget_ber_regression(self):
        from experiments.awgn_ber import run
        cfg = dict(n_fft=64, cp_len=16, ebn0_db=[0, 6], seeds=[120, 121],
                   bits_per_seed=256000, batch_symbols=100, conventions=['useful', 'transmitted'])
        runs, summary = run(cfg)
        self.assertEqual(len(runs), 8)
        self.assertTrue(all(r['bits'] == 512000 for r in summary))
        self.assertTrue(all(r['theory_in_family_interval'] for r in summary))
        for db in cfg['ebn0_db']:
            points = [r for r in summary if r['ebn0_db'] == db]
            self.assertGreater(points[1]['ber'], points[0]['ber'])
