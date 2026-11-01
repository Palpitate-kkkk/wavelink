import unittest
import numpy as np
from wavelink.receiver import comb_grid
from wavelink.estimation import TapDomainLS
from wavelink.signed_estimation import SignedTapDomainLS
from wavelink.reference import rotate_grid_reference


class SignedEstimationTests(unittest.TestCase):
    def setUp(self):
        self.layout = comb_grid(2)
        self.rng = np.random.default_rng(341101)
        self.p = np.ones((20, 28), complex)

    def test_causal_support_matches_existing_estimator(self):
        old, new = TapDomainLS(self.layout, 17), SignedTapDomainLS(self.layout, range(17))
        np.testing.assert_allclose(old.mapping, new.mapping, atol=1e-13)
        self.assertAlmostEqual(old.noise_gain, new.noise_gain)

    def test_negative_delay_is_identified_from_pilots(self):
        taps = np.zeros(64, complex)
        taps[52], taps[56], taps[0] = .1, .3j, 1.
        h = np.fft.fft(taps)
        y = np.broadcast_to(h, (20, 64))
        model = SignedTapDomainLS(self.layout, range(-16, 1))
        np.testing.assert_allclose(model.estimate(y, self.p), y[:, self.layout.bins(self.layout.data_tones)], atol=1e-12)
        self.assertLess(model.expected_nmse(h, 0.)[0], 1e-24)
        causal = TapDomainLS(self.layout, 17)
        self.assertGreater(causal.expected_nmse(h, 0.)[0], 1e-3)
        fp = causal.pilot_design
        hp = h[self.layout.bins(self.layout.pilot_tones)]
        residual = hp-fp@np.linalg.pinv(fp)@hp
        self.assertLess(np.sum(abs(residual)**2)/np.sum(abs(hp)**2), .25)

    def test_transported_support_is_same_estimator_for_arbitrary_noisy_input(self):
        y = self.rng.normal(size=(20, 64))+1j*self.rng.normal(size=(20, 64))
        old = TapDomainLS(self.layout, 17)
        for shift in range(17):
            model = SignedTapDomainLS(self.layout, range(-shift, 17-shift))
            phase = np.exp(-2j*np.pi*self.layout.bins(self.layout.data_tones)*shift/64)
            result = model.estimate(rotate_grid_reference(y, shift), self.p)*phase
            np.testing.assert_allclose(result, old.estimate(y, self.p), atol=1e-12)
            self.assertAlmostEqual(model.noise_gain, old.noise_gain, places=10)

    def test_expected_noise_variance_matches_monte_carlo(self):
        model = SignedTapDomainLS(self.layout, range(-8, 9))
        h = np.ones(64, complex)
        noise = (self.rng.normal(size=(6000, 64))+1j*self.rng.normal(size=(6000, 64)))*np.sqrt(.05/2)
        estimate = model.estimate(h+noise, np.ones((6000, 28)))
        mse = np.mean(abs(estimate-1)**2)
        bias, variance = model.expected_nmse(h, .05)
        self.assertAlmostEqual(mse, bias+variance, delta=.001)

    def test_invalid_or_aliased_support_rejected(self):
        for support in (None, (), (0, 0), (1, 0), (True,), (1.5,), (-33,), (32,), range(-16, 17)):
            with self.assertRaises(ValueError):
                SignedTapDomainLS(self.layout, support)


if __name__ == '__main__':
    unittest.main()
