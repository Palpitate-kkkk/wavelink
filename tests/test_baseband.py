import unittest
import numpy as np
from wavelink import ofdm_demodulate, ofdm_modulate, qpsk_demodulate, qpsk_modulate


class BasebandTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.bits = self.rng.integers(0, 2, 8 * 64 * 2, dtype=np.uint8)
        self.grid = qpsk_modulate(self.bits).reshape(8, 64)

    def test_qpsk_known_mapping(self):
        bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
        expected = np.array([1+1j, 1-1j, -1-1j, -1+1j]) / np.sqrt(2)
        np.testing.assert_allclose(qpsk_modulate(bits), expected)
        np.testing.assert_array_equal(qpsk_demodulate(expected), bits)

    def test_unitary_energy(self):
        samples = ofdm_modulate(self.grid, 0)
        self.assertAlmostEqual(float(np.mean(np.abs(self.grid)**2)), 1.0)
        self.assertAlmostEqual(float(np.sum(np.abs(samples)**2)), float(np.sum(np.abs(self.grid)**2)))

    def test_noiseless_roundtrip_cp_boundaries(self):
        for cp in (0, 16, 64):
            with self.subTest(cp=cp):
                tx = ofdm_modulate(self.grid, cp)
                rx = ofdm_demodulate(tx, 64, cp)
                np.testing.assert_allclose(rx, self.grid, atol=1e-12)
                np.testing.assert_array_equal(qpsk_demodulate(rx.ravel()), self.bits)

    def test_known_multipath_with_sufficient_cp(self):
        h = np.array([1, 0, 0.25+0.15j, 0, -0.1j])
        tx = ofdm_modulate(self.grid, 16)
        rx = ofdm_demodulate(np.convolve(tx, h)[:tx.size], 64, 16)
        expected = self.grid * np.fft.fft(h, 64)
        np.testing.assert_allclose(rx, expected, atol=1e-12)
        equalized = rx / np.fft.fft(h, 64)
        np.testing.assert_array_equal(qpsk_demodulate(equalized.ravel()), self.bits)

    def test_insufficient_cp_breaks_circular_channel_model(self):
        h = np.array([1, 0, 0.25+0.15j, 0, -0.1j])
        tx = ofdm_modulate(self.grid, 1)
        rx = ofdm_demodulate(np.convolve(tx, h)[:tx.size], 64, 1)
        self.assertGreater(float(np.mean(np.abs(rx - self.grid * np.fft.fft(h, 64))**2)), 1e-6)

    def test_invalid_inputs(self):
        for bad in ([0, 1, 0], [0, 2], [], [[0, 1]]):
            with self.subTest(bits=bad), self.assertRaises(ValueError):
                qpsk_modulate(bad)
        for cp in (-1, 65, 1.5, True):
            with self.subTest(cp=cp), self.assertRaises(ValueError):
                ofdm_modulate(self.grid, cp)
        with self.assertRaises(ValueError):
            ofdm_demodulate(np.zeros(79), 64, 16)
        with self.assertRaises(ValueError):
            qpsk_demodulate([complex("nan")])


if __name__ == "__main__":
    unittest.main()
