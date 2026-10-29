"""Pilot-domain LS constrained to a causal finite-length channel subspace."""
import numpy as np


class TapDomainLS:
    """Fit contiguous delays 0..length-1 with a partial DFT design matrix.

    Unit-modulus known pilots are required: division by pilots then ordinary
    LS is equivalent to LS on received pilots with white noise. Unequal pilot
    powers would require weighted LS, which this implementation rejects.
    No true channel, noise variance, or actual delay support enters estimate().
    """
    def __init__(self, layout, length):
        if isinstance(length, bool) or not isinstance(length, (int, np.integer)):
            raise ValueError('length must be an integer')
        if not 1 <= length <= min(layout.n_fft, len(layout.pilot_tones)):
            raise ValueError('need 1 <= length <= number of pilot observations')
        self.layout, self.length = layout, int(length)
        delays = np.arange(length)
        self.pilot_design = np.exp(-2j*np.pi*np.outer(layout.pilot_tones, delays)/layout.n_fft)
        self.data_design = np.exp(-2j*np.pi*np.outer(layout.data_tones, delays)/layout.n_fft)
        u, singular, vh = np.linalg.svd(self.pilot_design, full_matrices=False)
        self.rank = int(np.count_nonzero(singular > singular[0]*1e-12))
        self.condition = float(singular[0]/singular[-1])
        if self.rank != length or self.condition > 1e8:
            raise ValueError('pilot geometry cannot stably identify the assumed taps')
        # SVD pseudoinverse avoids explicitly forming/inverting F^H F.
        inverse = (vh.conj().T / singular) @ u.conj().T
        self.mapping = self.data_design @ inverse
        self.noise_gain = float(np.sum(abs(self.mapping)**2))

    def estimate(self, received, pilots):
        y, p = np.asarray(received), np.asarray(pilots)
        if y.ndim != 2 or y.shape[1] != self.layout.n_fft or len(y) == 0:
            raise ValueError('received shape mismatch')
        if p.shape != (len(y), len(self.layout.pilot_tones)):
            raise ValueError('pilot shape mismatch')
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
            raise ValueError('finite inputs required')
        if not np.allclose(abs(p), 1, rtol=0, atol=1e-12):
            raise ValueError('unit-modulus pilots required for unweighted LS')
        observations = y[:, self.layout.bins(self.layout.pilot_tones)] / p
        return observations @ self.mapping.T

    def expected_nmse(self, true_response, noise_variance):
        """Analysis only: bias/variance of this linear estimator under AWGN.

        Requires a static diagonal OFDM channel (sufficient CP, perfect sync).
        True response is used by the evaluator only, never by estimate().
        """
        h = np.asarray(true_response, complex)
        if h.shape != (self.layout.n_fft,) or not np.all(np.isfinite(h)):
            raise ValueError('finite full FFT channel response required')
        if not np.isfinite(noise_variance) or noise_variance < 0:
            raise ValueError('nonnegative noise variance required')
        hd = h[self.layout.bins(self.layout.data_tones)]
        denom = float(np.sum(abs(hd)**2))
        if denom <= 0:
            raise ValueError('NMSE undefined for a zero data-channel response')
        bias = self.mapping @ h[self.layout.bins(self.layout.pilot_tones)]-hd
        bias_nmse = float(np.sum(abs(bias)**2)/denom)
        variance_nmse = noise_variance*self.noise_gain/denom
        return bias_nmse, variance_nmse
