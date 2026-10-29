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


def _tap_powers(powers, n_fft):
    raw = np.asarray(powers)
    if np.iscomplexobj(raw):
        raise ValueError('tap powers must be real')
    p = np.asarray(raw, float)
    if p.ndim != 1 or not 1 <= len(p) <= n_fft or not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError('finite nonnegative tap powers with positive total required')
    return p


class TapDomainLMMSE(TapDomainLS):
    """Zero-mean, uncorrelated-tap LMMSE with explicit PDP and noise prior.

    Powers are E|h[l]|^2, not normalized internally. Zero powers are allowed.
    Positive noise regularizes even rank-deficient pilot geometries. This is
    per-symbol frequency estimation; no averaging across OFDM symbols.
    Inherits the unit-modulus pilot validation and data-tone output contract.
    """
    def __init__(self, layout, powers, noise_variance):
        powers = _tap_powers(powers, layout.n_fft)
        if not np.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError('strictly positive assumed noise variance required')
        self.layout, self.length = layout, len(powers)
        self.powers = powers.copy()
        self.noise_variance = float(noise_variance)
        delays = np.arange(self.length)
        self.pilot_design = np.exp(-2j*np.pi*np.outer(layout.pilot_tones, delays)/layout.n_fft)
        self.data_design = np.exp(-2j*np.pi*np.outer(layout.data_tones, delays)/layout.n_fft)
        cross = (self.data_design*powers) @ self.pilot_design.conj().T
        covariance = (self.pilot_design*powers) @ self.pilot_design.conj().T
        covariance += noise_variance*np.eye(len(layout.pilot_tones))
        # Solve C X = R_pd, then take Hermitian transpose; no explicit inverse.
        self.mapping = np.linalg.solve(covariance, cross.conj().T).conj().T
        self.noise_gain = float(np.sum(abs(self.mapping)**2))


def ensemble_nmse(mapping, layout, true_powers, true_noise_variance):
    """Evaluator: ensemble E||A z-Hd||^2 / E||Hd||^2 under true PDP.

    Independent zero-mean taps, sufficient CP and unit pilots are assumed.
    Works for any fixed complex linear pilot-to-data mapping, including a
    mismatched estimator. This is not E[per-realization normalized error].
    """
    p = _tap_powers(true_powers, layout.n_fft)
    a = np.asarray(mapping, complex)
    if a.shape != (len(layout.data_tones), len(layout.pilot_tones)) or not np.all(np.isfinite(a)):
        raise ValueError('finite pilot-to-data matrix required')
    if not np.isfinite(true_noise_variance) or true_noise_variance < 0:
        raise ValueError('nonnegative true noise variance required')
    delays = np.arange(len(p))
    fp = np.exp(-2j*np.pi*np.outer(layout.pilot_tones,delays)/layout.n_fft)
    fd = np.exp(-2j*np.pi*np.outer(layout.data_tones,delays)/layout.n_fft)
    residual = a @ fp-fd
    signal = float(np.sum(abs(residual)**2*p))
    noise = float(true_noise_variance*np.sum(abs(a)**2))
    denominator = len(layout.data_tones)*p.sum()
    return signal/denominator, noise/denominator
