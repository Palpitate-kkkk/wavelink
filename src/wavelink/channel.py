"""Complex baseband AWGN; variance always means E[|n|^2]."""
import math
from statistics import NormalDist
import numpy as np


def add_awgn(samples, noise_variance, rng):
    """Add CN(0, noise_variance); each real component has variance N0/2.

    Do not estimate power from this realization or rescale realized noise.
    The caller supplies an explicit numpy Generator for reproducibility.
    """
    x = np.asarray(samples, dtype=np.complex128)
    if not x.size or not np.all(np.isfinite(x)):
        raise ValueError("samples must be nonempty and finite")
    if not np.isscalar(noise_variance) or not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    return x + np.sqrt(noise_variance / 2) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))


def qpsk_noise_variance(ebn0_db, n_fft=64, cp_len=16, convention="useful"):
    """Es=1, two uncoded bits/tone, all tones active; perfect synchronization.

    useful: Eb excludes CP; transmitted: Eb includes expected CP energy.
    Unitary FFT preserves noise variance. No pilots, guards or coding here.
    """
    from .baseband import _validate_sizes
    _validate_sizes(n_fft, cp_len)
    if convention not in ("useful", "transmitted"):
        raise ValueError("convention must be useful or transmitted")
    if not np.isscalar(ebn0_db) or not np.isfinite(ebn0_db):
        raise ValueError("ebn0_db must be finite")
    overhead = 1 + cp_len / n_fft if convention == "transmitted" else 1.0
    variance = overhead * 0.5 * 10.0 ** (-float(ebn0_db) / 10)
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("Eb/N0 outside representable range")
    return variance


def qpsk_ber(ebn0_db):
    """Coherent Gray QPSK BER, useful-bit Eb/N0, no coding or fading."""
    return 0.5 * math.erfc(math.sqrt(10.0 ** (float(ebn0_db) / 10)))


def wilson_interval(errors, bits, confidence=0.95):
    """Two-sided Wilson score interval for a fixed-size Bernoulli sample.

    Zero errors still yields a positive upper limit. Not an anytime interval.
    """
    if not isinstance(errors, (int, np.integer)) or not isinstance(bits, (int, np.integer)):
        raise ValueError("counts must be integers")
    if bits <= 0 or not 0 <= errors <= bits or not 0 < confidence < 1:
        raise ValueError("invalid counts or confidence")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = errors / bits
    d = 1 + z*z / bits
    center = (p + z*z / (2*bits)) / d
    radius = z * math.sqrt(p*(1-p)/bits + z*z/(4*bits*bits)) / d
    return max(0.0, center-radius), min(1.0, center+radius)
