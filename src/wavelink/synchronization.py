"""Known-boundary repeated-training CFO estimation, not frame acquisition."""
import numpy as np


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name+' must be a positive integer')


def rotate_frequency(samples, epsilon, n_fft, sample_offset=0):
    """Multiply by exp(j 2 pi epsilon n/N), epsilon in subcarrier spacings.

    sample_offset is an absolute sample counter, allowing chunk continuity.
    It is not a symbol timing estimate. Sign reversal performs compensation.
    """
    x = np.asarray(samples, complex)
    _positive_int(n_fft, 'n_fft')
    if isinstance(sample_offset, bool) or not isinstance(sample_offset, (int, np.integer)) or sample_offset < 0:
        raise ValueError('nonnegative integer sample_offset required')
    if isinstance(epsilon, bool) or not np.isscalar(epsilon) or np.iscomplexobj(epsilon) or not np.isfinite(epsilon):
        raise ValueError('finite real normalized CFO required')
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError('finite nonempty sample vector required')
    n = np.arange(len(x), dtype=float) + sample_offset
    return x * np.exp(2j*np.pi*float(epsilon)*n/n_fft)


def estimate_repeated_cfo(useful_training, lag, n_fft):
    """angle(sum conj(r[n]) r[n+lag]) N/(2 pi lag).

    Input excludes training CP and contains only a repeated, static-channel
    region. No true channel/CFO/noise parameter is accepted. Principal-phase
    capture interval is [-N/(2 lag), N/(2 lag)], endpoints indistinguishable.
    The normalized correlation is a diagnostic, not an acceptance threshold.
    Overlapping products are correlated; count is not independent sample size.
    """
    r = np.asarray(useful_training, complex)
    _positive_int(lag, 'lag')
    _positive_int(n_fft, 'n_fft')
    if r.ndim != 1 or len(r) <= lag or not np.all(np.isfinite(r)):
        raise ValueError('finite repeated training longer than lag required')
    first, second = r[:-lag], r[lag:]
    product = np.vdot(first, second)
    scale = float(np.linalg.norm(first)*np.linalg.norm(second))
    if scale == 0 or abs(product) <= 1e-15*scale:
        raise ValueError('training correlation undefined')
    return float(np.angle(product)*n_fft/(2*np.pi*lag)), float(abs(product)/scale)
