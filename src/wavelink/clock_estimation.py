"""Offline clock estimation from scheduled copies of a known training marker."""
import numpy as np


def estimate_clock(samples, marker, starts, *, radius, min_score=.7):
    """Return observed/nominal rate ratio and per-marker diagnostics.

    starts are transmitter coordinates; radius bounds initial offset plus drift.
    At least three markers must pass the normalized correlation gate.
    """
    x, p, t = map(np.asarray, (samples, marker, starts))
    if (x.ndim != 1 or p.ndim != 1 or len(p) < 3 or t.ndim != 1
            or t.dtype.kind not in 'iu' or len(t) < 3 or np.any(t[1:] <= t[:-1])
            or not isinstance(radius, int) or radius < 2 or not 0 < min_score <= 1
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(p))):
        raise ValueError('invalid samples, training schedule, or search parameters')
    energy = np.linalg.norm(p)
    if not energy:
        raise ValueError('empty training energy')
    anchors = []
    for start in map(int, t):
        lo, hi = start-radius, start+radius
        if lo < 0 or hi+len(p) > len(x):
            raise ValueError('search window outside recording')
        windows = np.lib.stride_tricks.sliding_window_view(x[lo:hi+len(p)], len(p))
        denom = np.linalg.norm(windows, axis=1)*energy
        score = np.divide(abs(windows @ p.conj()), denom,
                          out=np.zeros(len(windows)), where=denom > 0)
        j = int(np.argmax(score))
        if j == 0 or j == len(score)-1 or score[j] < min_score:
            continue
        a, b, c = score[j-1:j+2]
        delta = .5*(a-c)/(a-2*b+c) if a-2*b+c < 0 else 0.
        anchors.append((start, lo+j+delta, float(b)))
    if len(anchors) < 3:
        raise ValueError('insufficient accepted training markers')
    nominal, observed, _ = np.asarray(anchors).T
    centered = nominal-nominal.mean()
    drift = observed-nominal
    slope = float(centered @ drift / (centered @ centered))
    offset = float(drift.mean()-slope*nominal.mean())
    residual = drift-offset-slope*nominal
    return dict(rate_ratio=1+slope, ppm=1e6*slope, offset_samples=offset,
                residual_rms_samples=float(np.sqrt(np.mean(residual**2))),
                anchors=anchors)
