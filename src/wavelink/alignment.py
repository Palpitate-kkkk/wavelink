"""Observable leading-tap alignment using the fixed acquisition marker only."""
from dataclasses import dataclass
import numpy as np
from .acquisition import Acquisition, marker_grid
from .synchronization import rotate_frequency


@dataclass(frozen=True)
class MarkerAlignment:
    start: int
    shift: int
    noise_floor: float
    threshold: float
    peak_power: float
    significant_tap: bool


def align_marker(samples, proposal, noise_multiple):
    """Move a captured boundary forward to the first significant causal tap.

    Correct the known 64-point marker with the SAME acquired CFO, estimate its
    impulse response, and threshold taps0..16 against mean tail power17..63.
    The tail can contain CFO/ISI leakage; it is not certified AWGN variance.
    Only the original fixed acquisition prefix is read. This cannot repair an
    already-late boundary, and weak first paths can be skipped. No truth enters.
    """
    if not isinstance(proposal, Acquisition) or not proposal.accepted:
        raise ValueError('accepted acquisition proposal required')
    if isinstance(proposal.start, bool) or not isinstance(proposal.start, (int, np.integer)) or not 0 <= proposal.start < 64:
        raise ValueError('original acquisition start must be in0..63')
    if isinstance(noise_multiple, bool) or not np.isscalar(noise_multiple) or np.iscomplexobj(noise_multiple) or not np.isfinite(noise_multiple) or noise_multiple <= 0:
        raise ValueError('positive finite real noise_multiple required')
    x = np.asarray(samples, complex)
    first = proposal.start+160
    if x.ndim != 1 or len(x) < first+64 or not np.all(np.isfinite(x[first:first+64])):
        raise ValueError('finite complete marker required')
    marker = rotate_frequency(x[first:first+64], -proposal.epsilon, 64, first)
    taps = np.fft.ifft(np.fft.fft(marker, norm='ortho')/marker_grid())
    powers = abs(taps)**2
    noise = float(np.mean(powers[17:]))
    peak = float(np.max(powers[:17]))
    threshold = max(float(noise_multiple)*noise, peak*1e-12, 1e-24)
    indices = np.flatnonzero(powers[:17] > threshold)
    shift = int(indices[0]) if len(indices) else 0
    return MarkerAlignment(int(proposal.start)+shift, shift, noise, threshold, peak, bool(len(indices)))
