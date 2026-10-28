"""Static SISO multipath and comb-pilot LS receiver primitives.

Tone labels are signed FFT frequencies; arrays retain native NumPy FFT order.
"""
from dataclasses import dataclass
import numpy as np
from .baseband import _validate_sizes


@dataclass(frozen=True)
class ResourceGrid:
    n_fft: int
    active_tones: tuple
    pilot_tones: tuple

    def __post_init__(self):
        _validate_sizes(self.n_fft, 0)
        for tones in (self.active_tones, self.pilot_tones):
            if not tones or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) for k in tones):
                raise ValueError('tone lists must contain integers')
            if tuple(sorted(set(tones))) != tuple(tones):
                raise ValueError('tone lists must be unique and sorted')
            if any(k == 0 or k < -(self.n_fft//2) or k > (self.n_fft-1)//2 for k in tones):
                raise ValueError('invalid signed tone or DC allocation')
        if not set(self.pilot_tones).issubset(self.active_tones) or not self.data_tones:
            raise ValueError('pilots must be active; at least one data tone required')
        for band in self.bands:
            if band[0] not in self.pilot_tones or band[-1] not in self.pilot_tones:
                raise ValueError('each active band must be bracketed by pilots')

    @property
    def bands(self):
        a = np.asarray(self.active_tones)
        return np.split(a, np.flatnonzero(np.diff(a) != 1)+1)

    @property
    def data_tones(self):
        return tuple(k for k in self.active_tones if k not in self.pilot_tones)

    def bins(self, tones):
        return np.asarray(tones, dtype=int) % self.n_fft

    def pack(self, data, pilots):
        data, pilots = np.asarray(data), np.asarray(pilots)
        if data.ndim != 2 or data.shape[1] != len(self.data_tones) or data.shape[0] == 0:
            raise ValueError('data shape must be (symbols, data tones)')
        if pilots.shape != (data.shape[0], len(self.pilot_tones)):
            raise ValueError('pilot shape does not match resource grid')
        if not np.all(np.isfinite(data)) or not np.all(np.isfinite(pilots)) or np.any(abs(pilots) == 0):
            raise ValueError('finite symbols and nonzero pilots required')
        grid = np.zeros((len(data), self.n_fft), dtype=complex)
        grid[:, self.bins(self.data_tones)] = data
        grid[:, self.bins(self.pilot_tones)] = pilots
        return grid


def comb_grid(spacing=4):
    """64-point pedagogical grid: 52 active tones; not an 802.11 implementation."""
    if isinstance(spacing, bool) or not isinstance(spacing, (int, np.integer)) or spacing < 1:
        raise ValueError('spacing must be a positive integer')
    bands = (list(range(-26, 0)), list(range(1, 27)))
    pilots = sorted({k for band in bands for k in band[::spacing]+[band[-1]]})
    return ResourceGrid(64, tuple(bands[0]+bands[1]), tuple(pilots))


def apply_multipath(samples, taps, history=None):
    """Causal linear FIR with explicit history; never normalize taps implicitly.

    Returns an output of equal length and L-1 input samples for the next chunk.
    Missing history means the transmitter was silent before this call.
    """
    x, h = np.asarray(samples, complex), np.asarray(taps, complex)
    if x.ndim != 1 or h.ndim != 1 or not x.size or not h.size:
        raise ValueError('samples and taps must be nonempty vectors')
    state = np.zeros(len(h)-1, complex) if history is None else np.asarray(history, complex)
    if state.shape != (len(h)-1,) or not all(np.all(np.isfinite(a)) for a in (x, h, state)):
        raise ValueError('invalid history or nonfinite inputs')
    if not np.any(abs(h) > 0):
        raise ValueError('channel cannot be identically zero')
    joined = np.concatenate((state, x))
    y = np.convolve(joined, h)[len(h)-1:len(h)-1+len(x)]
    return y, joined[-(len(h)-1):].copy() if len(h) > 1 else np.empty(0, complex)


def ls_channel_estimate(received, pilots, layout):
    """Pilot LS Yp/Xp, followed by real/imag linear interpolation per band.

    No interpolation through DC/guards, extrapolation or oracle channel use.
    Returns estimates on data tones only, in layout.data_tones order.
    """
    y, p = np.asarray(received), np.asarray(pilots)
    if y.ndim != 2 or y.shape[1] != layout.n_fft or y.shape[0] == 0:
        raise ValueError('received grid shape mismatch')
    if p.shape != (len(y), len(layout.pilot_tones)) or np.any(abs(p) == 0):
        raise ValueError('invalid pilots')
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError('nonfinite received symbols or pilots')
    hp = y[:, layout.bins(layout.pilot_tones)] / p
    pt, dt = np.array(layout.pilot_tones), np.array(layout.data_tones)
    out = np.empty((len(y), len(dt)), complex)
    for band in layout.bands:
        pm = (pt >= band[0]) & (pt <= band[-1])
        dm = (dt >= band[0]) & (dt <= band[-1])
        for row in range(len(y)):
            out[row, dm] = np.interp(dt[dm], pt[pm], hp[row, pm].real) + 1j*np.interp(dt[dm], pt[pm], hp[row, pm].imag)
    return out


def zero_force(received, estimated_channel, power_floor=1e-12):
    """ZF on usable tones; return zero (erasure) at numerically singular tones.

    A zero symbol maps to bits 00 in the existing hard demapper. This guard
    prevents inf/nan; it does not recover information lost in a spectral null.
    """
    y, h = np.broadcast_arrays(np.asarray(received, complex), np.asarray(estimated_channel, complex))
    if not y.size or not np.all(np.isfinite(y)) or not np.all(np.isfinite(h)):
        raise ValueError('finite nonempty arrays required')
    if not np.isfinite(power_floor) or power_floor < 0:
        raise ValueError('invalid power floor')
    return np.divide(y, h, out=np.zeros(y.shape, complex), where=abs(h)**2 > power_floor)


def pilot_noise_variance(ebn0_db, layout, cp_len):
    """Total expected transmitted energy per DATA bit, including pilots/CP.

    Unit-energy independent zero-mean QPSK on every active tone, including
    known pilots; no coding. Eb = Na/(2 Nd) * (N+CP)/N.
    Do not use unchanged for boosted or deterministic coherent pilot grids.
    """
    _validate_sizes(layout.n_fft, cp_len)
    if not np.isfinite(ebn0_db):
        raise ValueError('finite Eb/N0 required')
    eb = len(layout.active_tones)/(2*len(layout.data_tones)) * (1+cp_len/layout.n_fft)
    return eb * 10**(-float(ebn0_db)/10)
