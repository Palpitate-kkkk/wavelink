"""Bounded-slot acquisition from a fixed repeated header and unique marker.

This is a sample-rate, integer-delay baseband front end, not an unbounded
streaming packet detector. The combined score is a heuristic, not a p-value.
"""
from dataclasses import dataclass
import numpy as np
from .baseband import qpsk_modulate, ofdm_modulate


@dataclass(frozen=True)
class Acquisition:
    start: int
    epsilon: float
    score: float
    repeat_correlation: float
    marker_fraction: float
    accepted: bool


def marker_grid():
    """Fixed public training code; independent of every experiment seed."""
    rng = np.random.default_rng(180031)
    return qpsk_modulate(rng.integers(0, 2, 128, dtype=np.uint8))


def acquisition_header():
    rng = np.random.default_rng(180032)
    grid = np.zeros(64, complex)
    grid[::2] = np.sqrt(2)*qpsk_modulate(rng.integers(0, 2, 64, dtype=np.uint8))
    half = np.fft.ifft(grid, norm='ortho')[:32]
    repeat = np.tile(half, 4)
    return np.r_[repeat[-16:], repeat, ofdm_modulate(marker_grid()[None, :], 16)]


def acquire(samples, threshold, search_count=64):
    """Search starts 0..search_count-1 and integer CFO shifts -4,-2,0,2,4.

    Reads only samples[:search_count-1+224]. No true start, CFO, PDP, noise
    variance, pilot bits, or payload samples beyond that fixed prefix enter.
    Contiguous causal taps 0..16 are assumed only to score the unique marker.
    Threshold is locked externally on development signal/noise examples.
    """
    r = np.asarray(samples, complex)
    if isinstance(search_count, bool) or not isinstance(search_count, (int, np.integer)) or search_count < 1:
        raise ValueError('positive integer search_count required')
    if isinstance(threshold, bool) or not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('finite threshold in [0,1] required')
    needed = int(search_count)-1+224
    if r.ndim != 1 or len(r) < needed or not np.all(np.isfinite(r[:needed])):
        raise ValueError('finite search prefix covering every complete header required')
    r = r[:needed]
    repeated = np.lib.stride_tricks.sliding_window_view(r[16:16+search_count-1+128], 128)
    a, b = repeated[:, :96], repeated[:, 32:]
    product = np.sum(a.conj()*b, axis=1)
    scale = np.sqrt(np.sum(abs(a)**2, axis=1)*np.sum(abs(b)**2, axis=1))
    correlation = np.divide(abs(product), scale, out=np.zeros_like(scale), where=scale > 0)
    correlation = np.clip(correlation, 0., 1.)
    coarse = np.angle(product)/np.pi
    marker = np.lib.stride_tricks.sliding_window_view(r[160:], 64)
    energy = np.sum(abs(marker)**2, axis=1)
    corrected = np.fft.fft(marker*np.exp(-2j*np.pi*coarse[:, None]*np.arange(64)/64), axis=1, norm='ortho')
    reference = marker_grid()
    scores, fractions = [], []
    integers = (-4, -2, 0, 2, 4)
    for integer in integers:
        taps = np.fft.ifft(np.roll(corrected, -integer, axis=1)/reference, axis=1)
        projected = 64*np.sum(abs(taps[:, :17])**2, axis=1)
        fraction = np.divide(projected, energy, out=np.zeros_like(energy), where=energy > 0)
        fraction = np.clip(fraction, 0., 1.)
        fractions.append(fraction)
        scores.append(correlation**2*fraction)
    scores = np.asarray(scores)
    ki, start = np.unravel_index(np.argmax(scores), scores.shape)
    score = float(scores[ki, start])
    # Reject an exactly empty input even at threshold zero.
    return Acquisition(int(start), float(coarse[start]+integers[ki]), score,
                       float(correlation[start]), float(fractions[ki][start]), bool(score > 0 and score >= threshold))
