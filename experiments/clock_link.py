"""Paired offline clock correction on an independent OFDM fixture."""
import json
from fractions import Fraction

import numpy as np

from experiments.clock_audit import synthesize_clock
from wavelink import ofdm_modulate, ofdm_demodulate, qpsk_modulate, qpsk_demodulate
from wavelink.clock_estimation import estimate_clock
from wavelink.rate_conversion import resample_complex


def episode(ppm, snr, seed, *, diagnostics=None):
    rng = np.random.default_rng(seed)
    starts = np.arange(1000, 82000, 10000)
    marker = np.fft.ifft(np.r_[0, rng.choice([-1., 1.], 39), np.zeros(88)])
    marker *= np.hanning(128)
    marker /= np.sqrt(np.mean(abs(marker)**2))
    tones = np.r_[np.arange(1, 24), np.arange(41, 64)]
    bits = rng.integers(0, 2, (len(starts), 20, 2*len(tones)), dtype=np.uint8)
    grid = np.zeros((len(starts), 20, 64), complex)
    grid[:, :, tones] = qpsk_modulate(bits.ravel()).reshape(len(starts), 20, -1)
    tx = np.zeros(84000, complex)
    for i, start in enumerate(starts):
        if i != 4:
            tx[start:start+128] = marker
        tx[start+256:start+1856] = ofdm_modulate(grid[i], cp_len=16)
    h = np.array([1, 0, .32+.2j, 0, 0, -.18j])
    h /= np.linalg.norm(h)
    ratio = Fraction(1000000+ppm, 1000000)
    y = synthesize_clock(np.convolve(tx, h)[:len(tx)], ratio.numerator, ratio.denominator)
    # CFO in radians per observed sample. Genie removal is shared by all decoding arms.
    carrier = np.exp(2j*np.pi*.12*np.arange(len(y))/64)
    noise = (rng.normal(size=len(y))+1j*rng.normal(size=len(y)))/np.sqrt(2)
    y = y*carrier+noise*10**(-snr/20)
    try:
        estimate = estimate_clock(y, marker, starts, radius=32)
    except ValueError as exc:
        if str(exc) != 'insufficient accepted training markers':
            raise
        estimate = None
    corrected_input = y*carrier.conj()
    if diagnostics is not None:
        diagnostics(corrected_input, marker, starts)
    estimated = None if estimate is None else Fraction(
        1_000_000_000, round(estimate['rate_ratio']*1_000_000_000)
    ).limit_denominator(100000)
    response = np.fft.fft(h, 64)[tones]
    rows = []
    for name, correction in [('raw', Fraction(1)), ('estimated', estimated), ('oracle', 1/ratio)]:
        errors = erased = 0
        if correction is None:
            erased = bits.size
        else:
            z, _ = resample_complex(corrected_input, correction.numerator, correction.denominator)
            for i, start in enumerate(starts):
                frame = z[start+256:start+1856]
                if len(frame) != 1600:
                    erased += bits[i].size
                    continue
                grid_rx = ofdm_demodulate(frame, n_fft=64, cp_len=16)
                decided = qpsk_demodulate((grid_rx[:, tones]/response).ravel())
                errors += int(np.count_nonzero(decided != bits[i].ravel()))
        rows.append(dict(seed=seed, snr_db=snr, true_ppm=ppm, method=name,
                         estimate_ppm=None if estimate is None else estimate['ppm'],
                         applied_ratio=None if correction is None else str(correction),
                         errors=errors, erased=erased, offered=bits.size))
    return rows


if __name__ == '__main__':
    print(json.dumps([row for s in (20, 5, -8) for p in (-200, 0, 200)
                      for seed in range(620101, 620105)
                      for row in episode(p, s, seed)], indent=2))
