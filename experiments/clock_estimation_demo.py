"""Controlled scheduled-marker study; the estimator never receives true ppm."""
import json
from math import gcd
import numpy as np
from experiments.clock_audit import synthesize_clock
from wavelink.clock_estimation import estimate_clock


def fixture(ppm, seed, snr, *, missing=False):
    rng = np.random.default_rng(seed)
    spectrum = np.zeros(128, complex)
    spectrum[1:40] = rng.choice([-1, 1], 39)+1j*rng.choice([-1, 1], 39)
    marker = np.fft.ifft(spectrum)*np.hanning(128)
    marker /= np.sqrt(np.mean(abs(marker)**2))
    starts = np.arange(1000, 82000, 10000)
    x = np.zeros(84000, complex)
    for i, start in enumerate(starts):
        if not (missing and i == 4):
            x[start+5:start+5+len(marker)] = marker
    g = gcd(1000000+ppm, 1000000)
    y = synthesize_clock(x, (1000000+ppm)//g, 1000000//g)
    noise = (rng.normal(size=len(y))+1j*rng.normal(size=len(y)))/np.sqrt(2)
    return y+noise*10**(-snr/20), marker, starts


def run():
    rows = []
    for snr in (20, 5, -8):
        for ppm in (-200, -100, 0, 100, 200):
            for seed in range(440101, 440105):
                y, marker, starts = fixture(ppm, seed, snr, missing=True)
                try:
                    result = estimate_clock(y, marker, starts, radius=32)
                except ValueError as exc:
                    if str(exc) != 'insufficient accepted training markers':
                        raise
                    result = None
                rows.append(dict(snr_db=snr, true_ppm=ppm, seed=seed,
                                 estimate_ppm=None if result is None else result['ppm'],
                                 accepted=0 if result is None else len(result['anchors'])))
    return rows


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
