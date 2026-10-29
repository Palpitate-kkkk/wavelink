"""Pilot-only method-of-moments calibration for block-static channels."""
from dataclasses import dataclass
import numpy as np
from .estimation import TapDomainLS


@dataclass(frozen=True)
class PilotStatistics:
    powers: np.ndarray
    raw_powers: np.ndarray
    noise_variance: float
    frames: int
    symbols: int
    clipped_taps: int


def learn_pilot_statistics(observations, layout, length):
    """Estimate N0 and uncorrelated-tap powers from z=Yp/Xp only.

    Shape is (independent frames, symbols within a static frame, pilots).
    Known unit-modulus pilots must have been divided out. Adjacent disjoint
    symbol pairs cancel the static channel. Mean pilot observations are fitted
    to contiguous taps 0..length-1; noise power is subtracted before clipping.
    No true channel, PDP, SNR, or noise variance enters this interface.
    """
    z = np.asarray(observations, complex)
    if z.ndim != 3 or z.shape[0] < 2 or z.shape[1] < 2 or z.shape[2] != len(layout.pilot_tones) or not np.all(np.isfinite(z)):
        raise ValueError('finite observations with >=2 frames and >=2 symbols required')
    design = TapDomainLS(layout,length).pilot_design
    inverse = np.linalg.pinv(design, rcond=1e-12)
    pairs = z.shape[1]//2
    differences = z[:,1:2*pairs:2]-z[:,0:2*pairs:2]
    noise = max(float(np.mean(abs(differences)**2)/2),1e-12)
    taps = z.mean(axis=1) @ inverse.T
    raw = np.mean(abs(taps)**2,axis=0)-noise/z.shape[1]*np.sum(abs(inverse)**2,axis=1)
    # A fixed numerical floor only; never normalize to the true total power.
    powers = np.maximum(raw,1e-12)
    return PilotStatistics(powers,raw,noise,len(z),z.shape[1],int(np.count_nonzero(raw<0)))
