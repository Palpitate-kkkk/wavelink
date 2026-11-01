"""Explicit signed-delay LS priors; negative delays are reference coordinates."""
import numpy as np
from .estimation import TapDomainLS


class SignedTapDomainLS(TapDomainLS):
    """Fit a declared signed support using pilots only, without support selection.

    Delay labels lie in [-N//2, (N-1)//2]; their periodic DFT meaning is
    explicit. Negative labels do not imply a physically noncausal transmitter.
    Inherited estimate/expected_nmse retain the original unit-pilot contract.
    """
    def __init__(self, layout, delays):
        try:
            support = tuple(delays)
        except TypeError as exc:
            raise ValueError('integer delay sequence required') from exc
        if (not support or any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) for d in support)
                or tuple(sorted(set(support))) != support):
            raise ValueError('strictly increasing integer delays required')
        if any(d < -(layout.n_fft//2) or d > (layout.n_fft-1)//2 for d in support):
            raise ValueError('signed delay outside unique FFT period')
        if len(support) > len(layout.pilot_tones):
            raise ValueError('more unknown taps than pilot observations')
        self.layout, self.delays, self.length = layout, tuple(map(int, support)), len(support)
        self.pilot_design = np.exp(-2j*np.pi*np.outer(layout.pilot_tones, support)/layout.n_fft)
        self.data_design = np.exp(-2j*np.pi*np.outer(layout.data_tones, support)/layout.n_fft)
        u, singular, vh = np.linalg.svd(self.pilot_design, full_matrices=False)
        self.rank = int(np.count_nonzero(singular > singular[0]*1e-12))
        self.condition = float(singular[0]/singular[-1])
        if self.rank != self.length or self.condition > 1e8:
            raise ValueError('pilot geometry cannot stably identify the assumed taps')
        inverse = (vh.conj().T/singular) @ u.conj().T
        self.mapping = self.data_design @ inverse
        self.noise_gain = float(np.sum(abs(self.mapping)**2))
