"""Change the statistical phase reference without moving received FFT windows."""
import numpy as np
from .slot_tracking import SlotTrackedLMMSE


def rotate_grid_reference(received, shift):
    """Unitary diagonal phase rotation; NOT a time-domain window extraction."""
    y = np.asarray(received, complex)
    if y.ndim != 2 or not all(y.shape) or not np.all(np.isfinite(y)):
        raise ValueError('finite nonempty FFT grid required')
    if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)):
        raise ValueError('integer reference shift required')
    n = y.shape[1]
    phase = np.exp(2j*np.pi*np.arange(n)*(int(shift) % n)/n)
    return y*phase


class ReferenceTrackedLMMSE(SlotTrackedLMMSE):
    """Train in a marker-selected reference; return estimates in the RAW one.

    Current header supplies shift, not true timing. A wrong shift can put
    channel energy outside the child's causal17-tap model even though the
    physical receive window never changes. No synchronization guarantee.
    """
    def step(self, received=None, pilots=None, shift=0, final=False):
        if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)) or not 0 <= shift <= 16:
            raise ValueError('integer marker shift in0..16 required')
        if received is None:
            if shift:
                raise ValueError('missing input has no marker shift')
            return super().step(None, pilots, final=final)
        canonical = rotate_grid_reference(received, shift)
        estimate = super().step(canonical, pilots, final=final)
        phase = np.exp(-2j*np.pi*self.layout.bins(self.layout.data_tones)*int(shift)/self.layout.n_fft)
        self.last['reference_shift'] = int(shift)
        return estimate*phase
