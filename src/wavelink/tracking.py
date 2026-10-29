"""Causal pilot-statistics updates: estimate current frame, then advance."""
from collections import deque
from time import perf_counter_ns
import numpy as np
from .estimation import TapDomainLMMSE
from .statistics import learn_pilot_statistics


class CausalPilotLMMSE:
    """A frozen or rolling-window model with a strict estimate/advance order.

    Every policy starts from ALL calibration frames, then updates using only
    the latest `window` observed frames after scoring each current frame.
    No change point, true PDP, SNR or true noise is accepted by this API.
    """
    def __init__(self, calibration, layout, length, window=64, update_every=None):
        if isinstance(window,bool) or not isinstance(window,(int,np.integer)) or window<2:
            raise ValueError('window must be an integer >=2')
        if update_every is not None and (isinstance(update_every,bool) or not isinstance(update_every,(int,np.integer)) or update_every<1):
            raise ValueError('update_every must be None or a positive integer')
        z=np.asarray(calibration,complex)
        self.layout,self.length=layout,length
        started=perf_counter_ns()
        self.statistics=learn_pilot_statistics(z,layout,length)
        if window>len(z):raise ValueError('window cannot exceed calibration pool')
        self.model=TapDomainLMMSE(layout,self.statistics.powers,self.statistics.noise_variance)
        self.initial_fit_ns=perf_counter_ns()-started
        self.history=deque((x.copy() for x in z[-window:]),maxlen=int(window))
        self.symbols=z.shape[1]
        self.update_every=update_every
        self.frames_observed=0
        self.update_count=0
        self.update_ns=0
        self.fit_frame_visits=0
        self.last_update_after_frame=-1
        self._pending=None

    def estimate(self, received, pilots):
        if self._pending is not None:
            raise RuntimeError('advance after scoring before estimating the next frame')
        result=self.model.estimate(received,pilots)
        if len(received)!=self.symbols:
            raise ValueError('frame symbol count must match calibration')
        self._pending=(np.asarray(received)[:,self.layout.bins(self.layout.pilot_tones)]/pilots).copy()
        return result

    def advance(self):
        """Consume the scored frame's pilots; any new model applies NEXT frame."""
        if self._pending is None:
            raise RuntimeError('estimate a frame before advance')
        self.history.append(self._pending)
        self._pending=None
        self.frames_observed+=1
        if self.update_every is None or self.frames_observed%self.update_every:
            return False
        started=perf_counter_ns()
        fit=learn_pilot_statistics(np.asarray(self.history),self.layout,self.length)
        model=TapDomainLMMSE(self.layout,fit.powers,fit.noise_variance)
        self.update_ns+=perf_counter_ns()-started
        self.fit_frame_visits+=len(self.history)
        self.statistics,self.model=fit,model
        self.update_count+=1
        self.last_update_after_frame=self.frames_observed-1
        return True
