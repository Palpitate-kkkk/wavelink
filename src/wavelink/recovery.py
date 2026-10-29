"""Residual hysteresis and causal re-calibration after support warnings."""
from time import perf_counter_ns
import numpy as np
from .guard import SupportGuard
from .tracking import TriggeredPilotLMMSE


class RecoveringSupportGuard(SupportGuard):
    """Release fallback only after consecutive low-residual checks.

    Entry stays at the frozen guard threshold (4 by default). Exit uses a
    lower residual threshold and consecutive checks, then re-calibrates an
    L=13 primary on the latest 64 observed frames. All state changes happen
    after scoring and apply only to the next frame. Low residual does not
    certify the absence of an out-of-support channel.
    """
    def __init__(self, calibration, layout, length, cp_len, threshold=4.,
                 exit_threshold=1.5, release_checks=8, **kwargs):
        if isinstance(release_checks,bool) or not isinstance(release_checks,(int,np.integer)) or release_checks<1:
            raise ValueError('release_checks must be a positive integer')
        if isinstance(exit_threshold,bool) or not np.isfinite(exit_threshold) or not 0<exit_threshold<threshold:
            raise ValueError('need 0 < exit threshold < entry threshold')
        super().__init__(calibration,layout,length,cp_len,threshold=threshold,**kwargs)
        self.length=length;self.exit_threshold=float(exit_threshold)
        self.release_checks=int(release_checks);self.low_checks=0;self.release_log=[]

    def advance(self):
        was_alarmed=self.alarmed
        updated=super().advance()
        if not self.alarmed or not was_alarmed:
            self.low_checks=0
            return updated
        if self.frames_observed%self.check_every:return updated
        started=perf_counter_ns()
        residual,noise=np.mean(self.moments,axis=0)
        score=float(residual/max(self.residual_rank*noise/self.symbols,1e-12))
        self.low_checks=self.low_checks+1 if score<self.exit_threshold else 0
        self.monitor_ns+=perf_counter_ns()-started
        if self.low_checks<self.release_checks:return updated
        # Refit from observed raw history, not from a frozen pre-alarm model.
        started=perf_counter_ns()
        primary=TriggeredPilotLMMSE(np.asarray(self.history),self.layout,self.length,threshold=.75)
        self.update_ns+=perf_counter_ns()-started
        self.fit_frame_visits+=len(self.history)
        self.feature_frames+=primary.feature_frames
        self.primary=primary;self.model=primary.model;self.statistics=primary.statistics
        self.history=primary.history;self.alarmed=False;self.low_checks=0
        # At most one model is effective on the next frame; a refresh made
        # by the parent on this same frame is superseded, but its time stays.
        if updated:
            self.update_count-=1;self.trigger_log.pop()
        self.update_count+=1;self.last_update_after_frame=self.frames_observed-1
        event=dict(after_frame=self.last_update_after_frame,score=score,noise_score=0.,pdp_score=0.,event_kind='release')
        self.trigger_log.append(event);self.release_log.append(event.copy())
        return True
