"""Pilot-subspace warning with a causal, latched regularized fallback."""
from collections import deque
from time import perf_counter_ns
import numpy as np
from .estimation import TapDomainLS, TapDomainLMMSE
from .statistics import PilotStatistics
from .tracking import TriggeredPilotLMMSE


class SupportGuard:
    """Heuristic warning; no true delay, change time, PDP or noise input.

    Score = sum ||Q^H mean_symbols(z)||^2 / sum ((Np-L)*Nhat/T),
    over recent frames. Q spans the orthogonal complement of the ORIGINAL
    pilot design. Under supported block-static channels its expectation is
    approximately one (a ratio with an estimated denominator). This is not a
    calibrated significance test or a unique diagnosis of delay mismatch.

    An alarm latches until the episode ends. From the NEXT frame use uniform
    ensemble power over delays 0..CP, with pilot-difference noise estimated
    from the monitor window. Only noise is subsequently refreshed. The
    uniform total-power prior is an explicit configured assumption.
    """
    def __init__(self, calibration, layout, length, cp_len, threshold=2.,
                 monitor_window=16, check_every=8, refresh_every=32,
                 fallback_power=1.):
        z=np.asarray(calibration,complex)
        for name,v in [('cp_len',cp_len),('monitor_window',monitor_window),('check_every',check_every),('refresh_every',refresh_every)]:
            if isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1:raise ValueError(name+' must be a positive integer')
        if cp_len>=layout.n_fft or length>cp_len or length>=len(layout.pilot_tones):raise ValueError('need L < Npilots and L <= CP < Nfft')
        if not 2<=monitor_window<=len(z):raise ValueError('monitor window must fit calibration and be >=2')
        for v in (threshold,fallback_power):
            if isinstance(v,bool) or not np.isfinite(v) or v<=0:raise ValueError('positive finite threshold/power required')
        self.primary=TriggeredPilotLMMSE(z,layout,length,threshold=.75)
        self.layout=layout;self.symbols=z.shape[1];self.cp_len=cp_len
        self.threshold=float(threshold);self.check_every=check_every;self.refresh_every=refresh_every
        self.fallback_power=float(fallback_power)
        started=perf_counter_ns()
        fp=TapDomainLS(layout,length).pilot_design
        q=np.linalg.qr(fp,mode='complete')[0]
        self.residual_basis=q[:,length:];self.residual_rank=len(layout.pilot_tones)-length
        self.moments=deque((self._feature(x) for x in z[-monitor_window:]),maxlen=monitor_window)
        self.model=self.primary.model;self.statistics=self.primary.statistics
        self.history=self.primary.history
        self.initial_fit_ns=self.primary.initial_fit_ns+perf_counter_ns()-started
        self.frames_observed=0;self.last_update_after_frame=-1;self.update_count=0
        self.update_ns=0;self.fit_frame_visits=0;self.feature_ns=0;self.monitor_ns=0
        self.monitor_checks=0;self.feature_frames=monitor_window+self.primary.feature_frames;self.alarmed=False;self.alarm_after_frame=None
        self.trigger_log=[];self.warning_log=[];self._pending=None

    def _feature(self,z):
        pairs=len(z)//2
        noise=float(np.mean(abs(z[1:2*pairs:2]-z[:2*pairs:2])**2)/2)
        residual=float(np.sum(abs(self.residual_basis.conj().T@z.mean(axis=0))**2))
        return residual,noise

    def estimate(self,received,pilots):
        if self._pending is not None:raise RuntimeError('advance after scoring before next estimate')
        if len(received)!=self.symbols:raise ValueError('frame symbol count must match calibration')
        result=(self.model.estimate(received,pilots) if self.alarmed else self.primary.estimate(received,pilots))
        z=(np.asarray(received)[:,self.layout.bins(self.layout.pilot_tones)]/pilots).copy()
        self._pending=z
        return result

    def advance(self):
        if self._pending is None:raise RuntimeError('estimate a frame before advance')
        started=perf_counter_ns()
        self.moments.append(self._feature(self._pending));self.feature_frames+=1
        if self.alarmed:self.history.append(self._pending)
        self._pending=None;self.frames_observed+=1
        self.feature_ns+=perf_counter_ns()-started
        updated=False;event_kind='primary';score=0.
        if not self.alarmed:
            before=(self.primary.update_ns,self.primary.feature_ns,self.primary.monitor_ns,self.primary.monitor_checks,self.primary.feature_frames)
            updated=self.primary.advance()
            self.update_ns+=self.primary.update_ns-before[0]
            self.feature_ns+=self.primary.feature_ns-before[1]
            self.monitor_ns+=self.primary.monitor_ns-before[2]
            self.monitor_checks+=self.primary.monitor_checks-before[3]
            self.feature_frames+=self.primary.feature_frames-before[4]
            self.model=self.primary.model;self.statistics=self.primary.statistics
        if self.frames_observed%self.check_every==0:
            started=perf_counter_ns()
            residual,noise=np.mean(self.moments,axis=0)
            score=float(residual/max(self.residual_rank*noise/self.symbols,1e-12))
            self.monitor_checks+=1;self.monitor_ns+=perf_counter_ns()-started
            if not self.alarmed and score>self.threshold:
                self.alarmed=True;self.alarm_after_frame=self.frames_observed-1
                self.warning_log.append(dict(after_frame=self.alarm_after_frame,score=score))
                event_kind='alarm'
        rebuild=self.alarmed and (event_kind=='alarm' or (self.frames_observed-1-self.alarm_after_frame)%self.refresh_every==0)
        if rebuild:
            started=perf_counter_ns()
            noise=max(float(np.mean(self.moments,axis=0)[1]),1e-12)
            powers=np.full(self.cp_len+1,self.fallback_power/(self.cp_len+1))
            self.model=TapDomainLMMSE(self.layout,powers,noise)
            self.statistics=PilotStatistics(powers,powers.copy(),noise,len(self.moments),self.symbols,0)
            self.update_ns+=perf_counter_ns()-started
            updated=True
            if event_kind!='alarm':event_kind='fallback_refresh'
        if updated:
            self.update_count+=1;self.last_update_after_frame=self.frames_observed-1
            self.trigger_log.append(dict(after_frame=self.last_update_after_frame,score=score,noise_score=0.,pdp_score=0.,event_kind=event_kind))
        return updated
