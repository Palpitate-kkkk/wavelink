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


class _MomentWindow:
    """Rolling sum of per-frame LS tap powers and pilot-difference noise."""
    def __init__(self, size, values):
        self.values=deque(maxlen=int(size))
        self.total=np.zeros_like(values[0])
        for value in values[-size:]:self.add(value)

    def add(self, value):
        if len(self.values)==self.values.maxlen:self.total-=self.values[0]
        self.values.append(value.copy());self.total+=value

    def mean(self):
        return self.total/len(self.values)


class TriggeredPilotLMMSE(CausalPilotLMMSE):
    """Observation-only heuristic with cached moments and a cooldown.

    Monitoring happens AFTER scoring. The score is max(absolute log noise
    ratio, relative L1 PDP discrepancy). Thresholds must be set externally
    using development data. This is not a calibrated hypothesis test.
    """
    def __init__(self, calibration, layout, length, window=64, monitor_window=16,
                 check_every=8, cooldown=32, threshold=.5):
        for name,value in [('monitor_window',monitor_window),('check_every',check_every),('cooldown',cooldown)]:
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:raise ValueError(name+' must be a positive integer')
        if monitor_window<2 or monitor_window>window:raise ValueError('need 2 <= monitor_window <= fit window')
        if isinstance(threshold,bool) or not np.isfinite(threshold) or threshold<=0:raise ValueError('positive finite threshold required')
        super().__init__(calibration,layout,length,window,None)
        started=perf_counter_ns()
        from .estimation import TapDomainLS
        self.inverse=np.linalg.pinv(TapDomainLS(layout,length).pilot_design,rcond=1e-12)
        self.noise_diagonal=np.sum(abs(self.inverse)**2,axis=1)
        features=[self._feature(z) for z in np.asarray(calibration)]
        self.fit_moments=_MomentWindow(window,features)
        self.monitor_moments=_MomentWindow(monitor_window,features)
        self.initial_fit_ns+=perf_counter_ns()-started
        self.check_every,self.cooldown,self.threshold=check_every,cooldown,float(threshold)
        self.monitor_checks=0;self.feature_frames=len(features)
        self.feature_ns=0;self.monitor_ns=0
        self.last_trigger_observed=None
        self.trigger_log=[]

    def _feature(self,z):
        pairs=len(z)//2
        noise=float(np.mean(abs(z[1:2*pairs:2]-z[0:2*pairs:2])**2)/2)
        taps=self.inverse @ z.mean(axis=0)
        return np.r_[abs(taps)**2,noise]

    def _statistics(self, moments):
        from .statistics import PilotStatistics
        mean=moments.mean();noise=max(float(mean[-1]),1e-12)
        raw=mean[:-1]-noise/self.symbols*self.noise_diagonal
        return PilotStatistics(np.maximum(raw,1e-12),raw,noise,len(moments.values),self.symbols,int(np.count_nonzero(raw<0)))

    def advance(self):
        if self._pending is None:raise RuntimeError('estimate a frame before advance')
        started=perf_counter_ns()
        feature=self._feature(self._pending)
        self.history.append(self._pending);self._pending=None
        self.fit_moments.add(feature);self.monitor_moments.add(feature)
        self.frames_observed+=1;self.feature_frames+=1
        self.feature_ns+=perf_counter_ns()-started
        if self.frames_observed%self.check_every:return False
        started=perf_counter_ns()
        recent=self._statistics(self.monitor_moments)
        noise_score=abs(float(np.log(recent.noise_variance/self.statistics.noise_variance)))
        pdp_score=float(np.sum(abs(recent.powers-self.statistics.powers))/max(float(self.statistics.powers.sum()),1e-12))
        score=max(noise_score,pdp_score);self.monitor_checks+=1
        self.monitor_ns+=perf_counter_ns()-started
        if score<=self.threshold or (self.last_trigger_observed is not None and self.frames_observed-self.last_trigger_observed<self.cooldown):return False
        started=perf_counter_ns()
        fit=self._statistics(self.fit_moments)
        self.model=TapDomainLMMSE(self.layout,fit.powers,fit.noise_variance)
        self.statistics=fit;self.update_count+=1
        self.last_trigger_observed=self.frames_observed
        self.last_update_after_frame=self.frames_observed-1
        self.update_ns+=perf_counter_ns()-started
        # Cached sums: no full-history re-scan at a trigger. Feature work is
        # reported separately; zero visits does not mean zero computation.
        self.trigger_log.append(dict(after_frame=self.last_update_after_frame,score=score,noise_score=noise_score,pdp_score=pdp_score))
        return True


def make_tracker(spec,calibration,layout,length):
    if spec.get('kind')=='triggered':
        return TriggeredPilotLMMSE(calibration,layout,length,window=spec['window'],monitor_window=spec['monitor_window'],
            check_every=spec['check_every'],cooldown=spec['cooldown'],threshold=spec['threshold'])
    if spec.get('kind','periodic')!='periodic':raise ValueError('unknown tracker kind')
    return CausalPilotLMMSE(calibration,layout,length,spec['window'],spec['update_every'])
