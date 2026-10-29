"""Causal covariance-innovation trigger for pilot-only LMMSE adaptation."""
from collections import deque
from time import perf_counter_ns
import numpy as np
from .tracking import TriggeredPilotLMMSE
from .estimation import TapDomainLMMSE


class InnovationTriggeredLMMSE(TriggeredPilotLMMSE):
    """Add a noise-normalized pilot-mean residual to the frozen trigger.

    For C=Fp diag(p) Fp^H and lambda=Nmodel/T, B=lambda(C+lambda I)^-1.
    Score=mean_recent ||B zbar||² / (||B Fp sqrt(p)||F² + Nrecent/T ||B||F²).
    This is an observation-only heuristic, NOT a calibrated significance test.
    A shared cooldown covers BOTH trigger mechanisms. Updates apply next frame.
    """
    def __init__(self,calibration,layout,length,innovation_threshold=2.,**kwargs):
        if isinstance(innovation_threshold,bool) or not np.isfinite(innovation_threshold) or innovation_threshold<=0:
            raise ValueError('positive finite innovation threshold required')
        super().__init__(calibration,layout,length,threshold=.75,**kwargs)
        started=perf_counter_ns()
        monitor=self.monitor_moments.values.maxlen
        self.pilot_means=deque((z.mean(axis=0).copy() for z in np.asarray(calibration)[-monitor:]),maxlen=monitor)
        self.innovation_threshold=float(innovation_threshold);self.innovation_checks=0
        self.feature_frames+=len(self.pilot_means)
        self._refresh_probe()
        self.initial_fit_ns+=perf_counter_ns()-started

    def _refresh_probe(self):
        fp=self.model.pilot_design;p=self.statistics.powers
        covariance=(fp*p)@fp.conj().T
        lam=self.statistics.noise_variance/self.symbols
        self.innovation_matrix=lam*np.linalg.solve(covariance+lam*np.eye(len(fp)),np.eye(len(fp)))
        self.innovation_signal=float(np.sum(abs((self.innovation_matrix@fp)*np.sqrt(p))**2))
        self.innovation_noise_gain=float(np.sum(abs(self.innovation_matrix)**2))

    def _innovation_score(self,noise):
        z=np.asarray(self.pilot_means)
        residual=float(np.mean(np.sum(abs(z@self.innovation_matrix.T)**2,axis=1)))
        expected=self.innovation_signal+max(float(noise),1e-12)/self.symbols*self.innovation_noise_gain
        return residual/max(expected,1e-12)

    def advance(self):
        if self._pending is None:raise RuntimeError('estimate a frame before advance')
        started=perf_counter_ns()
        self.pilot_means.append(self._pending.mean(axis=0).copy())
        self.feature_frames+=1;self.feature_ns+=perf_counter_ns()-started
        # Parent uses the same cached fitting window; probe still corresponds
        # to the model that scored this frame even if parent rebuilds below.
        updated=super().advance()
        if self.frames_observed%self.check_every:return updated
        started=perf_counter_ns()
        recent=self._statistics(self.monitor_moments)
        score=self._innovation_score(recent.noise_variance)
        self.innovation_checks+=1;self.monitor_checks+=1
        self.monitor_ns+=perf_counter_ns()-started
        cooldown_ok=self.last_trigger_observed is None or self.frames_observed-self.last_trigger_observed>=self.cooldown
        if not updated and score>self.innovation_threshold and cooldown_ok:
            started=perf_counter_ns()
            fit=self._statistics(self.fit_moments)
            self.model=TapDomainLMMSE(self.layout,fit.powers,fit.noise_variance)
            self.statistics=fit;self.update_count+=1
            self.last_trigger_observed=self.frames_observed
            self.last_update_after_frame=self.frames_observed-1
            self.update_ns+=perf_counter_ns()-started
            self.trigger_log.append(dict(after_frame=self.last_update_after_frame,score=score,noise_score=0.,pdp_score=0.))
            updated=True;reason='innovation'
        else:reason='both' if score>self.innovation_threshold else 'statistics'
        if updated:
            self.trigger_log[-1].update(event_kind=reason,innovation_score=score)
            started=perf_counter_ns();self._refresh_probe();self.update_ns+=perf_counter_ns()-started
        return updated
