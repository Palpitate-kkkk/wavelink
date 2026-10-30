"""One bounded follow-up fit after an observation-triggered innovation update."""
from time import perf_counter_ns
import numpy as np
from .innovation import InnovationTriggeredLMMSE
from .estimation import TapDomainLMMSE


class FollowupPilotLMMSE(InnovationTriggeredLMMSE):
    """Schedule one delayed fit; newer observation-driven fits supersede it.

    Only innovation/both events arm a plan. A newer statistics-only event
    cancels it, a newer innovation event replaces it, and a follow-up never
    re-arms itself. All mechanisms share cooldown and apply NEXT frame.
    """
    def __init__(self,calibration,layout,length,refit_delay=64,**kwargs):
        cooldown=kwargs.get('cooldown',32);check=kwargs.get('check_every',8)
        if any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1 for v in (cooldown,check)):
            raise ValueError('positive integer cooldown and check_every required')
        if isinstance(refit_delay,bool) or not isinstance(refit_delay,(int,np.integer)) or refit_delay<cooldown or refit_delay%check:
            raise ValueError('refit_delay must be an integer >=cooldown and a multiple of check_every')
        super().__init__(calibration,layout,length,**kwargs)
        self.refit_delay=int(refit_delay);self.pending_refit_after_frame=None;self.pending_origin=None
        self.schedule_log=[];self.followup_count=0;self.cancelled_plans=0;self.armed_plans=0

    def _record_plan(self,action,frame):
        self.schedule_log.append(dict(action=action,after_frame=frame,due_after_frame=self.pending_refit_after_frame,origin_after_frame=self.pending_origin))

    def advance(self):
        updated=super().advance()
        started=perf_counter_ns();frame=self.frames_observed-1
        if updated:
            if self.pending_refit_after_frame is not None:
                self._record_plan('cancel',frame);self.cancelled_plans+=1
                self.pending_refit_after_frame=None;self.pending_origin=None
            if self.trigger_log[-1]['event_kind'] in ('innovation','both'):
                self.pending_origin=frame;self.pending_refit_after_frame=frame+self.refit_delay
                self._record_plan('arm',frame);self.armed_plans+=1
            self.monitor_ns+=perf_counter_ns()-started
            return True
        due=self.pending_refit_after_frame is not None and frame>=self.pending_refit_after_frame
        self.monitor_ns+=perf_counter_ns()-started
        if not due:return False
        # An ordinary rebuild above would have cancelled/replaced this plan.
        # Validate cooldown rather than bypassing it for a scheduled fit.
        if self.last_trigger_observed is not None and self.frames_observed-self.last_trigger_observed<self.cooldown:
            raise AssertionError('scheduled fit violates shared cooldown')
        started=perf_counter_ns();fit=self._statistics(self.fit_moments)
        self.model=TapDomainLMMSE(self.layout,fit.powers,fit.noise_variance);self.statistics=fit
        self.update_count+=1;self.followup_count+=1
        self.last_trigger_observed=self.frames_observed;self.last_update_after_frame=frame
        self._refresh_probe()
        self.trigger_log.append(dict(after_frame=frame,score=0.,noise_score=0.,pdp_score=0.,event_kind='followup',innovation_score=0.))
        self._record_plan('execute',frame)
        self.pending_refit_after_frame=None;self.pending_origin=None
        self.update_ns+=perf_counter_ns()-started
        return True
