"""Observation-clock tracking behind a lossy acquisition front end."""
from collections import deque
import numpy as np
from .estimation import TapDomainLS
from .followup import FollowupPilotLMMSE
from .tracking import CausalPilotLMMSE


class SlotTrackedLMMSE:
    """Bootstrap with 64 admitted bursts, otherwise use per-burst LS.

    The physical slot clock always advances. The frozen child's clock advances
    only for admitted observations AFTER bootstrap. Missing/rejected slots do
    not insert zeros, expire its cooldown, or execute a scheduled refit. A
    gate filters statistics only; every acquired burst is still estimated.
    """
    def __init__(self, layout, gate=None, adaptive=True):
        if gate is not None and (isinstance(gate, bool) or not np.isscalar(gate) or not np.isfinite(gate) or not 0 < gate <= 1):
            raise ValueError('gate must be None or finite in (0,1]')
        if not isinstance(adaptive, bool):
            raise ValueError('adaptive must be bool')
        self.layout, self.gate, self.adaptive = layout, gate, adaptive
        self.fallback = TapDomainLS(layout, 17)
        fp = self.fallback.pilot_design
        self.projector = fp @ np.linalg.pinv(fp, rcond=1e-12)
        self.calibration = []
        self.history_slots = deque(maxlen=64)
        self.tracker = None
        self.slots = 0
        self.bootstrap_slot = None
        self.captured = self.consumed = self.missing = self.gate_rejected = 0
        self.maximum_history_span = 0
        self.events = []
        self.last = None
        self.closed = False

    def step(self, received=None, pilots=None, final=False):
        """Return a current-prior estimate, then ingest; final forbids fitting.

        None denotes an unavailable or acquisition-rejected slot. No scorer
        truth, change point, true CFO, PDP or noise is accepted by this API.
        """
        if self.closed:
            raise RuntimeError('receiver already finalized')
        if not isinstance(final, bool):
            raise ValueError('final must be bool')
        slot = self.slots
        before = self.tracker is not None
        ratio = None
        admitted = False
        if received is None:
            if pilots is not None:
                raise ValueError('pilots require a received grid')
            result = None
        else:
            y, p = np.asarray(received, complex), np.asarray(pilots, complex)
            if y.shape != (20, self.layout.n_fft) or p.shape != (20, len(self.layout.pilot_tones)):
                raise ValueError('20-symbol frame and matching pilots required')
            if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)) or not np.allclose(abs(p), 1., rtol=0, atol=1e-12):
                raise ValueError('finite grid and unit-modulus pilots required')
            z = y[:, self.layout.bins(self.layout.pilot_tones)]/p
            residual = z-z@self.projector.T
            energy = float(np.sum(abs(z)**2))
            ratio = float(np.clip(np.sum(abs(residual)**2)/energy, 0., 1.)) if energy > 0 else 1.
            admitted = self.gate is None or ratio <= self.gate
            # Current prior is fixed before any observation ingestion. A final
            # or gated-out burst uses model.estimate without child pending data.
            if before and admitted and not final:
                result = self.tracker.estimate(y, p)
            else:
                model = self.tracker.model if before else self.fallback
                result = model.estimate(y, p)
        consumed = received is not None and admitted and not final
        updated = False
        if consumed:
            if before:
                updated = self.tracker.advance()
            else:
                self.calibration.append(z.copy())
                if len(self.calibration) == 64:
                    calibration = np.asarray(self.calibration)
                    self.tracker = (FollowupPilotLMMSE(calibration, self.layout, 17, innovation_threshold=2., refit_delay=64)
                                    if self.adaptive else CausalPilotLMMSE(calibration, self.layout, 17, 64, None))
                    self.bootstrap_slot = slot
                    self.events.append(dict(after_slot=slot, applies_from_slot=slot+1, kind='bootstrap', observation_index=-1))
                    self.calibration.clear()
            self.history_slots.append(slot)
            self.consumed += 1
            span = self.history_slots[-1]-self.history_slots[0]+1
            self.maximum_history_span = max(self.maximum_history_span, span)
            if updated:
                kind = self.tracker.trigger_log[-1]['event_kind']
                self.events.append(dict(after_slot=slot, applies_from_slot=slot+1, kind=kind,
                                        observation_index=self.tracker.last_update_after_frame))
        self.slots += 1
        self.captured += int(received is not None)
        self.missing += int(received is None)
        self.gate_rejected += int(received is not None and not admitted)
        self.last = dict(slot=slot, ready_before=before, ready_after=self.tracker is not None,
                         captured=received is not None, pilot_ratio=ratio, statistics_admissible=admitted,
                         statistics_consumed=consumed, updated=updated,
                         observation_clock=self.tracker.frames_observed if self.tracker else 0,
                         history_span=self.history_slots[-1]-self.history_slots[0]+1 if self.history_slots else 0)
        self.closed = final
        return result
