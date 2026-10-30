"""Incremental implementation of the existing periodic pilot estimator."""
from time import perf_counter_ns
import numpy as np
from .tracking import CausalPilotLMMSE, _MomentWindow
from .estimation import TapDomainLS, TapDomainLMMSE
from .statistics import PilotStatistics


class CachedPeriodicLMMSE(CausalPilotLMMSE):
    """Same window, clipping and update schedule as CausalPilotLMMSE.

    Cache per-frame sufficient statistics rather than rescan pilot history.
    Floating-point summation order differs; mathematical estimates coincide.
    No trigger, innovation probe or true channel information is used.
    """
    def __init__(self, calibration, layout, length, window=64, update_every=32):
        super().__init__(calibration, layout, length, window, update_every)
        started = perf_counter_ns()
        self.inverse = np.linalg.pinv(TapDomainLS(layout, length).pilot_design, rcond=1e-12)
        self.noise_diagonal = np.sum(abs(self.inverse)**2, axis=1)
        self.moments = _MomentWindow(window, [self._feature(z) for z in self.history])
        self.initial_fit_ns += perf_counter_ns() - started
        self.feature_frames = len(self.history)
        self.feature_ns = 0

    def _feature(self, z):
        pairs = len(z)//2
        noise = float(np.mean(abs(z[1:2*pairs:2] - z[0:2*pairs:2])**2)/2)
        taps = self.inverse @ z.mean(axis=0)
        return np.r_[abs(taps)**2, noise]

    def advance(self):
        if self._pending is None:
            raise RuntimeError('estimate a frame before advance')
        started = perf_counter_ns()
        self.moments.add(self._feature(self._pending))
        self.history.append(self._pending)
        self._pending = None
        self.frames_observed += 1
        self.feature_frames += 1
        self.feature_ns += perf_counter_ns() - started
        if self.update_every is None or self.frames_observed % self.update_every:
            return False
        started = perf_counter_ns()
        mean = self.moments.mean()
        noise = max(float(mean[-1]), 1e-12)
        raw = mean[:-1] - noise/self.symbols*self.noise_diagonal
        fit = PilotStatistics(np.maximum(raw, 1e-12), raw, noise,
                              len(self.moments.values), self.symbols, int(np.count_nonzero(raw < 0)))
        self.model = TapDomainLMMSE(self.layout, fit.powers, fit.noise_variance)
        self.statistics = fit
        self.update_count += 1
        self.last_update_after_frame = self.frames_observed - 1
        self.update_ns += perf_counter_ns() - started
        return True
