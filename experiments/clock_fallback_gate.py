"""Calibrate a fixed-window marker gate on separate seeds from evaluation."""
import json

import numpy as np

from experiments.clock_link import episode


def quality(samples, marker, starts):
    scores = []
    energy = np.linalg.norm(marker)
    for start in starts:
        window = samples[start:start+len(marker)]
        denom = np.linalg.norm(window)*energy
        scores.append(float(abs(np.vdot(marker, window))/denom)
                      if len(window) == len(marker) and denom else 0.)
    return float(np.median(scores))


def choose(raw, estimated, score, threshold):
    if estimated['applied_ratio'] is None and score >= threshold:
        return raw
    return estimated


def collect(seeds):
    rows = []
    for snr in (5, 0, -4, -8):
        for ppm in (-200, 0, 200):
            for seed in seeds:
                scores = []
                raw, estimated, _ = episode(
                    ppm, snr, seed, diagnostics=lambda *args: scores.append(quality(*args)))
                rows.append(dict(score=scores[0], raw=raw, estimated=estimated))
    return rows


def summarize(rows, threshold):
    selected = [r['raw'] for r in rows
                if choose(r['raw'], r['estimated'], r['score'], threshold) is r['raw']]
    return dict(released_recordings=len(selected),
                released_bits=sum(r['offered']-r['erased'] for r in selected),
                added_errors=sum(r['errors'] for r in selected))


def run():
    dev = collect(range(810101, 810107))
    candidates = []
    for threshold in [i/100 for i in range(0, 101, 5)] + [float('inf')]:
        result = summarize(dev, threshold)
        # Cap pooled development BER at 30%; use stricter threshold on ties.
        if result['added_errors'] <= .30*result['released_bits']:
            candidates.append((result['released_bits'], threshold))
    _, threshold = max(candidates)
    test = collect(range(910101, 910109))
    return dict(threshold=None if not np.isfinite(threshold) else threshold,
                dev_summary=summarize(dev, threshold), test_summary=summarize(test, threshold),
                test_unconditional=summarize(test, 0), development=dev, evaluation=test)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
