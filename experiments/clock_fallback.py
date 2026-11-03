"""Evaluate rejection-triggered fallback without consulting true clock or bits."""
import json

from experiments.clock_link import episode


def fallback(raw, estimated):
    if estimated['applied_ratio'] is None:
        return dict(raw, method='fallback', used_raw=True)
    return dict(estimated, method='fallback', used_raw=False)


def run():
    rows = []
    for snr in (5, 0, -4, -8):
        for ppm in (-200, 0, 200):
            for seed in range(730101, 730107):
                raw, estimated, _ = episode(ppm, snr, seed)
                rows.append(dict(raw=raw, reject=estimated,
                                 fallback=fallback(raw, estimated)))
    return rows


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
