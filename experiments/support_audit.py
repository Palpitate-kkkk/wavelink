"""Signed-prior audit on ideal diagonal channels, not a new BER benchmark."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from wavelink.audit import audit_report
from wavelink.receiver import comb_grid
from wavelink.estimation import TapDomainLS
from wavelink.signed_estimation import SignedTapDomainLS
from wavelink.reference import rotate_grid_reference
from experiments.captured_tracking_ber import seed_sets
from experiments.density_ber import digest
from experiments.multipath_ber import write_csv


def validate(cfg):
    if digest('reports/reference/metadata.json') != cfg['parent_metadata_sha256']:
        raise ValueError('frozen parent metadata required')
    if not audit_report('.', 'reports/reference')['ok']:
        raise ValueError('parent source/artifact consistency required')
    if cfg['shifts'] != list(range(17)) or cfg['noise_variance'] != .05 or cfg['symbols'] != 20:
        raise ValueError('fixed shift/noise/symbol protocol required')
    parent = json.loads(Path('experiments/followup.json').read_text())['states']['initial']
    expected = dict(initial=parent,
                    weak_first=json.loads(Path('experiments/alignment.json').read_text())['weak_first_powers'],
                    weaker_first=json.loads(Path('experiments/reference.json').read_text())['weaker_first_powers'])
    if cfg['profiles'] != expected:
        raise ValueError('inherited PDPs required')
    seeds = cfg['seeds']
    if len(seeds) != 16 or len(set(seeds)) != 16 or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError('sixteen unique integer roots required')
    used = set()
    for path in Path('experiments').glob('*.json'):
        if path.name != 'support.json':
            used.update(seed_sets(json.loads(path.read_text())))
    if set(seeds) & used:
        raise ValueError('independent new roots required')


def evaluate(cfg):
    layout = comb_grid(2)
    pb, db = layout.bins(layout.pilot_tones), layout.bins(layout.data_tones)
    pilots = np.ones((cfg['symbols'], len(pb)), complex)
    raw_ls = TapDomainLS(layout, 17)
    rows = []
    for profile, powers in cfg['profiles'].items():
        powers = np.asarray(powers)
        for seed in cfg['seeds']:
            # Same latent channel/noise roots across profiles and all shifts.
            rng = np.random.default_rng(seed)
            taps = (rng.normal(size=17)+1j*rng.normal(size=17))*np.sqrt(powers/2)
            h = np.fft.fft(taps, 64)
            noise = (rng.normal(size=(cfg['symbols'], 64))+1j*rng.normal(size=(cfg['symbols'], 64)))*np.sqrt(cfg['noise_variance']/2)
            y = h[None, :]+noise
            baseline = raw_ls.estimate(y, pilots)
            denominator = float(np.mean(abs(h[db])**2))
            for shift in cfg['shifts']:
                canonical = rotate_grid_reference(y, shift)
                hc = rotate_grid_reference(h[None, :], shift)[0]
                phase = np.exp(-2j*np.pi*db*shift/64)
                for name, support in [('causal', range(17)), ('centered', range(-8, 9)),
                                      ('negative', range(-16, 1)), ('transported', range(-shift, 17-shift))]:
                    model = SignedTapDomainLS(layout, support)
                    estimate = model.estimate(canonical, pilots)*phase
                    error = float(np.mean(abs(estimate-h[db])**2)/denominator)
                    bias, variance = model.expected_nmse(hc, cfg['noise_variance'])
                    projector = model.pilot_design @ np.linalg.pinv(model.pilot_design, rcond=1e-12)
                    hp = hc[pb]
                    ratio = float(np.sum(abs(hp-projector@hp)**2)/np.sum(abs(hp)**2))
                    z = canonical[:, pb]
                    observed_ratio = float(np.sum(abs(z-z@projector.T)**2)/np.sum(abs(z)**2))
                    outside = float(sum(abs(taps[d])**2 for d in range(17) if d-shift not in support)/np.sum(abs(taps)**2))
                    delta = float(np.max(abs(estimate-baseline)))
                    if name == 'transported' and delta > 1e-10:
                        raise AssertionError('coordinate-only LS control differs')
                    rows.append(dict(profile=profile, seed=seed, shift=shift, estimator=name,
                                     outside_support_power=outside, pilot_ratio_noiseless=ratio,
                                     pilot_ratio_observed=observed_ratio, gate_pass=observed_ratio <= .25,
                                     data_bias_nmse=bias, noise_nmse=variance, predicted_nmse=bias+variance,
                                     observed_nmse=error, max_difference_from_raw_ls=delta,
                                     condition=model.condition, noise_gain=model.noise_gain))
    summary = []
    for profile in cfg['profiles']:
        for shift in cfg['shifts']:
            for name in ('causal', 'centered', 'negative', 'transported'):
                g = [r for r in rows if (r['profile'], r['shift'], r['estimator']) == (profile, shift, name)]
                row = dict(profile=profile, shift=shift, estimator=name, episodes=len(g),
                           gate_pass_count=sum(r['gate_pass'] for r in g))
                for key in ('outside_support_power', 'pilot_ratio_noiseless', 'pilot_ratio_observed',
                            'data_bias_nmse', 'noise_nmse', 'predicted_nmse', 'observed_nmse'):
                    values = [r[key] for r in g]
                    row[key] = float(np.mean(values))
                    row[key+'_se'] = float(np.std(values, ddof=1)/np.sqrt(len(g)))
                row['max_difference_from_raw_ls'] = max(r['max_difference_from_raw_ls'] for r in g)
                summary.append(row)
    return dict(runs=rows, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiments/support.json'))
    parser.add_argument('--output', type=Path, default=Path('results/support'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate(cfg)
    paths = sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))+sorted(Path('experiments').glob('*.json'))
    paths += [Path('reports/reference/metadata.json')]
    hashes = {p.as_posix(): digest(p) for p in paths}
    args.output.mkdir(parents=True, exist_ok=True)
    lock = dict(config=cfg, source_sha256=hashes, frozen_before_evaluation_utc=datetime.now(timezone.utc).isoformat(),
                scope='Ideal diagonal static channels only; known experimental reference shifts swept, never estimated; four declared priors, no selection or BER claim')
    (args.output/'protocol.json').write_text(json.dumps(lock, indent=2)+'\n')
    lock_hash = digest(args.output/'protocol.json')
    result = evaluate(cfg)
    if any(digest(p) != h for p, h in hashes.items()) or digest(args.output/'protocol.json') != lock_hash:
        raise RuntimeError('source/protocol changed during evaluation')
    for name, rows in result.items():
        write_csv(args.output/(name+'.csv'), rows)
    artifacts = {name: digest(args.output/name) for name in ('runs.csv', 'summary.csv', 'protocol.json')}
    meta = dict(config=cfg, source_sha256=hashes, artifact_sha256=artifacts,
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                command=['python', '-m', 'experiments.support_audit', *sys.argv[1:]],
                python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
                interpretation='Per-realization normalized channel MSE; independent unit known pilots, diagonal channel and white noise variance.05; no CFO/ISI/acquisition simulation. Pilot gate.25 is diagnostic only; all outputs retained.',
                uncertainty='SE over16 independent channel/noise roots; same latent draws reused across profiles, shifts and models; symbols are not independent channel episodes.',
                control='Transport support0..16 to -shift..16-shift and undo phase: algebraically original LS even with arbitrary noisy pilots. Fixed causal support after rotation changes the physical prior.',
                boundary='Signed supports are fixed ablations, not a learned selector or a repaired LMMSE tracker. Observational replay of v0.21 is separate and not new statistical evidence.')
    (args.output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(dict(rows=len(result['runs']), summary_rows=len(result['summary']),
                          max_coordinate_control_error=max(r['max_difference_from_raw_ls'] for r in result['runs'] if r['estimator']=='transported'))))


if __name__ == '__main__':
    main()
