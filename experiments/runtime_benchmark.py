"""Frozen receiver replay timing; run from repository root as a module."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns
import contextlib
import numpy as np
from wavelink import qpsk_demodulate
from wavelink.receiver import comb_grid, pilot_noise_variance, zero_force
from wavelink.tracking import CausalPilotLMMSE
from wavelink.periodic import CachedPeriodicLMMSE
from wavelink.innovation import InnovationTriggeredLMMSE
from wavelink.followup import FollowupPilotLMMSE
from experiments.density_ber import coupled_frame, digest
from experiments.generalization_ber import timeline
from experiments.lmmse_ber import se
from experiments.multipath_ber import write_csv

METHODS = ('periodic', 'cached_periodic', 'innovation', 'followup')


def factory(name, cal, layout):
    if name == 'periodic':
        return CausalPilotLMMSE(cal, layout, 17, 64, 32)
    if name == 'cached_periodic':
        return CachedPeriodicLMMSE(cal, layout, 17, 64, 32)
    if name == 'innovation':
        return InnovationTriggeredLMMSE(cal, layout, 17, innovation_threshold=2.)
    if name == 'followup':
        return FollowupPilotLMMSE(cal, layout, 17, innovation_threshold=2., refit_delay=64)
    raise ValueError('unknown receiver')


def orders(seed, cycles):
    """Randomized Latin rotations: each method occupies each position per cycle."""
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(cycles):
        base = list(rng.permutation(METHODS))
        for shift in rng.permutation(len(base)):
            result.append(base[int(shift):] + base[:int(shift)])
    return result


def prepare(cfg, spec, db, si, seed, onset):
    layout = comb_grid(2)
    noise = pilot_noise_variance(db, layout, cfg['cp_len'])
    cal = []
    for t in range(cfg['calibration_frames']):
        y, p, _, _ = coupled_frame(seed, si, 1, t, cfg['states']['initial'], noise,
                                   layout, cfg['cp_len'], cfg['symbols_per_frame'])
        cal.append(y[:, layout.bins(layout.pilot_tones)]/p)
    powers, noises = timeline(cfg, spec, onset)
    frames = []
    for t in range(cfg['episode_frames']):
        y, p, bits, _ = coupled_frame(seed, si, 2, t, powers[t], noise*noises[t],
                                    layout, cfg['cp_len'], cfg['symbols_per_frame'])
        for array in (y, p, bits):
            array.setflags(write=False)
        frames.append((y, p, bits))
    cal = np.asarray(cal)
    cal.setflags(write=False)
    return cal, frames, layout


def replay(name, cal, frames, layout):
    # External timers include complete constructor and receiver loop; no sum of
    # internal counters is substituted for wall time. Output hashing is outside.
    bins = layout.bins(layout.data_tones)
    decoded = []
    events = []
    started = perf_counter_ns()
    tracker = factory(name, cal, layout)
    initialized = perf_counter_ns()
    for t, (y, p, _) in enumerate(frames):
        h = tracker.estimate(y, p)
        decoded.append(qpsk_demodulate(zero_force(y[:, bins], h).ravel()))
        if t < len(frames)-1 and tracker.advance():
            events.append(t)
    finished = perf_counter_ns()
    received = np.asarray(decoded)
    errors = int(np.count_nonzero(received != np.asarray([b.ravel() for _, _, b in frames])))
    return dict(init_ms=(initialized-started)/1e6, online_ms=(finished-initialized)/1e6,
                total_ms=(finished-started)/1e6, updates=tracker.update_count,
                fit_frame_visits=tracker.fit_frame_visits, errors=errors,
                bits=received.size, output_sha256=hashlib.sha256(received.tobytes()).hexdigest(),
                event_sha256=hashlib.sha256(json.dumps(events).encode()).hexdigest())


def equivalence(cal, frames, layout):
    """Untimed all-frame estimates/statistics and effective event comparison."""
    a, b = (factory(name, cal, layout) for name in METHODS[:2])
    maximum = 0.
    for t, (y, p, _) in enumerate(frames):
        x, z = a.estimate(y, p), b.estimate(y, p)
        maximum = max(maximum, float(np.max(abs(x-z))))
        np.testing.assert_allclose(z, x, rtol=1e-10, atol=1e-10)
        if t < len(frames)-1:
            if a.advance() != b.advance():
                raise AssertionError('different periodic update schedule')
            np.testing.assert_allclose(b.statistics.powers, a.statistics.powers, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(b.statistics.noise_variance, a.statistics.noise_variance, rtol=1e-10, atol=1e-10)
    return dict(max_abs_estimate_difference=maximum, frames=len(frames), updates=a.update_count, passed=True)


def validate(cfg):
    if digest('reports/followup/metadata.json') != cfg['baseline_metadata_sha256']:
        raise ValueError('baseline metadata changed')
    meta = json.loads(Path('reports/followup/metadata.json').read_text())
    for path, expected in meta['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError('frozen source changed: '+path)
    if cfg['protocol'] != 'experiments/followup.json':
        raise ValueError('frozen protocol required')
    if cfg['scenarios'] != ['stationary', 'weak15', 'repeat24']:
        raise ValueError('fixed timing workload required')
    if cfg['ebn0_db'] != [6, 18] or cfg['onset_phases'] != [0, 8, 16, 24]:
        raise ValueError('fixed SNR and phase design required')
    if isinstance(cfg['cycles'], bool) or not isinstance(cfg['cycles'], int) or cfg['cycles'] < 1:
        raise ValueError('positive integer timing cycles required')
    seeds = cfg['seeds']
    if len(seeds) != 4 or len(set(seeds)) != 4 or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError('four distinct integer seeds required')
    used = set()
    for path in Path('experiments').glob('*.json'):
        if path.name == 'runtime.json':
            continue
        old = json.loads(path.read_text())
        used.update(old.get('seeds', []))
        for split in ('development', 'test'):
            used.update(old.get(split, {}).get('seeds', []))
    if used & set(seeds):
        raise ValueError('fresh timing seeds required')
    return json.loads(Path(cfg['protocol']).read_text())


def summarize(runs):
    result = []
    for scenario in dict.fromkeys(r['scenario'] for r in runs):
        for db in dict.fromkeys(r['ebn0_db'] for r in runs):
            cell = [r for r in runs if r['scenario'] == scenario and r['ebn0_db'] == db]
            # Replays of one seed are technical repeats, not independent links.
            for method in METHODS:
                rows = []
                ratios = []
                for seed in sorted(set(r['seed'] for r in cell)):
                    g = [r for r in cell if r['seed'] == seed and r['receiver'] == method]
                    control = [r for r in cell if r['seed'] == seed and r['receiver'] == 'cached_periodic']
                    med = {k: float(np.median([r[k] for r in g])) for k in ('init_ms', 'online_ms', 'total_ms')}
                    rows.append(med)
                    ratios.append(med['online_ms']/float(np.median([r['online_ms'] for r in control])))
                r = dict(scenario=scenario, ebn0_db=db, receiver=method, independent_seed_branches=len(rows),
                         replays_per_seed=len(g), mean_online_ratio_to_cached=float(np.mean(ratios)), ratio_seed_se=se(ratios))
                for k in ('init_ms', 'online_ms', 'total_ms'):
                    r[k] = float(np.mean([x[k] for x in rows]))
                    r[k+'_seed_se'] = se([x[k] for x in rows])
                chosen = [r for r in cell if r['receiver'] == method and r['repeat'] == 0]
                r['mean_updates'] = float(np.mean([x['updates'] for x in chosen]))
                r['ber'] = sum(x['errors'] for x in chosen)/sum(x['bits'] for x in chosen)
                result.append(r)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiments/runtime.json'))
    parser.add_argument('--output', type=Path, default=Path('results/runtime'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    protocol = validate(cfg)
    sources = sorted(Path('src').rglob('*.py')) + sorted(Path('experiments').glob('*.py')) + [args.config, Path(cfg['protocol'])]
    hashes = {p.as_posix(): digest(p) for p in sources}
    started = datetime.now(timezone.utc).isoformat()
    runs, checks, order_log = [], [], []
    specs = {s['name']: s for s in protocol['scenarios']}
    # GC policy is fixed and restored, including on failure. Native clocks and
    # instrumentation in each frozen class remain enabled for all methods.
    was_enabled = gc.isenabled()
    try:
        gc.disable()
        for scenario in cfg['scenarios']:
            for si, db in enumerate(cfg['ebn0_db']):
                for seed, phase in zip(cfg['seeds'], cfg['onset_phases']):
                    context = dict(scenario=scenario, ebn0_db=db, seed=seed, onset_phase=phase)
                    cal, frames, layout = prepare(protocol, specs[scenario], db, si, seed, 64+phase)
                    fingerprint = hashlib.sha256(cal.tobytes())
                    for y, p, bits in frames:
                        for array in (y, p, bits):
                            fingerprint.update(array.tobytes())
                    checks.append(dict(context, input_sha256=fingerprint.hexdigest(), **equivalence(cal, frames, layout)))
                    # One whole untimed replay per method, retained as output reference.
                    expected = {m: replay(m, cal, frames, layout) for m in METHODS}
                    for repeat, order in enumerate(orders(seed+si, cfg['cycles'])):
                        for position, method in enumerate(order):
                            row = replay(method, cal, frames, layout)
                            for key in ('output_sha256', 'event_sha256', 'errors', 'bits', 'updates'):
                                if row[key] != expected[method][key]:
                                    raise AssertionError('replay changed deterministic result: '+key)
                            if method == 'cached_periodic':
                                for key in ('output_sha256', 'event_sha256', 'errors', 'bits', 'updates'):
                                    if row[key] != expected['periodic'][key]:
                                        raise AssertionError('incremental baseline differs: '+key)
                            runs.append(dict(context, repeat=repeat, position=position, receiver=method, **row))
                            order_log.append(dict(context, repeat=repeat, position=position, receiver=method))
                    print(json.dumps(context), flush=True)
    finally:
        if was_enabled:
            gc.enable()
    if any(digest(p) != h for p, h in hashes.items()):
        raise RuntimeError('source changed during measurement')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('runs', runs), ('summary', summarize(runs)), ('equivalence', checks), ('order', order_log)]:
        write_csv(args.output/(name+'.csv'), rows)
    info = io.StringIO()
    with contextlib.redirect_stdout(info):
        np.show_config()
    meta = dict(config=cfg, source_sha256=hashes, base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                command=['python', '-m', 'experiments.runtime_benchmark', *sys.argv[1:]],
                started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(), numpy=np.__version__, platform=platform.platform(), machine=platform.machine(),
                blas_configuration=info.getvalue(), clock=vars(__import__('time').get_clock_info('perf_counter')),
                thread_environment={k: os.environ.get(k) for k in ('VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                gc='disabled during all workloads, original state restored',
                timing='External full constructor and estimate/ZF/QPSK/advance loop; includes internal instrumentation, allocations and output/event collection; excludes waveform generation, truth/error scoring, hashing, equivalence assertions and report I/O',
                ordering='One whole untimed warmup per method per episode, then seeded randomized Latin rotations; each method occupies each position equally',
                uncertainty='Median of technical repeats per seed, then mean and descriptive SE of four seed branches; scenario streams shared, not independent experiment replications; timings are machine/session specific',
                scope='Frozen v0.15 policies; no timing-based algorithm selection; cached periodic output equivalence checked; no claims of hard real time or cross-hardware speedup')
    (args.output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(dict(equivalence_pass=len(checks), timed_replays=len(runs))))


if __name__ == '__main__':
    main()
