"""Frozen phase-reference normalization; physical FFT windows remain unchanged."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from wavelink import qpsk_demodulate
from wavelink.acquisition import acquire
from wavelink.alignment import align_marker
from wavelink.synchronization import rotate_frequency
from wavelink.reference import ReferenceTrackedLMMSE
from wavelink.slot_tracking import SlotTrackedLMMSE
from wavelink.followup import FollowupPilotLMMSE
from wavelink.receiver import comb_grid, pilot_noise_variance, zero_force
from wavelink.estimation import TapDomainLS
from experiments.acquisition_ber import packet
from experiments.alignment_ber import validate as validate_parent, summarize
from experiments.captured_tracking_ber import seed_sets, grid
from experiments.density_ber import digest
from experiments.multipath_ber import write_csv


SCENARIOS = ['stationary', 'weak_drift', 'gap32', 'alias64', 'dirty_calibration', 'weak_first', 'weaker_first']


def validate(cfg):
    for path, key in [('reports/alignment/metadata.json', 'baseline_metadata_sha256'), ('reports/alignment/selection.json', 'alignment_selection_sha256')]:
        if digest(path) != cfg[key]:
            raise ValueError('frozen alignment record required: '+path)
    old = json.loads(Path('reports/alignment/metadata.json').read_text())
    for p, h in old['source_sha256'].items():
        if digest(p) != h:
            raise ValueError('frozen source/protocol changed: '+p)
    if json.loads(Path('reports/alignment/selection.json').read_text())['noise_multiple'] != 8.:
        raise ValueError('frozen marker multiple8 required')
    if cfg['scenarios'] != SCENARIOS or cfg['ebn0_db'] != [6, 18]:
        raise ValueError('fixed seven-workload/two-SNR design required')
    expected = [0.001, 0., 0., 0., .12, 0., 0., 0., 0., 0., 0., 0., .879, 0., 0., 0., 0.]
    if cfg['weaker_first_powers'] != expected:
        raise ValueError('fixed new weaker-first PDP required')
    seeds = cfg['seeds']
    if len(seeds) != 16 or len(set(seeds)) != 16 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds):
        raise ValueError('sixteen distinct integer episode seeds required')
    used = set()
    for p in Path('experiments').glob('*.json'):
        if p.name != 'reference.json':
            used.update(seed_sets(json.loads(p.read_text())))
    if used & set(seeds):
        raise ValueError('new root seeds required')
    return validate_parent(json.loads(Path('experiments/alignment.json').read_text()))


def episode(cfg, protocol, scenario, db, si, seed):
    layout = comb_grid(2)
    bins = layout.bins(layout.data_tones)
    ls = TapDomainLS(layout, 17)
    noise = pilot_noise_variance(db, layout, 16)
    receivers = dict(raw_adaptive=SlotTrackedLMMSE(layout, gate=.25), raw_frozen=SlotTrackedLMMSE(layout, gate=.25, adaptive=False),
                     physical_adaptive=SlotTrackedLMMSE(layout, gate=.25),
                     reference_adaptive=ReferenceTrackedLMMSE(layout, gate=.25),
                     reference_frozen=ReferenceTrackedLMMSE(layout, gate=.25, adaptive=False),
                     oracle_adaptive=SlotTrackedLMMSE(layout))
    values = {k: [] for k in ['raw_ls', 'physical_ls', *receivers]}
    costs = {k: dict(consumed_unsafe_window=0, consumed_unsafe_reference=0, gap_updates=0) for k in receivers}
    masks = {k: [] for k in receivers}
    timing = dict(captured=0, shifted=0, raw_safe=0, physical_safe=0, reference_safe=0,
                  physical_new_unsafe=0, reference_new_unsafe=0)
    raw_starts, reference_starts = [], []
    weak_powers = json.loads(Path('experiments/alignment.json').read_text())['weak_first_powers']
    for slot in range(384):
        role, t = (1, slot) if slot < 64 else (2, slot-64)
        if scenario == 'weak_first':
            powers = weak_powers
        elif scenario == 'weaker_first':
            powers = cfg['weaker_first_powers']
        else:
            powers = protocol['states']['weak15'] if scenario != 'stationary' and role == 2 and t >= 80 else protocol['states']['initial']
        dmax = max(i for i, p in enumerate(powers) if p > 0)
        epsilon = 5.2 if ((scenario == 'alias64' and role == 2 and 96 <= t < 160) or
                         (scenario == 'dirty_calibration' and role == 1 and 16 <= t < 48)) else 1.2
        missing = scenario == 'gap32' and role == 2 and 96 <= t < 128
        latent, pilots, bits, start, _ = packet(seed*100000+role*10000+t, si, powers, noise)
        observed = rotate_frequency(latent, epsilon, 64)
        raw = physical = true = None
        shift = 0
        safe_raw = safe_physical = False
        if not missing:
            proposal = acquire(observed, .2)
            if proposal.accepted:
                a = align_marker(observed, proposal, 8.)
                shift = a.shift
                raw = grid(observed, proposal.start, proposal.epsilon)
                physical = grid(observed, a.start, proposal.epsilon)
                safe_raw = dmax-16 <= proposal.start-start <= 0 and abs(proposal.epsilon-epsilon) <= .05
                safe_physical = dmax-16 <= a.start-start <= 0 and abs(proposal.epsilon-epsilon) <= .05
                if role == 2:
                    timing['captured'] += 1
                    timing['shifted'] += int(shift > 0)
                    timing['raw_safe'] += int(safe_raw)
                    timing['physical_safe'] += int(safe_physical)
                    timing['reference_safe'] += int(safe_raw)
                    timing['physical_new_unsafe'] += int(safe_raw and not safe_physical)
                    # Both branches consume this exact raw FFT window.
                    raw_starts.append(proposal.start)
                    reference_starts.append(proposal.start)
            true = grid(observed, start, epsilon)
            np.testing.assert_allclose(true, grid(latent, start, 0.), rtol=1e-10, atol=1e-10)
        if role == 2:
            for method, y in [('raw_ls', raw), ('physical_ls', physical)]:
                if y is None:
                    values[method].append((960, 960))
                else:
                    h = ls.estimate(y, pilots)
                    error = int(np.count_nonzero(qpsk_demodulate(zero_force(y[:, bins], h).ravel()) != bits.ravel()))
                    values[method].append((error, 0))
        for name, receiver in receivers.items():
            y = true if name == 'oracle_adaptive' else physical if name == 'physical_adaptive' else raw
            if name.startswith('reference_'):
                h = receiver.step(y, pilots if y is not None else None, shift=shift if y is not None else 0, final=slot == 383)
                window_safe, model_safe = safe_raw, safe_physical
            else:
                h = receiver.step(y, pilots if y is not None else None, final=slot == 383)
                window_safe = model_safe = True if name == 'oracle_adaptive' else safe_physical if name == 'physical_adaptive' else safe_raw
            consumed = receiver.last['statistics_consumed']
            costs[name]['consumed_unsafe_window'] += int(consumed and not window_safe)
            costs[name]['consumed_unsafe_reference'] += int(consumed and not model_safe)
            costs[name]['gap_updates'] += int(missing and receiver.last['updated'])
            masks[name].append(int(consumed))
            if role == 2:
                if h is None:
                    values[name].append((960, 960))
                else:
                    error = int(np.count_nonzero(qpsk_demodulate(zero_force(y[:, bins], h).ravel()) != bits.ravel()))
                    values[name].append((error, 0))
    logs, records = [], []
    for name, receiver in receivers.items():
        child = receiver.tracker
        if isinstance(child, FollowupPilotLMMSE):
            assert child.armed_plans == child.cancelled_plans+child.followup_count+int(child.pending_refit_after_frame is not None)
        if costs[name]['gap_updates'] or any(e['applies_from_slot'] >= 384 for e in receiver.events):
            raise AssertionError('gap/final fit')
        records.append(dict(receiver=name, **costs[name], captured=receiver.captured, consumed=receiver.consumed,
                            bootstrap_slot=receiver.bootstrap_slot if receiver.bootstrap_slot is not None else '',
                            updates=child.update_count if child else 0, history_span=receiver.maximum_history_span,
                            mask_sha256=hashlib.sha256(bytes(masks[name])).hexdigest()))
        logs.extend(dict(receiver=name, **e) for e in receiver.events)
    assert raw_starts == reference_starts
    timing['raw_window_sha256'] = hashlib.sha256(bytes(raw_starts)).hexdigest()
    timing['reference_window_sha256'] = hashlib.sha256(bytes(reference_starts)).hexdigest()
    return values, records, logs, timing


def evaluate(cfg, protocol):
    rows, costs, events, timing = [], [], [], []
    for scenario in cfg['scenarios']:
        for si, db in enumerate(cfg['ebn0_db']):
            for seed in cfg['seeds']:
                values, counts, logs, alignment = episode(cfg, protocol, scenario, db, si, seed)
                context = dict(scenario=scenario, ebn0_db=db, seed=seed)
                costs.extend(dict(context, **r) for r in counts)
                events.extend(dict(context, **r) for r in logs)
                timing.append(dict(context, **alignment))
                for method, data in values.items():
                    for window, (a, b) in {'all': (0, 320), 'late': (224, 320)}.items():
                        selected = data[a:b]
                        erased = sum(x[1] for x in selected)
                        errors = sum(x[0] for x in selected)-erased
                        offered = (b-a)*960
                        loss = (errors+erased)/offered
                        row = dict(context, receiver=method, window=window, offered_bits=offered,
                                   accepted_bits=offered-erased, errors=errors, erased_bits=erased, loss=loss)
                        for ref, label in [('raw_adaptive', 'raw'), ('raw_ls', 'ls'), ('oracle_adaptive', 'oracle')]:
                            row['paired_minus_'+label] = loss-sum(x[0] for x in values[ref][a:b])/offered
                        rows.append(row)
                print(json.dumps(context), flush=True)
    return dict(runs=rows, summary=summarize(rows), costs=costs, events=events, windows=timing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiments/reference.json'))
    parser.add_argument('--output', type=Path, default=Path('results/reference'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    protocol = validate(cfg)
    paths = (sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))+sorted(Path('experiments').glob('*.json'))
             +[Path('reports/alignment/metadata.json'), Path('reports/alignment/selection.json')])
    hashes = {p.as_posix(): digest(p) for p in paths}
    args.output.mkdir(parents=True, exist_ok=True)
    lock = dict(config=cfg, source_sha256=hashes, frozen_before_evaluation_utc=datetime.now(timezone.utc).isoformat(),
                policy='No new selection: inherited marker8, pilot gate.25 and capture.2; diagonal phase rotation only for reference branches; original FFT windows retained')
    (args.output/'protocol.json').write_text(json.dumps(lock, indent=2)+'\n')
    lock_hash = digest(args.output/'protocol.json')
    started = datetime.now(timezone.utc).isoformat()
    result = evaluate(cfg, protocol)
    if any(digest(p) != h for p, h in hashes.items()) or digest(args.output/'protocol.json') != lock_hash:
        raise RuntimeError('source/protocol changed')
    for name, rows in result.items():
        write_csv(args.output/(name+'.csv'), rows)
    artifacts = {p.name: digest(p) for p in sorted(args.output.glob('*.csv'))}
    artifacts['protocol.json'] = lock_hash
    meta = dict(config=cfg, source_sha256=hashes, artifact_sha256=artifacts, protocol_sha256=lock_hash,
                evaluation_started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                command=['python', '-m', 'experiments.reference_ber', *sys.argv[1:]], python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                reference='Same acquired FFT grid; multiply frequency bin k by exp(+j2pi*k*shift/64) for model input, restore channel estimate by inverse phase before equalizing ORIGINAL data',
                safety='No time-window move in reference branches; raw/ref window hashes match. This preserves raw timing conditions only, not correct acquisition or causal17-tap model support in transformed coordinates',
                causality='Shift from original known marker prefix only; old SlotTrackedLMMSE gate/calibration/admitted clock unchanged; estimates produced before fitting, no gap/final fit',
                scope='Fixed prior parameters,16 new seeds per cell,7 workloads including new weaker_first p0=.001,p4=.12,p12=.879; no new tuning',
                uncertainty='Whole independent episode paired SE, scenarios share latent streams; all/late overlap; no hard real-time or physical-first-path guarantee',
                energy='Same224-point header and initial64+test320 slots; no extra pilots/transmissions; extra phase rotations are not timed',
                provenance='SHA256 consistency manifest is not an independent authenticity/signature proof; raw and virtual inputs preserve the existing modeled limitations')
    (args.output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(dict(episodes=224, oracle_grid_checks=84992, window_moves_in_reference=0)))


if __name__ == '__main__':
    main()
