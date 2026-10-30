"""Development-locked marker alignment and captured-statistics ablation."""
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
from wavelink.slot_tracking import SlotTrackedLMMSE
from wavelink.followup import FollowupPilotLMMSE
from wavelink.receiver import comb_grid, pilot_noise_variance, zero_force
from wavelink.estimation import TapDomainLS
from experiments.acquisition_ber import packet
from experiments.captured_tracking_ber import validate as validate_parent, seed_sets, grid
from experiments.density_ber import digest
from experiments.lmmse_ber import se
from experiments.multipath_ber import write_csv


SCENARIOS = ['stationary', 'weak_drift', 'gap32', 'alias64', 'dirty_calibration']


def validate(cfg):
    if digest('reports/captured_tracking/metadata.json') != cfg['baseline_metadata_sha256']:
        raise ValueError('frozen captured-tracking metadata required')
    old = json.loads(Path('reports/captured_tracking/metadata.json').read_text())
    for p, h in old['source_sha256'].items():
        if digest(p) != h:
            raise ValueError('frozen source/config changed: '+p)
    if digest('reports/captured_tracking/selection.json') != cfg['gate_selection_sha256'] or json.loads(Path('reports/captured_tracking/selection.json').read_text())['gate'] != .25:
        raise ValueError('frozen .25 pilot gate required')
    if cfg['noise_multiples'] != [2., 4., 8.] or cfg['ebn0_db'] != [6, 18]:
        raise ValueError('fixed alignment candidates and SNR required')
    if cfg['development']['scenarios'] != SCENARIOS or cfg['test']['scenarios'] != SCENARIOS+['weak_first']:
        raise ValueError('held-out weak-first workload required')
    expected = [0.01, 0., 0., 0., .12, 0., 0., 0., 0., 0., 0., 0., .87, 0., 0., 0., 0.]
    if cfg['weak_first_powers'] != expected:
        raise ValueError('fixed weak-first PDP required')
    dev, test = cfg['development']['seeds'], cfg['test']['seeds']
    if len(dev) != 4 or len(test) != 16 or len(set(dev+test)) != 20 or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in dev+test):
        raise ValueError('new disjoint episode seeds required')
    used = set()
    for p in Path('experiments').glob('*.json'):
        if p.name != 'alignment.json':
            used.update(seed_sets(json.loads(p.read_text())))
    if used & set(dev+test):
        raise ValueError('fresh seeds required')
    return validate_parent(json.loads(Path('experiments/captured_tracking.json').read_text()))


def episode(cfg, protocol, scenario, db, si, seed, multiples):
    layout = comb_grid(2)
    bins = layout.bins(layout.data_tones)
    ls_model = TapDomainLS(layout, 17)
    noise = pilot_noise_variance(db, layout, 16)
    receivers = dict(raw_adaptive=SlotTrackedLMMSE(layout, gate=.25),
                     raw_frozen=SlotTrackedLMMSE(layout, gate=.25, adaptive=False),
                     oracle_adaptive=SlotTrackedLMMSE(layout))
    for c in multiples:
        receivers[f'aligned_{c}'] = SlotTrackedLMMSE(layout, gate=.25)
        receivers[f'frozen_{c}'] = SlotTrackedLMMSE(layout, gate=.25, adaptive=False)
    metrics = {name: [] for name in ['raw_ls', *[f'ls_{c}' for c in multiples], *receivers]}
    audits = {name: dict(consumed_unsafe=0, bootstrap_unsafe=0, scored_ready=0, gap_updates=0) for name in receivers}
    masks = {name: [] for name in receivers}
    # Alignment diagnostics are evaluator-only, never supplied to receivers.
    timing = {c: dict(captured=0, shifted=0, shift_sum=0, safe_before=0, safe_after=0,
                     safe_to_unsafe=0, unsafe_to_safe=0, strict_before=0, strict_after=0,
                     calibration_shifted=0, no_significant=0) for c in multiples}
    capture_mask = []
    for slot in range(384):
        role, t = (1, slot) if slot < 64 else (2, slot-64)
        powers = (cfg['weak_first_powers'] if scenario == 'weak_first' else
                  protocol['states']['weak15'] if scenario != 'stationary' and role == 2 and t >= 80 else
                  protocol['states']['initial'])
        dmax = max(i for i, p in enumerate(powers) if p > 0)
        epsilon = 5.2 if ((scenario == 'alias64' and role == 2 and 96 <= t < 160) or
                         (scenario == 'dirty_calibration' and role == 1 and 16 <= t < 48)) else 1.2
        missing = scenario == 'gap32' and role == 2 and 96 <= t < 128
        latent, pilots, bits, start, _ = packet(seed*100000+role*10000+t, si, powers, noise)
        observed = rotate_frequency(latent, epsilon, 64)
        raw = true = None
        aligned = {c: None for c in multiples}
        safe_raw = False
        safe_aligned = {c: False for c in multiples}
        if not missing:
            proposal = acquire(observed, .2)
            if proposal.accepted:
                raw = grid(observed, proposal.start, proposal.epsilon)
                safe_raw = dmax-16 <= proposal.start-start <= 0 and abs(proposal.epsilon-epsilon) <= .05
                for c in multiples:
                    a = align_marker(observed, proposal, c)
                    aligned[c] = grid(observed, a.start, proposal.epsilon)
                    safe = dmax-16 <= a.start-start <= 0 and abs(proposal.epsilon-epsilon) <= .05
                    safe_aligned[c] = safe
                    r = timing[c]
                    if role == 1:
                        r['calibration_shifted'] += int(a.shift > 0)
                    else:
                        r['captured'] += 1
                        r['shifted'] += int(a.shift > 0)
                        r['shift_sum'] += a.shift
                        r['safe_before'] += int(safe_raw)
                        r['safe_after'] += int(safe)
                        r['safe_to_unsafe'] += int(safe_raw and not safe)
                        r['unsafe_to_safe'] += int(not safe_raw and safe)
                        r['strict_before'] += int(proposal.start == start)
                        r['strict_after'] += int(a.start == start)
                        r['no_significant'] += int(not a.significant_tap)
            true = grid(observed, start, epsilon)
            np.testing.assert_allclose(true, grid(latent, start, 0.), rtol=1e-10, atol=1e-10)
        if role == 2:
            capture_mask.append(int(raw is not None))
            for name, y in [('raw_ls', raw), *[(f'ls_{c}', aligned[c]) for c in multiples]]:
                if y is None:
                    metrics[name].append((960, 960))
                else:
                    h = ls_model.estimate(y, pilots)
                    count = int(np.count_nonzero(qpsk_demodulate(zero_force(y[:, bins], h).ravel()) != bits.ravel()))
                    metrics[name].append((count, 0))
        for name, receiver in receivers.items():
            if name == 'oracle_adaptive':
                y, safe = true, True
            elif name.startswith('raw_'):
                y, safe = raw, safe_raw
            else:
                c = float(name.split('_')[-1])
                y, safe = aligned[c], safe_aligned[c]
            ready = receiver.tracker is not None
            h = receiver.step(y, pilots if y is not None else None, final=slot == 383)
            consumed = receiver.last['statistics_consumed']
            audits[name]['consumed_unsafe'] += int(consumed and not safe)
            audits[name]['bootstrap_unsafe'] += int(consumed and not ready and not safe)
            audits[name]['gap_updates'] += int(missing and receiver.last['updated'])
            masks[name].append(int(consumed))
            if role == 2:
                audits[name]['scored_ready'] += int(ready and h is not None)
                if h is None:
                    metrics[name].append((960, 960))
                else:
                    count = int(np.count_nonzero(qpsk_demodulate(zero_force(y[:, bins], h).ravel()) != bits.ravel()))
                    metrics[name].append((count, 0))
    costs, events = [], []
    for name, receiver in receivers.items():
        child = receiver.tracker
        if isinstance(child, FollowupPilotLMMSE):
            assert child.armed_plans == child.cancelled_plans+child.followup_count+int(child.pending_refit_after_frame is not None)
        if audits[name]['gap_updates'] or any(e['applies_from_slot'] >= 384 for e in receiver.events):
            raise AssertionError('gap or final unused fit')
        costs.append(dict(receiver=name, **audits[name], captured=receiver.captured, consumed=receiver.consumed,
                          bootstrap_slot=receiver.bootstrap_slot if receiver.bootstrap_slot is not None else '',
                          updates=child.update_count if child else 0, followups=getattr(child, 'followup_count', 0),
                          history_span=receiver.maximum_history_span,
                          mask_sha256=hashlib.sha256(bytes(masks[name])).hexdigest()))
        events.extend(dict(receiver=name, **e) for e in receiver.events)
    return metrics, costs, events, [dict(noise_multiple=c, **r) for c, r in timing.items()], hashlib.sha256(bytes(capture_mask)).hexdigest()


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['scenario'], r['ebn0_db'], r['receiver'], r['window']), []).append(r)
    result = []
    for key, g in groups.items():
        r = dict(zip(('scenario', 'ebn0_db', 'receiver', 'window'), key))
        r.update(episodes=len(g), offered_bits=sum(x['offered_bits'] for x in g), errors=sum(x['errors'] for x in g),
                 erased_bits=sum(x['erased_bits'] for x in g), accepted_bits=sum(x['accepted_bits'] for x in g))
        r['accepted_ber'] = r['errors']/r['accepted_bits'] if r['accepted_bits'] else ''
        for metric in ('loss', 'paired_minus_raw', 'paired_minus_ls', 'paired_minus_oracle'):
            r[metric] = float(np.mean([x[metric] for x in g]))
            r[metric+'_se'] = se([x[metric] for x in g]) if len(g) > 1 else ''
        result.append(r)
    return result


def evaluate(cfg, protocol, split, multiples):
    runs, costs, events, timing, capture = [], [], [], [], []
    for scenario in cfg[split]['scenarios']:
        for si, db in enumerate(cfg['ebn0_db']):
            for seed in cfg[split]['seeds']:
                values, count, log, align, mask = episode(cfg, protocol, scenario, db, si, seed, multiples)
                context = dict(scenario=scenario, ebn0_db=db, seed=seed)
                costs.extend(dict(context, **r) for r in count)
                events.extend(dict(context, **r) for r in log)
                timing.extend(dict(context, **r) for r in align)
                capture.append(dict(context, capture_mask_sha256=mask))
                for name, values_for_method in values.items():
                    for window, (a, b) in {'all': (0, 320), 'late': (224, 320)}.items():
                        selected = values_for_method[a:b]
                        erased = sum(x[1] for x in selected)
                        errors = sum(x[0] for x in selected)-erased
                        offered = (b-a)*960
                        loss = (errors+erased)/offered
                        row = dict(context, receiver=name, window=window, offered_bits=offered, accepted_bits=offered-erased,
                                   errors=errors, erased_bits=erased, loss=loss)
                        for ref, label in [('raw_adaptive', 'raw'), ('raw_ls', 'ls'), ('oracle_adaptive', 'oracle')]:
                            row['paired_minus_'+label] = loss-sum(x[0] for x in values[ref][a:b])/offered
                        runs.append(row)
                print(json.dumps(context), flush=True)
    return dict(runs=runs, summary=summarize(runs), costs=costs, events=events, alignment=timing, capture=capture)


def select(results, multiples):
    index = {(r['scenario'], r['ebn0_db'], r['receiver']): r for r in results['summary'] if r['window'] == 'all'}
    candidates = []
    for c in multiples:
        rows = [r for r in results['summary'] if r['receiver'] == f'aligned_{c}' and r['window'] == 'all']
        ratios = [r['loss']/index[r['scenario'], r['ebn0_db'], 'raw_adaptive']['loss'] for r in rows]
        candidates.append(dict(noise_multiple=c, worst_loss_ratio=max(ratios), mean_loss=float(np.mean([r['loss'] for r in rows])), eligible=max(ratios) <= 1.05))
    eligible = [c for c in candidates if c['eligible']]
    chosen = min(eligible, key=lambda c: (c['mean_loss'], c['noise_multiple']))['noise_multiple'] if eligible else min(multiples)
    return chosen, candidates, bool(eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiments/alignment.json'))
    parser.add_argument('--output', type=Path, default=Path('results/alignment'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    protocol = validate(cfg)
    paths = (sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))
             +sorted(Path('experiments').glob('*.json'))
             +[Path('reports/captured_tracking/metadata.json'), Path('reports/captured_tracking/selection.json')])
    hashes = {p.as_posix(): digest(p) for p in paths}
    args.output.mkdir(parents=True, exist_ok=True)
    dev = evaluate(cfg, protocol, 'development', cfg['noise_multiples'])
    chosen, candidates, eligible = select(dev, cfg['noise_multiples'])
    for name, rows in dev.items():
        write_csv(args.output/('development_'+name+'.csv'), rows)
    lock = dict(noise_multiple=chosen, candidates=candidates, eligible_candidate_found=eligible, config=cfg,
                source_sha256=hashes, development_runs_sha256=digest(args.output/'development_runs.csv'),
                locked_before_test_utc=datetime.now(timezone.utc).isoformat(),
                rule='Each development all-window loss<=1.05 raw adaptive; minimum equal-cell mean loss among eligible, tie smaller multiple; else smallest multiple retaining failure; weak_first excluded from development')
    (args.output/'selection.json').write_text(json.dumps(lock, indent=2)+'\n')
    lock_hash = digest(args.output/'selection.json')
    started = datetime.now(timezone.utc).isoformat()
    result = evaluate(cfg, protocol, 'test', [chosen])
    if any(digest(p) != h for p, h in hashes.items()) or digest(args.output/'selection.json') != lock_hash:
        raise RuntimeError('source/protocol/selection changed')
    for name, rows in result.items():
        write_csv(args.output/(name+'.csv'), rows)
    meta = dict(config=cfg, source_sha256=hashes, selection_sha256=lock_hash, test_started_at_utc=started,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                command=['python', '-m', 'experiments.alignment_ber', *sys.argv[1:]], python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                alignment='Known marker only: full acquired CFO unchanged; first tap0..16 above multiple*mean tail17..63 power; numeric relative/absolute floors; if none leave start unchanged; forward shift0..16 only',
                causality='Uses original acquisition prefix only, ending<=287; no true start/channel or data bits; same accepted packets and training; next model used only next slot',
                clock='Frozen v0.19 admitted-observation clock, gate.25,64-observation bootstrap, identical gaps/final boundaries; only input FFT reference changes',
                scope='New episode seeds, same five workloads plus held-out weak_first with powers0=.01,4=.12,12=.87; no extra timing/CFO truth in actual front end',
                costs='Same header energy231.216221811 and initial64+test320 transmission budget; alignment adds marker FFT/IFFT computations, no runtime claim',
                ablations='Raw/aligned LS, raw/aligned gated frozen/adaptive priors, true-boundary/CFO learned adaptive reference; labels never filter observations or scores',
                uncertainty='Per-cell16 independent episodes, scenarios share latent streams; paired whole-episode SE; all/late windows overlap',
                limits='Unknown early weak path can be missed; no backward correction or guaranteed physical TX origin; post-alignment support/ICI/CPE may remain mismatched; no theory-NMSE claims')
    (args.output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(dict(noise_multiple=chosen, eligible=eligible, test_episodes=192, oracle_grid_checks=72704)))


if __name__ == '__main__':
    main()
