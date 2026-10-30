"""Causal acquisition-to-statistics integration with development-locked pilot gate."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from wavelink import ofdm_demodulate, qpsk_demodulate
from wavelink.acquisition import acquire, acquisition_header
from wavelink.synchronization import rotate_frequency
from wavelink.slot_tracking import SlotTrackedLMMSE
from wavelink.followup import FollowupPilotLMMSE
from wavelink.receiver import comb_grid, pilot_noise_variance, zero_force
from wavelink.estimation import TapDomainLS
from experiments.acquisition_ber import packet, validate as validate_acquisition
from experiments.density_ber import digest
from experiments.lmmse_ber import se
from experiments.multipath_ber import write_csv


def seed_sets(c):
    result = set()
    if isinstance(c, dict):
        for k, v in c.items():
            if k == 'seeds':
                result.update(v)
            else:
                result.update(seed_sets(v))
    elif isinstance(c, list):
        for v in c:
            result.update(seed_sets(v))
    return result


def validate(cfg):
    if digest('reports/acquisition/metadata.json') != cfg['baseline_metadata_sha256']:
        raise ValueError('frozen acquisition metadata required')
    old = json.loads(Path('reports/acquisition/metadata.json').read_text())
    for p, h in old['source_sha256'].items():
        if digest(p) != h:
            raise ValueError('frozen source changed: '+p)
    if digest('reports/acquisition/selection.json') != cfg['capture_selection_sha256']:
        raise ValueError('frozen capture selection required')
    if json.loads(Path('reports/acquisition/selection.json').read_text())['threshold'] != .2:
        raise ValueError('capture threshold must remain .2')
    if cfg['gates'] != [.15, .25, .35] or cfg['ebn0_db'] != [6, 18]:
        raise ValueError('fixed candidate and SNR design required')
    if cfg['scenarios'] != ['stationary', 'weak_drift', 'gap32', 'alias64', 'dirty_calibration']:
        raise ValueError('fixed integration workloads required')
    dev, test = cfg['development']['seeds'], cfg['test']['seeds']
    if len(dev) != 4 or len(test) != 16 or len(set(dev+test)) != 20 or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in dev+test):
        raise ValueError('four development and sixteen independent test seeds required')
    used = set()
    for p in Path('experiments').glob('*.json'):
        if p.name != 'captured_tracking.json':
            used.update(seed_sets(json.loads(p.read_text())))
    if used & set(dev+test):
        raise ValueError('fresh episode seeds required')
    return validate_acquisition(json.loads(Path('experiments/acquisition.json').read_text()))


def grid(observed, start, epsilon):
    first = start+224
    corrected = rotate_frequency(observed[first:first+21*80], -epsilon, 64, first)
    return ofdm_demodulate(corrected, 64, 16)[1:]


def episode(protocol, scenario, db, si, seed, gates):
    layout = comb_grid(2)
    bins = layout.bins(layout.data_tones)
    ls_model = TapDomainLS(layout, 17)
    noise = pilot_noise_variance(db, layout, 16)
    receivers = dict(capture_adaptive=SlotTrackedLMMSE(layout), oracle_adaptive=SlotTrackedLMMSE(layout))
    for gate in gates:
        receivers[f'gate_{gate}'] = SlotTrackedLMMSE(layout, gate=gate)
        receivers[f'frozen_{gate}'] = SlotTrackedLMMSE(layout, gate=gate, adaptive=False)
    metrics = {k: [] for k in ['ls', *receivers]}
    # Counts are evaluator-only; no classification is fed back into step().
    audit = {k: dict(consumed_bad=0, calibration_bad=0, scored_ready=0, scored_captured=0,
                    scored_statistics_rejected=0, captured_bad=0, gap_updates=0) for k in receivers}
    mask = {k: [] for k in receivers}
    capture_records = []
    for slot in range(384):
        role, t = (1, slot) if slot < 64 else (2, slot-64)
        power = protocol['states']['initial']
        if scenario != 'stationary' and role == 2 and t >= 80:
            power = protocol['states']['weak15']
        epsilon = 5.2 if ((scenario == 'alias64' and role == 2 and 96 <= t < 160) or
                         (scenario == 'dirty_calibration' and role == 1 and 16 <= t < 48)) else 1.2
        missing = scenario == 'gap32' and role == 2 and 96 <= t < 128
        # Derived packet seeds are >2^31 and occupy disjoint arithmetic blocks;
        # role/t have separate offsets, with t<10000 for every configured run.
        packet_seed = seed*100000+role*10000+t
        latent, pilots, bits, start, _ = packet(packet_seed, si, power, noise)
        observed = rotate_frequency(latent, epsilon, 64)
        acquired_grid = true_grid = None
        correct = False
        if not missing:
            proposal = acquire(observed, .2)
            if proposal.accepted:
                acquired_grid = grid(observed, proposal.start, proposal.epsilon)
                dmax = max(i for i, p in enumerate(power) if p > 0)
                correct = dmax-16 <= proposal.start-start <= 0 and abs(proposal.epsilon-epsilon) <= .05
            true_grid = grid(observed, start, epsilon)
            zero_grid = grid(latent, start, 0.)
            np.testing.assert_allclose(true_grid, zero_grid, rtol=1e-10, atol=1e-10)
        if role == 2:
            capture_records.append(dict(frame=t, epsilon=epsilon, external_gap=int(missing),
                                        accepted=int(acquired_grid is not None), correct=int(correct)))
            if acquired_grid is None:
                metrics['ls'].append((bits.size, bits.size, 0))
            else:
                ls = ls_model.estimate(acquired_grid, pilots)
                decoded = qpsk_demodulate(zero_force(acquired_grid[:, bins], ls).ravel())
                metrics['ls'].append((int(np.count_nonzero(decoded != bits.ravel())), 0, bits.size))
        for name, receiver in receivers.items():
            y = true_grid if name == 'oracle_adaptive' else acquired_grid
            ready = receiver.tracker is not None
            h = receiver.step(y, pilots if y is not None else None, final=slot == 383)
            state = receiver.last
            bad = y is not None and name != 'oracle_adaptive' and not correct
            audit[name]['consumed_bad'] += int(state['statistics_consumed'] and bad)
            audit[name]['calibration_bad'] += int(state['statistics_consumed'] and not ready and bad)
            audit[name]['captured_bad'] += int(bad)
            audit[name]['gap_updates'] += int(missing and state['updated'])
            mask[name].append(int(state['statistics_consumed']))
            if role == 2:
                audit[name]['scored_ready'] += int(ready and h is not None)
                audit[name]['scored_captured'] += int(h is not None)
                audit[name]['scored_statistics_rejected'] += int(y is not None and not state['statistics_admissible'])
                if h is None:
                    metrics[name].append((bits.size, bits.size, 0))
                else:
                    decoded = qpsk_demodulate(zero_force(y[:, bins], h).ravel())
                    metrics[name].append((int(np.count_nonzero(decoded != bits.ravel())), 0, bits.size))
    costs, events = [], []
    for name, receiver in receivers.items():
        child = receiver.tracker
        if child is not None and isinstance(child, FollowupPilotLMMSE):
            if child.armed_plans != child.cancelled_plans+child.followup_count+int(child.pending_refit_after_frame is not None):
                raise AssertionError('plan identity broken')
        costs.append(dict(receiver=name, **audit[name], consumed=receiver.consumed, captured=receiver.captured,
                          missing=receiver.missing, gate_rejected=receiver.gate_rejected,
                          bootstrap_slot=receiver.bootstrap_slot if receiver.bootstrap_slot is not None else '',
                          updates=child.update_count if child else 0, followups=getattr(child, 'followup_count', 0),
                          pending_plan=int(getattr(child, 'pending_refit_after_frame', None) is not None),
                          maximum_history_span=receiver.maximum_history_span,
                          observation_clock=child.frames_observed if child else 0,
                          admitted_mask_sha256=hashlib.sha256(bytes(mask[name])).hexdigest()))
        events.extend(dict(receiver=name, **e) for e in receiver.events)
        if audit[name]['gap_updates'] != 0:
            raise AssertionError('missing slot triggered a fit')
        if any(e['applies_from_slot'] >= 384 for e in receiver.events):
            raise AssertionError('final unused model')
    return metrics, costs, events, capture_records


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['scenario'], r['ebn0_db'], r['receiver'], r['window']), []).append(r)
    result = []
    for key, g in groups.items():
        r = dict(zip(('scenario', 'ebn0_db', 'receiver', 'window'), key))
        r.update(episodes=len(g), offered_bits=sum(x['offered_bits'] for x in g),
                 accepted_bits=sum(x['accepted_bits'] for x in g), errors=sum(x['errors'] for x in g),
                 erased_bits=sum(x['erased_bits'] for x in g))
        r['accepted_ber'] = r['errors']/r['accepted_bits'] if r['accepted_bits'] else ''
        for metric in ('loss', 'paired_minus_capture', 'paired_minus_ls', 'paired_minus_oracle'):
            r[metric] = float(np.mean([x[metric] for x in g]))
            r[metric+'_se'] = se([x[metric] for x in g]) if len(g) > 1 else ''
        result.append(r)
    return result


def evaluate(cfg, protocol, split, gates):
    rows, costs, events, capture = [], [], [], []
    for scenario in cfg['scenarios']:
        for si, db in enumerate(cfg['ebn0_db']):
            for seed in cfg[split]['seeds']:
                values, counts, logs, detections = episode(protocol, scenario, db, si, seed, gates)
                context = dict(scenario=scenario, ebn0_db=db, seed=seed)
                costs.extend(dict(context, **r) for r in counts)
                events.extend(dict(context, **r) for r in logs)
                for window, (a, b) in {'all': (0, 320), 'early': (0, 96), 'impairment': (96, 160), 'late': (224, 320)}.items():
                    group = detections[a:b]
                    packed = bytes(int(r['external_gap'])+2*int(r['accepted'])+4*int(r['correct'])+8*int(r['epsilon'] == 5.2) for r in group)
                    capture.append(dict(context, window=window, frames=len(group),
                                        external_gaps=sum(r['external_gap'] for r in group),
                                        accepted=sum(r['accepted'] for r in group),
                                        correct=sum(r['correct'] for r in group),
                                        out_of_range=sum(r['epsilon'] == 5.2 for r in group),
                                        mask_sha256=hashlib.sha256(packed).hexdigest()))
                for method, v in values.items():
                    for window, (a, b) in {'all': (0, 320), 'late': (224, 320)}.items():
                        selected = v[a:b]
                        erasures = sum(x[1] for x in selected)
                        errors = sum(x[0] for x in selected)-erasures
                        offered = (b-a)*960
                        loss = (errors+erasures)/offered
                        row = dict(context, receiver=method, window=window, offered_bits=offered,
                                   accepted_bits=offered-erasures, errors=errors, erased_bits=erasures, loss=loss)
                        for ref, label in [('capture_adaptive', 'capture'), ('ls', 'ls'), ('oracle_adaptive', 'oracle')]:
                            row['paired_minus_'+label] = loss-sum(x[0] for x in values[ref][a:b])/offered
                        rows.append(row)
                print(json.dumps(context), flush=True)
    return dict(runs=rows, summary=summarize(rows), costs=costs, events=events, capture=capture)


def select(results, gates):
    index = {(r['scenario'], r['ebn0_db'], r['receiver']): r for r in results['summary'] if r['window'] == 'all'}
    candidates = []
    for gate in gates:
        rows = [r for r in results['summary'] if r['window'] == 'all' and r['receiver'] == f'gate_{gate}']
        worst = max(r['loss']/index[r['scenario'], r['ebn0_db'], 'capture_adaptive']['loss'] for r in rows)
        candidates.append(dict(gate=gate, worst_loss_ratio=worst, mean_loss=float(np.mean([r['loss'] for r in rows])), eligible=worst <= 1.05))
    eligible = [c for c in candidates if c['eligible']]
    chosen = min(eligible, key=lambda c: (c['mean_loss'], -c['gate']))['gate'] if eligible else max(gates)
    return chosen, candidates, bool(eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiments/captured_tracking.json'))
    parser.add_argument('--output', type=Path, default=Path('results/captured_tracking'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    protocol = validate(cfg)
    paths = (sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))
             +[args.config, Path('experiments/followup.json'), Path('experiments/acquisition.json'),
               Path('reports/acquisition/metadata.json'), Path('reports/acquisition/selection.json')])
    hashes = {p.as_posix(): digest(p) for p in paths}
    development = evaluate(cfg, protocol, 'development', cfg['gates'])
    chosen, candidates, eligible = select(development, cfg['gates'])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in development.items():
        write_csv(args.output/('development_'+name+'.csv'), rows)
    lock = dict(gate=chosen, candidates=candidates, eligible_candidate_found=eligible, config=cfg,
                source_sha256=hashes, locked_before_test_utc=datetime.now(timezone.utc).isoformat(),
                development_runs_sha256=digest(args.output/'development_runs.csv'),
                rule='Every development all-window cell loss <=1.05 capture-only adaptive; choose lowest equally weighted mean cell loss; tie larger gate; if none choose largest gate retaining failure')
    (args.output/'selection.json').write_text(json.dumps(lock, indent=2)+'\n')
    lock_hash = digest(args.output/'selection.json')
    started = datetime.now(timezone.utc).isoformat()
    result = evaluate(cfg, protocol, 'test', [chosen])
    if any(digest(p) != h for p, h in hashes.items()) or digest(args.output/'selection.json') != lock_hash:
        raise RuntimeError('source or development lock changed')
    for name, rows in result.items():
        write_csv(args.output/(name+'.csv'), rows)
    meta = dict(config=cfg, source_sha256=hashes, selection_sha256=lock_hash, test_started_at_utc=started,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                command=['python', '-m', 'experiments.captured_tracking_ber', *sys.argv[1:]], python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                clocks='Every physical slot advances wrapper; child advances only on admitted post-bootstrap observations; missing/gated slots neither zero-fill nor run cooldown/refits; final slot never consumed',
                bootstrap='First64 admitted observed bursts, LS until model ready NEXT slot; fixed initial64-slot budget, never extended; test scores include delayed bootstrap and LS fallback',
                receiver='Frozen FollowupPilotLMMSE behind acquisition0.2; gate filters only pilot statistics, every captured burst still decoded; frozen gated comparator uses identical bootstrap without model refits',
                gate='Pilot spatial LS residual energy / observed pilot energy, across all20 symbols; no true noise/PDP/timing used',
                scenarios='Initial64 calibration slots then320 scored slots; nonstationary states switch initial->weak15 at test80; gap32 removes receiver input at96..127; alias64 CFO5.2 at96..159; dirty_calibration CFO5.2 atcal16..47; otherwise CFO1.2',
                references='LS per-captured-burst; capture-only adaptation; quality-gated frozen/adaptive; oracle timing/CFO adaptive with same input gaps and budget, no true PDP/noise or error filtering',
                truth='Unsafe acquisition only labels posthoc contamination; never filters data or calibration; mismatched windows/residual CFO can invalidate diagonal NMSE formulas so none asserted',
                uncertainty='16 independent episode seeds per cell, states/scenarios share latent streams; whole-episode paired SE; no pooling frames as independent repeats',
                energy='224-point fixed header actual energy231.216221811, warmup65, scored1300; factor1.227858632 per transmitted burst, plus64 initial transmitted slots for320 scored slots; no extra gate pilots or retransmits',
                limits='Bounded slots, static taps perburst and independent between bursts, integer delays, no coding/CRC/hardware; accepted-clock policy can retain old physical-age data; no robust-superiority or runtime claim')
    (args.output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(dict(gate=chosen, eligible=eligible, test_episodes=160, scored_slots=51200)))


if __name__ == '__main__':
    main()
