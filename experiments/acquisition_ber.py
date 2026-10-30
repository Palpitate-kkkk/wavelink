"""Development-locked bounded-slot packet acquisition with erasure accounting."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from wavelink import qpsk_modulate, qpsk_demodulate, ofdm_modulate, ofdm_demodulate
from wavelink.acquisition import acquire, acquisition_header
from wavelink.synchronization import rotate_frequency, estimate_repeated_cfo
from wavelink.receiver import comb_grid, apply_multipath, pilot_noise_variance, zero_force
from wavelink.estimation import TapDomainLS
from experiments.density_ber import digest
from experiments.lmmse_ber import se
from experiments.multipath_ber import write_csv


OFFSETS = (0., .2, 1.2, -1.2, 3.2, 5.2)
BUFFER_SAMPLES = 1984


def validate(cfg):
    old_path = Path('reports/cfo/metadata.json')
    if digest(old_path) != cfg['baseline_metadata_sha256']:
        raise ValueError('frozen CFO metadata required')
    old = json.loads(old_path.read_text())
    for path, expected in old['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError('frozen source changed: '+path)
    if cfg['offsets'] != list(OFFSETS) or cfg['ebn0_db'] != [6, 18] or cfg['states'] != ['initial', 'weak15']:
        raise ValueError('fixed acquisition factorial required')
    if cfg['thresholds'] != [.05, .1, .2, .3, .4, .5] or cfg['search_count'] != 64:
        raise ValueError('fixed score candidates and search span required')
    pools = [cfg[k]['seeds'] for k in ('development', 'test', 'null_development', 'null_test')]
    if list(map(len, pools)) != [8, 32, 128, 512]:
        raise ValueError('fixed development/test budgets required')
    new = [x for pool in pools for x in pool]
    if len(set(new)) != len(new) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in new):
        raise ValueError('distinct integer seeds required')
    used = set()
    for path in Path('experiments').glob('*.json'):
        if path.name == 'acquisition.json':
            continue
        c = json.loads(path.read_text())
        used.update(c.get('seeds', []))
        for split in ('development', 'test'):
            used.update(c.get(split, {}).get('seeds', []))
    if used & set(new):
        raise ValueError('fresh disjoint seed pools required')
    return json.loads(Path('experiments/followup.json').read_text())


def packet(seed, si, state, noise):
    """Return latent waveform, known pilots and scorer-only bits/start/taps."""
    def rng(part):
        return np.random.default_rng(np.random.SeedSequence([seed, si, part]))
    layout = comb_grid(2)
    onset = int(rng(3).integers(0, 64))
    r = rng(0)
    h = (r.normal(size=len(state))+1j*r.normal(size=len(state)))*np.sqrt(np.asarray(state)/2)
    raw = rng(1).integers(0, 2, (21, len(layout.active_tones), 2), dtype=np.uint8)
    symbols = qpsk_modulate(raw.ravel()).reshape(21, -1)
    active = np.asarray(layout.active_tones)
    pi, di = np.searchsorted(active, layout.pilot_tones), np.searchsorted(active, layout.data_tones)
    payload = ofdm_modulate(layout.pack(symbols[:, di], symbols[:, pi]), 16)
    tx = np.zeros(BUFFER_SAMPLES, complex)
    sent = np.r_[acquisition_header(), payload]
    tx[onset:onset+len(sent)] = sent
    clean, _ = apply_multipath(tx, h)
    r = rng(2)
    standard = (r.normal(size=len(tx))+1j*r.normal(size=len(tx)))/np.sqrt(2)
    return clean+np.sqrt(noise)*standard, symbols[1:, pi], raw[1:, di, :].reshape(20, -1), onset, h


def decode(observed, pilots, start, epsilon, model):
    """Decode every accepted proposal, including wrong timing/frequency ones."""
    first = start+224
    samples = rotate_frequency(observed[first:first+21*80], -epsilon, 64, first)
    y = ofdm_demodulate(samples, 64, 16)[1:]
    h = model.estimate(y, pilots)
    data = y[:, model.layout.bins(model.layout.data_tones)]
    return qpsk_demodulate(zero_force(data, h).ravel())


def signal_rows(cfg, protocol, split):
    layout = comb_grid(2)
    model = TapDomainLS(layout, 17)
    rows = []
    offsets = OFFSETS[:-1] if split == 'development' else OFFSETS
    for state in cfg['states']:
        power = protocol['states'][state]
        max_delay = max(i for i, p in enumerate(power) if p > 0)
        for si, db in enumerate(cfg['ebn0_db']):
            noise = pilot_noise_variance(db, layout, 16)
            for seed in cfg[split]['seeds']:
                latent, pilots, bits, onset, _ = packet(seed, si, power, noise)
                zero = decode(latent, pilots, onset, 0., model)
                for epsilon in offsets:
                    observed = rotate_frequency(latent, epsilon, 64)
                    proposal = acquire(observed, 0., cfg['search_count'])
                    decoded = decode(observed, pilots, proposal.start, proposal.epsilon, model)
                    oracle = decode(observed, pilots, onset, epsilon, model)
                    np.testing.assert_array_equal(oracle, zero)
                    # Old-style fractional-only receiver still has oracle timing;
                    # it is a diagnostic, not a realizable timing baseline.
                    fractional, _ = estimate_repeated_cfo(observed[onset+16:onset+144], 32, 64)
                    fraction_bits = decode(observed, pilots, onset, fractional, model)
                    difference = proposal.start-onset
                    row = dict(state=state, ebn0_db=db, seed=seed, epsilon=epsilon, true_start=onset,
                               **{('estimated_epsilon' if k == 'epsilon' else k): v for k, v in asdict(proposal).items()}, timing_error=difference,
                               timing_safe=int(max_delay-16 <= difference <= 0),
                               strict_timing=int(difference == 0), frequency_error=proposal.epsilon-epsilon,
                               capture_correct=int(max_delay-16 <= difference <= 0 and abs(proposal.epsilon-epsilon) <= .05),
                               bits=bits.size, proposal_errors=int(np.count_nonzero(decoded != bits.ravel())),
                               oracle_errors=int(np.count_nonzero(oracle != bits.ravel())),
                               fractional_errors=int(np.count_nonzero(fraction_bits != bits.ravel())))
                    rows.append(row)
    return rows


def null_rows(cfg, split):
    layout = comb_grid(2)
    rows = []
    for si, db in enumerate(cfg['ebn0_db']):
        noise = pilot_noise_variance(db, layout, 16)
        for seed in cfg[split]['seeds']:
            rng = np.random.default_rng(np.random.SeedSequence([seed, si, 2]))
            samples = np.sqrt(noise/2)*(rng.normal(size=BUFFER_SAMPLES)+1j*rng.normal(size=BUFFER_SAMPLES))
            rows.append(dict(ebn0_db=db, seed=seed, **asdict(acquire(samples, 0., cfg['search_count']))))
    return rows


def select(signals, nulls, thresholds):
    """No test inputs accepted. Minimize offered-bit loss under null FAP gate."""
    candidates = []
    for threshold in thresholds:
        false_alarm = float(np.mean([r['score'] >= threshold for r in nulls]))
        # Decode failures are erasures with loss=1; wrong accepted proposals
        # retain actual errors. No true timing/CFO gate removes a received bit.
        losses = [r['proposal_errors']/r['bits'] if r['score'] >= threshold else 1. for r in signals]
        candidates.append(dict(threshold=threshold, null_false_alarm=false_alarm,
                               mean_offered_bit_loss=float(np.mean(losses)), eligible=false_alarm <= .01))
    eligible = [c for c in candidates if c['eligible']]
    chosen = min(eligible, key=lambda c: (c['mean_offered_bit_loss'], -c['threshold']))['threshold'] if eligible else max(thresholds)
    return chosen, candidates, bool(eligible)


def apply_threshold(rows, threshold):
    result = []
    for original in rows:
        r = dict(original)
        accepted = bool(r['score'] > 0 and r['score'] >= threshold)
        r['accepted'] = accepted
        r['threshold'] = threshold
        r['accepted_bits'] = r['bits'] if accepted else 0
        r['errors'] = r['proposal_errors'] if accepted else 0
        r['erased_bits'] = 0 if accepted else r['bits']
        r['offered_bit_loss'] = (r['errors']+r['erased_bits'])/r['bits']
        r['paired_loss_minus_oracle'] = r['offered_bit_loss']-r['oracle_errors']/r['bits']
        r['successful_capture'] = int(accepted and r['capture_correct'])
        r['false_capture'] = int(accepted and not r['capture_correct'])
        result.append(r)
    return result


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['state'], r['ebn0_db'], r['epsilon']), []).append(r)
    result = []
    for (state, db, epsilon), group in groups.items():
        r = dict(state=state, ebn0_db=db, epsilon=epsilon, packets=len(group),
                 offered_bits=sum(x['bits'] for x in group), accepted_bits=sum(x['accepted_bits'] for x in group),
                 errors=sum(x['errors'] for x in group), erased_bits=sum(x['erased_bits'] for x in group))
        r['accepted_ber'] = r['errors']/r['accepted_bits'] if r['accepted_bits'] else ''
        r['oracle_ber'] = sum(x['oracle_errors'] for x in group)/r['offered_bits']
        r['known_timing_fractional_ber'] = sum(x['fractional_errors'] for x in group)/r['offered_bits']
        for metric in ('offered_bit_loss', 'paired_loss_minus_oracle', 'successful_capture', 'false_capture', 'timing_safe', 'strict_timing'):
            values = [x[metric] for x in group]
            r[metric] = float(np.mean(values))
            r[metric+'_se'] = se(values) if len(values) > 1 else ''
        r['miss_rate'] = float(np.mean([not x['accepted'] for x in group]))
        result.append(r)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiments/acquisition.json'))
    parser.add_argument('--output', type=Path, default=Path('results/acquisition'))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    protocol = validate(cfg)
    paths = sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))+[args.config]
    hashes = {p.as_posix(): digest(p) for p in paths}
    args.output.mkdir(parents=True, exist_ok=True)
    development = signal_rows(cfg, protocol, 'development')
    dev_null = null_rows(cfg, 'null_development')
    threshold, candidates, eligible = select(development, dev_null, cfg['thresholds'])
    for name, rows in [('development', development), ('development_null', dev_null)]:
        write_csv(args.output/(name+'.csv'), rows)
    lock = dict(threshold=threshold, candidates=candidates, eligible_candidate_found=eligible,
                source_sha256=hashes, config=cfg, locked_before_test_utc=datetime.now(timezone.utc).isoformat(),
                development_sha256=digest(args.output/'development.csv'), null_development_sha256=digest(args.output/'development_null.csv'),
                rule='null FAP<=0.01 on development, then minimum mean offered-bit loss; tie higher threshold; if none eligible choose highest threshold and retain failure')
    (args.output/'selection.json').write_text(json.dumps(lock, indent=2)+'\n')
    lock_hash = digest(args.output/'selection.json')
    started = datetime.now(timezone.utc).isoformat()
    test = apply_threshold(signal_rows(cfg, protocol, 'test'), threshold)
    nulls = null_rows(cfg, 'null_test')
    for r in nulls:
        r['accepted'] = bool(r['score'] > 0 and r['score'] >= threshold)
        r['threshold'] = threshold
    null_summary = []
    for db in cfg['ebn0_db']:
        selected = [r for r in nulls if r['ebn0_db'] == db]
        count = sum(r['accepted'] for r in selected)
        # Exact one-sided 95% upper bound for zero observed false alarms.
        null_summary.append(dict(ebn0_db=db, slots=len(selected), false_alarms=count,
                                 empirical_fap=count/len(selected), zero_count_upper95=1-.05**(1/len(selected)) if count == 0 else ''))
    if any(digest(p) != h for p, h in hashes.items()) or digest(args.output/'selection.json') != lock_hash:
        raise RuntimeError('source or selection changed after development lock')
    for name, rows in [('runs', test), ('summary', summarize(test)), ('null_runs', nulls), ('null_summary', null_summary)]:
        write_csv(args.output/(name+'.csv'), rows)
    meta = dict(config=cfg, source_sha256=hashes, selection_sha256=lock_hash, test_started_at_utc=started,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                command=['python', '-m', 'experiments.acquisition_ber', *sys.argv[1:]], python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                training='224 samples: fixed repeated CP16+128, unique full-tone OFDM CP16+64; fixed actual header energy measured below; DC and guard tones permitted in training',
                header_energy=float(np.sum(abs(acquisition_header())**2)),
                transmitted_energy_factor=float((21*65+np.sum(abs(acquisition_header())**2))/(20*65)),
                receiver='Fixed pilot LS L17 + ZF + QPSK on accepted bursts, no adaptive statistics; all proposals decoded before scorer classifies safe timing/CFO',
                score='Squared repeated correlation times rank17 marker projection energy fraction; heuristic bounded in[0,1], not a calibrated likelihood or p-value',
                causality='Only first287 samples enter acquisition; first scored FFT useful sample at true_start+320; known slot/payload length/pilots only; no true timing/CFO/PDP in acquire',
                loss='All rejected packets incur erased bits; accepted wrong proposals retain actual bit errors; offered_bit_loss=(errors+erasures)/offered_bits, not BER; no truth-based filtering',
                timing='Scorer-only ISI-free interval [dmax-CP,0] relative to TX start; strict start equality reported separately; support bound17 fixed at receiver',
                cost='Fixed measured header energy + expected warmup65 + scored1300; noise-only prefix/tail radiate no TX energy; training CP energy is measured, not replaced by ensemble mean',
                scope='One packet or AWGN-only bounded slot; signal starts0..63, integer taps0..16, CFO candidates coarse[-1,1] plus[-4,-2,0,2,4]; no interferer, timing drift, phase noise, arbitrary stream or hardware',
                uncertainty='Per-cell32 independent packet seeds; states/offsets share latent streams; null slots512 perSNR with same seeds acrossSNR, no pooled independent-count claim',
                baseline='Known true boundary plus fractional-only compensation diagnoses integer ambiguity; full true-boundary/true-CFO LS reference checked against zero-CFO decisions for every packet',
                limitations='Adaptive follow-up receiver is not integrated: missing-frame scheduling and covariance estimation need a separate causal audit')
    (args.output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(dict(threshold=threshold, eligible=eligible, test_packets=len(test), test_null_slots=len(nulls), oracle_decision_checks=len(test))))


if __name__ == '__main__':
    main()
