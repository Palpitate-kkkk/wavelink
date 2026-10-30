"""Frozen follow-up receiver CFO audit with a repeated training header."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from wavelink import qpsk_modulate, qpsk_demodulate, ofdm_modulate, ofdm_demodulate
from wavelink.receiver import comb_grid, apply_multipath, pilot_noise_variance, zero_force
from wavelink.followup import FollowupPilotLMMSE
from wavelink.synchronization import rotate_frequency, estimate_repeated_cfo
from experiments.density_ber import digest
from experiments.generalization_ber import timeline
from experiments.lmmse_ber import se
from experiments.multipath_ber import write_csv


OFFSETS = (0., .05, .2, 1.2)
BRANCHES = ('uncorrected', 'estimated', 'oracle')


def training(seed, si, role, frame, n_fft=64, cp=16, repeats=4):
    # Even FFT tones create an N/2-periodic waveform. sqrt(2) maintains
    # mean time-sample energy 1, versus 52/64 in the ordinary data waveform.
    rng = np.random.default_rng(np.random.SeedSequence([seed, si, role, frame, 3]))
    grid = np.zeros(n_fft, complex)
    grid[::2] = np.sqrt(2)*qpsk_modulate(rng.integers(0, 2, n_fft, dtype=np.uint8))
    half = np.fft.ifft(grid, norm='ortho')[:n_fft//2]
    useful = np.tile(half, repeats)
    return np.r_[useful[-cp:], useful] if cp else useful


def burst(seed, si, role, frame, powers, noise, cfg):
    """Generate one shared zero-CFO latent noisy waveform, with explicit header."""
    layout = comb_grid(2)
    def rng(part):
        return np.random.default_rng(np.random.SeedSequence([seed, si, role, frame, part]))
    r = rng(0)
    h = (r.normal(size=len(powers))+1j*r.normal(size=len(powers)))*np.sqrt(np.asarray(powers)/2)
    active = np.asarray(layout.active_tones)
    ns = cfg['symbols_per_frame']+1
    raw = rng(1).integers(0, 2, (ns, len(active), 2), dtype=np.uint8)
    symbols = qpsk_modulate(raw.ravel()).reshape(ns, -1)
    pi = np.searchsorted(active, layout.pilot_tones)
    di = np.searchsorted(active, layout.data_tones)
    payload = ofdm_modulate(layout.pack(symbols[:, di], symbols[:, pi]), cfg['cp_len'])
    header = training(seed, si, role, frame)
    clean, _ = apply_multipath(np.r_[header, payload], h)
    r = rng(2)
    standard = (r.normal(size=len(clean))+1j*r.normal(size=len(clean)))/np.sqrt(2)
    return clean+np.sqrt(noise)*standard, symbols[1:, pi], raw[1:, di, :].reshape(cfg['symbols_per_frame'], -1)


def receive(latent, epsilon, sample_offset, cfg):
    """Header first; all estimated decisions use header only, before payload."""
    n, cp, lag, useful = 64, cfg['cp_len'], 32, 128
    header_len = cp+useful
    observed = rotate_frequency(latent, epsilon, n, sample_offset)
    estimate, correlation = estimate_repeated_cfo(observed[cp:header_len], lag, n)
    values = {}
    for branch, correction in [('uncorrected', 0.), ('estimated', estimate), ('oracle', epsilon)]:
        corrected = rotate_frequency(observed, -correction, n, sample_offset)
        values[branch] = ofdm_demodulate(corrected[header_len:], n, cp)[1:]
    baseline = ofdm_demodulate(latent[header_len:], n, cp)[1:]
    np.testing.assert_allclose(values['oracle'], baseline, rtol=1e-10, atol=1e-10)
    return values, estimate, correlation


def validate(cfg):
    if cfg['protocol'] != 'experiments/followup.json' or digest(cfg['protocol']) != cfg['protocol_sha256']:
        raise ValueError('frozen followup protocol required')
    if digest('reports/followup/metadata.json') != cfg['baseline_metadata_sha256']:
        raise ValueError('frozen metadata required')
    old = json.loads(Path('reports/followup/metadata.json').read_text())
    for p, h in old['source_sha256'].items():
        if digest(p) != h:
            raise ValueError('frozen source changed: '+p)
    if cfg['offsets'] != list(OFFSETS) or cfg['ebn0_db'] != [6, 18] or cfg['scenarios'] != ['stationary', 'weak15']:
        raise ValueError('fixed CFO factorial design required')
    seeds = cfg['seeds']
    if len(seeds) != 16 or len(set(seeds)) != 16 or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError('sixteen distinct integer seeds required')
    used = set()
    for p in Path('experiments').glob('*.json'):
        if p.name == 'cfo.json':
            continue
        c = json.loads(p.read_text())
        used.update(c.get('seeds', []))
        for split in ('development', 'test'):
            used.update(c.get(split, {}).get('seeds', []))
    if used & set(seeds):
        raise ValueError('new CFO evaluation seeds required')
    return json.loads(Path(cfg['protocol']).read_text())


def episode(cfg, spec, db, si, seed, onset):
    layout = comb_grid(2)
    bins = layout.bins(layout.data_tones)
    noise0 = pilot_noise_variance(db, layout, cfg['cp_len'])
    cal = {(e, b): [] for e in OFFSETS for b in BRANCHES}
    diagnostics = []
    # Complete header and warmup samples count toward continuous oscillator phase.
    stride = 144+(cfg['symbols_per_frame']+1)*80
    for role, count in [(1, cfg['calibration_frames']), (2, cfg['episode_frames'])]:
        if role == 2:
            trackers = {key: FollowupPilotLMMSE(np.asarray(z), layout, 17, innovation_threshold=2., refit_delay=64) for key, z in cal.items()}
            metrics = {key: [] for key in trackers}
            events = []
            powers, noises = timeline(cfg, spec, onset)
        for t in range(count):
            power = cfg['states']['initial'] if role == 1 else powers[t]
            noise = noise0 if role == 1 else noise0*noises[t]
            latent, p, bits = burst(seed, si, role, t, power, noise, cfg)
            grids = {}
            for epsilon in OFFSETS:
                received, estimate, correlation = receive(latent, epsilon, t*stride, cfg)
                diagnostics.append(dict(role=role, frame=t, epsilon=epsilon, estimated_epsilon=estimate,
                                        error=estimate-epsilon, wrapped_error=(estimate-epsilon+1)%2-1,
                                        correlation=correlation, products=96))
                for branch, y in received.items():
                    key = epsilon, branch
                    if role == 1:
                        cal[key].append(y[:, layout.bins(layout.pilot_tones)]/p)
                    else:
                        grids[key] = y
            if role == 1:
                continue
            decoded = {}
            for key, tracker in trackers.items():
                if tracker.frames_observed != t or tracker.last_update_after_frame >= t:
                    raise AssertionError('noncausal channel statistics')
                y = grids[key]
                h = tracker.estimate(y, p)
                decoded[key] = qpsk_demodulate(zero_force(y[:, bins], h).ravel())
                metrics[key].append(int(np.count_nonzero(decoded[key] != bits.ravel())))
            for epsilon in OFFSETS:
                # Exact noise coupling makes all oracle compensation grids equal
                # to zero CFO; verify the complete adaptive hard-decision path.
                np.testing.assert_array_equal(decoded[epsilon, 'oracle'], decoded[0., 'oracle'])
            for key, tracker in trackers.items():
                if t < count-1 and tracker.advance():
                    events.append(dict(epsilon=key[0], branch=key[1], after_frame=t,
                                       applies_from_frame=t+1, event_kind=tracker.trigger_log[-1]['event_kind']))
        if role == 2:
            for epsilon in OFFSETS:
                a = [r['after_frame'] for r in events if r['epsilon'] == epsilon and r['branch'] == 'oracle']
                b = [r['after_frame'] for r in events if r['epsilon'] == 0. and r['branch'] == 'oracle']
                if a != b:
                    raise AssertionError('oracle adaptive path differs from zero CFO')
    return metrics, diagnostics, events, trackers


def summarize(runs):
    groups = {}
    for r in runs:
        key = tuple(r[k] for k in ('scenario', 'ebn0_db', 'epsilon', 'branch', 'window'))
        groups.setdefault(key, []).append(r)
    result = []
    for key, rows in groups.items():
        r = dict(zip(('scenario', 'ebn0_db', 'epsilon', 'branch', 'window'), key))
        r.update(episodes=len(rows), errors=sum(x['errors'] for x in rows), bits=sum(x['bits'] for x in rows))
        for metric in ('ber', 'paired_minus_oracle', 'paired_minus_uncorrected', 'paired_minus_zero_oracle'):
            r[metric] = float(np.mean([x[metric] for x in rows]))
            r[metric+'_se'] = se([x[metric] for x in rows])
        result.append(r)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiments/cfo.json'))
    parser.add_argument('--output', type=Path, default=Path('results/cfo'))
    args = parser.parse_args()
    audit = json.loads(args.config.read_text())
    cfg = validate(audit)
    paths = sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))+[args.config, Path(audit['protocol'])]
    hashes = {p.as_posix(): digest(p) for p in paths}
    started = datetime.now(timezone.utc).isoformat()
    args.output.mkdir(parents=True, exist_ok=True)
    # Persist the complete protocol before any new evaluation, no tuning stage.
    lock = dict(config=audit, source_sha256=hashes, locked_at_utc=started)
    (args.output/'protocol.json').write_text(json.dumps(lock, indent=2)+'\n')
    protocol_hash = digest(args.output/'protocol.json')
    runs, errors, event_rows, costs = [], [], [], []
    phases = np.random.default_rng(audit['schedule_seed']).permutation(32)[:16]
    specs = {s['name']: s for s in cfg['scenarios']}
    for scenario in audit['scenarios']:
        for si, db in enumerate(audit['ebn0_db']):
            for seed, phase in zip(audit['seeds'], phases):
                onset = 64+int(phase)
                context = dict(scenario=scenario, ebn0_db=db, seed=seed, onset_frame=onset, onset_phase=int(phase))
                metrics, diagnostics, events, trackers = episode(cfg, specs[scenario], db, si, seed, onset)
                event_rows.extend(dict(context, **r) for r in events)
                for role in (1, 2):
                    for epsilon in OFFSETS:
                        selected = [r for r in diagnostics if r['role'] == role and r['epsilon'] == epsilon]
                        errors.append(dict(context, role=role, epsilon=epsilon, frames=len(selected),
                            bias=float(np.mean([r['error'] for r in selected])), rmse=float(np.sqrt(np.mean([r['error']**2 for r in selected]))),
                            wrapped_rmse=float(np.sqrt(np.mean([r['wrapped_error']**2 for r in selected]))),
                            mean_correlation=float(np.mean([r['correlation'] for r in selected])),
                            large_error_frames=sum(abs(r['error'])>.1 for r in selected)))
                windows = {'pre': (onset-32, onset), 'post_all': (onset, onset+224)}
                for key, counts in metrics.items():
                    epsilon, branch = key
                    tracker = trackers[key]
                    costs.append(dict(context, epsilon=epsilon, branch=branch, updates=tracker.update_count,
                                      followups=tracker.followup_count, pending_plan=int(tracker.pending_refit_after_frame is not None),
                                      last_update_after_frame=tracker.last_update_after_frame))
                    for window, (a, b) in windows.items():
                        bits = (b-a)*cfg['symbols_per_frame']*48
                        error = sum(counts[a:b])
                        row = dict(context, epsilon=epsilon, branch=branch, window=window, bits=bits, errors=error, ber=error/bits)
                        for ref, label in [((epsilon, 'oracle'), 'oracle'), ((epsilon, 'uncorrected'), 'uncorrected'), ((0., 'oracle'), 'zero_oracle')]:
                            row['paired_minus_'+label] = (error-sum(metrics[ref][a:b]))/bits
                        runs.append(row)
                print(json.dumps(context), flush=True)
    if any(digest(p) != h for p, h in hashes.items()) or digest(args.output/'protocol.json') != protocol_hash:
        raise RuntimeError('source or locked protocol changed')
    for name, rows in [('runs', runs), ('summary', summarize(runs)), ('frequency_errors', errors), ('events', event_rows), ('costs', costs)]:
        write_csv(args.output/(name+'.csv'), rows)
    meta = dict(config=audit, source_sha256=hashes, protocol_sha256=protocol_hash,
                base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                command=['python', '-m', 'experiments.cfo_ber', *sys.argv[1:]], started_at_utc=started,
                finished_at_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                sample_rate_hz=1000000, subcarrier_spacing_hz=15625, epsilon_units='CFO / subcarrier spacing',
                training='144 samples: CP16 plus four repeated 32-sample blocks; expected sample energy 1; 96 overlapping lag32 products',
                noise_coupling='CFO rotates clean waveform plus latent circular AWGN; marginally equivalent to receiver AWGN added after CFO, paired oracle exactly restores zero-CFO latent samples',
                causality='Training header precedes payload, same header/noise input for every branch; no scoring or true CFO in estimated branch; per-branch calibration through its own front end; no final unused update',
                energy='Nominal Eb/N0 holds data/pilot/CP convention fixed; including one warmup OFDM block and header increases transmitted expected energy per scored bit by (21*65+144)/(20*65)=1.1607692307692308; all branches pay same header',
                scope='Known boundaries, constant CFO across one episode, frame-static integer-delay independent taps; 16 of32 phases; no packet detection, timing offsets, Doppler or hardware',
                uncertainty='Independent seed/SNR branches with paired episode SE; scenarios and offsets share latent streams; no treating frames or overlapping correlation products as independent repeats',
                theory='Diagonal-channel NMSE and AWGN-only BER formulas are not asserted under residual CFO/ICI; oracle equality to zero CFO checked numerically for all frames and decisions/events')
    (args.output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(dict(episodes=len(audit['seeds'])*4, summary_cells=len(summarize(runs)), oracle_grid_checks=len(audit['seeds'])*4*384*4)))


if __name__ == '__main__':
    main()
