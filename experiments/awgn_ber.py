"""Fixed-budget QPSK/OFDM AWGN benchmark. Run from the repository root."""
import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
import numpy as np
from wavelink import qpsk_modulate, qpsk_demodulate, ofdm_modulate, ofdm_demodulate
from wavelink.channel import add_awgn, qpsk_noise_variance, qpsk_ber, wilson_interval


def run(cfg):
    n, cp = cfg['n_fft'], cfg['cp_len']
    budget = cfg['bits_per_seed']
    if budget <= 0 or budget % (2*n) or cfg['batch_symbols'] <= 0:
        raise ValueError('budget must contain whole OFDM symbols; batch size must be positive')
    if not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('seeds must be nonempty and unique')
    rows = []
    for convention in cfg['conventions']:
        overhead = 1 + cp/n if convention == 'transmitted' else 1.0
        for point, db in enumerate(cfg['ebn0_db']):
            variance = qpsk_noise_variance(db, n, cp, convention)
            theory = qpsk_ber(db - 10*math.log10(overhead))
            for seed in cfg['seeds']:
                # Common random numbers across conventions; independent SNR points/seeds.
                rng = np.random.default_rng(np.random.SeedSequence([seed, point]))
                errors = count = 0
                while count < budget:
                    size = min(cfg['batch_symbols'] * 2*n, budget-count)
                    bits = rng.integers(0, 2, size, dtype=np.uint8)
                    grid = qpsk_modulate(bits).reshape(-1, n)
                    tx = ofdm_modulate(grid, cp)
                    rx = add_awgn(tx, variance, rng)
                    decisions = qpsk_demodulate(ofdm_demodulate(rx, n, cp).ravel())
                    errors += int(np.count_nonzero(bits != decisions))
                    count += size
                lo, hi = wilson_interval(errors, count)
                rows.append(dict(convention=convention, ebn0_db=db, seed=seed,
                    bits=count, errors=errors, ber=errors/count, ci95_low=lo, ci95_high=hi,
                    theory=theory, noise_variance=variance))
    summary = []
    # Bonferroni/Wilson approximate simultaneous 99% family coverage for all plotted points.
    points = len(cfg['conventions']) * len(cfg['ebn0_db'])
    for convention in cfg['conventions']:
        for db in cfg['ebn0_db']:
            group = [r for r in rows if r['convention'] == convention and r['ebn0_db'] == db]
            errors, bits = sum(r['errors'] for r in group), sum(r['bits'] for r in group)
            lo, hi = wilson_interval(errors, bits)
            flo, fhi = wilson_interval(errors, bits, 1 - 0.01/points)
            theory = group[0]['theory']
            summary.append(dict(convention=convention, ebn0_db=db, bits=bits, errors=errors,
                ber=errors/bits, ci95_low=lo, ci95_high=hi, theory=theory,
                family_low=flo, family_high=fhi, theory_in_family_interval=flo <= theory <= fhi))
    return rows, summary


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for convention, color in [('useful', '#1877B4'), ('transmitted', '#C65C10')]:
        group = [r for r in summary if r['convention'] == convention]
        if not group:
            continue
        x = np.array([r['ebn0_db'] for r in group])
        y = np.array([r['ber'] for r in group])
        low = np.array([r['ci95_low'] for r in group])
        high = np.array([r['ci95_high'] for r in group])
        ax.semilogy(x, [r['theory'] for r in group], '-', color=color, label=f'{convention}: theory')
        ax.errorbar(x, y, yerr=[y-low, high-y], fmt='o', color=color, capsize=4, label=f'{convention}: simulation (95% CI)')
    ax.set(xlabel='Eb/N0 (dB), using the indicated energy convention', ylabel='Bit error rate',
        title='Uncoded QPSK / OFDM in AWGN — ideal synchronization')
    ax.grid(True, which='both', alpha=.25)
    ax.legend()
    fig.savefig(output / 'ber.png', dpi=180)
    fig.savefig(output / 'ber.svg')
    svg = output / 'ber.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiments/awgn.json'))
    parser.add_argument('--output', type=Path, default=Path('results/awgn'))
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows, summary = run(cfg)
    write_csv(args.output / 'runs.csv', rows)
    write_csv(args.output / 'summary.csv', summary)
    sources = sorted(Path('src').rglob('*.py')) + [Path(__file__), args.config]
    metadata = dict(config=cfg, python=platform.python_version(), numpy=np.__version__,
        platform=platform.platform(), machine=platform.machine(),
        source_sha256={p.resolve().relative_to(Path.cwd()).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        stopping_rule='fixed bits per seed; no early stopping',
        interval='95% pointwise Wilson; approximate 99% Bonferroni-Wilson family interval')
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if args.plot:
        plot(summary, args.output)
    failures = [r for r in summary if not r['theory_in_family_interval']]
    print(f'{len(rows)} runs; {len(summary)-len(failures)}/{len(summary)} points consistent with theory at family interval')
    if failures:
        raise SystemExit('Theory consistency check failed; inspect results, do not tune seeds.')


if __name__ == '__main__':
    main()
