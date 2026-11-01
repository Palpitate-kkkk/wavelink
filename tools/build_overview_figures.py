"""Build scientific overview SVGs from saved CSVs; no new receiver simulation."""
import csv
import argparse
import sys
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return list(csv.DictReader(Path(path).read_text().splitlines()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    if args.preview_dir: args.preview_dir.mkdir(parents=True, exist_ok=True)
    output = Path('reports/overview')
    output.mkdir(parents=True, exist_ok=True)
    sources = ['reports/reference/summary.csv', 'reports/reference/windows.csv', 'reports/support/summary.csv', 'tools/build_overview_figures.py']
    hashes = {p: digest(p) for p in sources}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.hashsalt': 'wavelink-overview-v023'})
    def save(fig, name):
        fig.savefig(output/name, metadata={'Date': None}, bbox_inches='tight')
        svg_path = output/name
        svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
        if args.preview_dir:
            fig.savefig(args.preview_dir/(Path(name).stem+'.png'), dpi=140, bbox_inches='tight')
        plt.close(fig)
    rows = read(sources[0])
    scenes = ['stationary', 'weak_drift', 'gap32', 'alias64', 'dirty_calibration', 'weak_first', 'weaker_first']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    for ax, db in zip(axes, ['6', '18']):
        g = [next(r for r in rows if (r['scenario'], r['ebn0_db'], r['receiver'], r['window'])==(s, db, 'reference_adaptive', 'all')) for s in scenes]
        ax.errorbar(range(7), [float(r['paired_minus_raw']) for r in g], yerr=[float(r['paired_minus_raw_se']) for r in g],
                    fmt='o', capsize=4, color='#17677d')
        ax.axhline(0, color='#555555', linewidth=.8)
        ax.set(title=f'Eb/N0 = {db} dB; mean +/- paired SE', ylabel='Reference minus raw adaptive loss')
        ax.set_xticks(range(7), [s.replace('_', '\n') for s in scenes], fontsize=8)
        ax.grid(axis='y', alpha=.2)
    save(fig, 'paired_loss.svg')
    windows = read(sources[1])
    groups = [(s, db) for s in ['weak_first', 'weaker_first'] for db in ['6', '18']]
    fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
    x = np.arange(4)
    for offset, field, label, color in [(-.25, 'raw_safe', 'Original window', '#7b8994'),
                                      (0, 'physical_safe', 'Moved physical window', '#c67948'),
                                      (.25, 'reference_safe', 'Reference only', '#17677d')]:
        values = [np.mean([int(r[field]) for r in windows if (r['scenario'],r['ebn0_db'])==g]) for g in groups]
        ax.bar(x+offset, values, width=.24, label=label, color=color)
    ax.set_xticks(x, [f'{s}\n{db} dB' for s, db in groups])
    ax.set(ylabel='Mean CP/CFO-safe accepted packets per 320 offered slots', ylim=(0, 330), title='Weak-first-path timing audit; 16 roots per cell')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=.2)
    save(fig, 'weak_path_windows.svg')
    support = read(sources[2])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained', sharey=True)
    for ax, profile in zip(axes, ['initial', 'weak_first', 'weaker_first']):
        for model, color in [('causal', '#c67948'), ('centered', '#56885e'), ('negative', '#17677d'), ('transported', '#8a819c')]:
            g = sorted([r for r in support if r['profile']==profile and r['estimator']==model], key=lambda r:int(r['shift']))
            ax.plot([int(r['shift']) for r in g], [float(r['data_bias_nmse']) for r in g], label=model, color=color, linewidth=1.6)
        ax.set(title=profile, xlabel='Specified phase-reference shift (samples)', ylim=(0, None))
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean channel bias NMSE (noise excluded)')
    axes[-1].legend(fontsize=8)
    save(fig, 'support_bias.svg')
    assert all(digest(p)==h for p,h in hashes.items())
    meta = dict(source_sha256=hashes, artifact_sha256={p.name: digest(p) for p in sorted(output.glob('*.svg'))},
                python=platform.python_version(), matplotlib=matplotlib.__version__, numpy=np.__version__,
                base_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                command=['python','tools/build_overview_figures.py', *sys.argv[1:]], scope='Plots of existing fixed reports only; no additional statistical trials; paired error bars are one standard error, not confidence intervals.')
    (output/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')


if __name__ == '__main__':
    main()
