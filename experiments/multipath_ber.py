"""Static-channel benchmark with shared realizations and independent frames."""
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
from wavelink.channel import add_awgn
from wavelink.receiver import comb_grid, apply_multipath, ls_channel_estimate, zero_force, pilot_noise_variance


def channel_taps(spec):
    delays = np.asarray(spec['delays'])
    gains = np.asarray([complex(*g) for g in spec['gains']])
    if delays.ndim != 1 or len(delays) != len(gains) or np.any(delays < 0) or not np.issubdtype(delays.dtype, np.integer) or len(set(delays)) != len(delays):
        raise ValueError('invalid channel delays')
    taps = np.zeros(int(max(delays))+1, complex)
    taps[delays] = gains
    energy = np.sum(abs(taps)**2)
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError('invalid gains')
    return taps/np.sqrt(energy)


def run(cfg):
    if cfg['frames_per_seed'] < 2 or cfg['symbols_per_frame'] < 1:
        raise ValueError('need at least two independent frames and one scored symbol')
    if len(set(cfg['seeds'])) != len(cfg['seeds']) or not cfg['seeds']:
        raise ValueError('unique seeds required')
    rows = []
    for ci, spec in enumerate(cfg['channels']):
        taps = channel_taps(spec)
        if len(taps) > 64:
            raise ValueError('this benchmark requires channel memory shorter than a symbol')
        for spacing in cfg['pilot_spacings']:
            layout = comb_grid(spacing)
            nd, npilot = len(layout.data_tones), len(layout.pilot_tones)
            bins = layout.bins(layout.data_tones)
            truth = np.fft.fft(taps, layout.n_fft)[bins]
            for cp in cfg['cp_lengths']:
                for si, db in enumerate(cfg['ebn0_db']):
                    n0 = pilot_noise_variance(db, layout, cp)
                    oracle_theory = float(np.mean([.5*math.erfc(abs(h)/math.sqrt(2*n0)) for h in truth])) if cp >= len(taps)-1 else ''
                    for seed in cfg['seeds']:
                        rng = np.random.default_rng(np.random.SeedSequence([seed, ci, spacing, cp, si]))
                        stats = {name: dict(errors=0, frame_ber_sum2=0., nmse_sum=0., nmse_sum2=0.) for name in ('none', 'oracle_zf', 'ls_linear_zf')}
                        for _ in range(cfg['frames_per_seed']):
                            # One independent warm-up symbol populates FIR history; not scored.
                            ns = cfg['symbols_per_frame']+1
                            bits = rng.integers(0, 2, (ns, nd*2), dtype=np.uint8)
                            data = qpsk_modulate(bits.ravel()).reshape(ns, nd)
                            pilots = qpsk_modulate(rng.integers(0,2,ns*npilot*2)).reshape(ns,npilot)
                            grid = layout.pack(data, pilots)
                            tx = ofdm_modulate(grid, cp)
                            propagated, _ = apply_multipath(tx, taps)
                            noisy = add_awgn(propagated, n0, rng)
                            received = ofdm_demodulate(noisy, 64, cp)[1:]
                            yd = received[:, bins]
                            estimate = ls_channel_estimate(received, pilots[1:], layout)
                            outputs = dict(none=yd, oracle_zf=zero_force(yd, truth), ls_linear_zf=zero_force(yd, estimate))
                            nmse = float(np.sum(abs(estimate-truth)**2)/(len(received)*np.sum(abs(truth)**2)))
                            for name, symbols in outputs.items():
                                errors = int(np.count_nonzero(qpsk_demodulate(symbols.ravel()) != bits[1:].ravel()))
                                metric = nmse if name == 'ls_linear_zf' else (0. if name == 'oracle_zf' else None)
                                s = stats[name]
                                s['errors'] += errors
                                s['frame_ber_sum2'] += (errors/bits[1:].size)**2
                                if metric is not None:
                                    s['nmse_sum'] += metric
                                    s['nmse_sum2'] += metric**2
                        frames = cfg['frames_per_seed']
                        total = frames*cfg['symbols_per_frame']*nd*2
                        for name, s in stats.items():
                            rows.append(dict(channel=spec['name'], pilot_spacing=spacing, cp_len=cp,
                                cp_sufficient=cp >= len(taps)-1, ebn0_db=db, receiver=name, seed=seed,
                                n_data=nd, n_pilots=npilot, frames=frames, bits=total, noise_variance=n0,
                                oracle_theory=oracle_theory if name == 'oracle_zf' else '',
                                errors=s['errors'], ber=s['errors']/total, frame_ber_sum2=s['frame_ber_sum2'],
                                nmse_sum=s['nmse_sum'] if name != 'none' else '',
                                nmse_sum2=s['nmse_sum2'] if name != 'none' else ''))
    summary = []
    keys = ('channel', 'pilot_spacing', 'cp_len', 'ebn0_db', 'receiver')
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    for key, group in groups.items():
        frames = sum(r['frames'] for r in group)
        bits, errors = sum(r['bits'] for r in group), sum(r['errors'] for r in group)
        ber = errors/bits
        variance = max(0., (sum(r['frame_ber_sum2'] for r in group)-frames*ber**2)/(frames-1))
        metric = '' if key[-1] == 'none' else sum(r['nmse_sum'] for r in group)/frames
        se_nmse = '' if metric == '' else np.sqrt(max(0., (sum(r['nmse_sum2'] for r in group)-frames*metric**2)/(frames-1))/frames)
        theory = group[0]['oracle_theory']
        consistent = '' if theory == '' else abs(errors-bits*theory) <= 6*np.sqrt(bits*theory)+5
        summary.append(dict(zip(keys,key), oracle_theory=theory, oracle_consistent=consistent, frames=frames, bits=bits, errors=errors, ber=ber,
                            frame_ber_se=np.sqrt(variance/frames), data_nmse=metric, frame_nmse_se=se_nmse))
    return rows, summary


def diagnostics(cfg):
    """Noise-free matched-grid ablation separates CP failure from noise overhead."""
    rows=[]
    for spec in cfg['channels']:
        taps=channel_taps(spec)
        for spacing in cfg['pilot_spacings']:
            layout=comb_grid(spacing)
            rng=np.random.default_rng(991+spacing)
            ns=128
            data=qpsk_modulate(rng.integers(0,2,ns*len(layout.data_tones)*2)).reshape(ns,-1)
            pilots=qpsk_modulate(rng.integers(0,2,ns*len(layout.pilot_tones)*2)).reshape(ns,-1)
            grid=layout.pack(data,pilots)
            h=np.fft.fft(taps,64)
            expected=grid[1:]*h
            for cp in cfg['cp_lengths']:
                tx=ofdm_modulate(grid,cp)
                y,_=apply_multipath(tx,taps)
                received=ofdm_demodulate(y,64,cp)[1:]
                hd=h[layout.bins(layout.data_tones)]
                est=ls_channel_estimate(received,pilots[1:],layout)
                rows.append(dict(channel=spec['name'],pilot_spacing=spacing,cp_len=cp,
                    circular_model_nmse=float(np.sum(abs(received-expected)**2)/np.sum(abs(expected)**2)),
                    noiseless_ls_data_nmse=float(np.sum(abs(est-hd)**2)/(len(est)*np.sum(abs(hd)**2)))))
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, cfg, out):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.fonttype'] = 'none'
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for column, spec in enumerate(cfg['channels']):
        name = spec['name']
        ax = axes[0,column]
        for receiver, label, marker in [('none','No equalization','x'),('oracle_zf','Oracle CSI + ZF','o'),('ls_linear_zf','Pilot LS + linear + ZF','s')]:
            group = [r for r in summary if r['channel']==name and r['cp_len']==16 and r['pilot_spacing']==4 and r['receiver']==receiver]
            if not group:
                continue
            x = [r['ebn0_db'] for r in group]
            y = [max(r['ber'],.5/r['bits']) for r in group]
            ax.semilogy(x,y,marker=marker,label=label)
            for r, value in zip(group,y):
                if r['errors']==0 and receiver == 'ls_linear_zf':
                    ax.annotate('0 observed errors',(r['ebn0_db'],value),xytext=(-8,10),ha='right',textcoords='offset points',fontsize=8)
        ax.set(title=f'{name}: CP=16, pilot spacing=4', xlabel='Total transmitted Eb/N0 (dB)', ylabel='Data BER')
        ax.legend(fontsize=8)
        ax = axes[1,column]
        for cp in cfg['cp_lengths']:
            for spacing in cfg['pilot_spacings']:
                group = [r for r in summary if r['channel']==name and r['cp_len']==cp and r['pilot_spacing']==spacing and r['receiver']=='ls_linear_zf']
                ax.semilogy([r['ebn0_db'] for r in group],[r['data_nmse'] for r in group], marker='o', linestyle='-' if cp==16 else '--',label=f'CP={cp}, spacing={spacing}')
        ax.set(title=f'{name}: LS estimation on data tones',xlabel='Total transmitted Eb/N0 (dB)',ylabel='Channel NMSE')
        ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(True,which='both',alpha=.25)
    fig.suptitle('Static multipath, comb pilots, ideal synchronization\nZero-error BER points shown at 0.5 / tested bits (display only)',fontsize=13)
    fig.savefig(out/'multipath.png',dpi=160)
    fig.savefig(out/'multipath.svg')
    svg=out/'multipath.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('experiments/multipath.json'))
    parser.add_argument('--output',type=Path,default=Path('results/multipath'))
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    rows, summary=run(cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'runs.csv',rows)
    write_csv(args.output/'summary.csv',summary)
    write_csv(args.output/'diagnostics.csv',diagnostics(cfg))
    sources=sorted(Path('src').rglob('*.py'))+[Path(__file__),args.config]
    meta=dict(config=cfg, python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        source_sha256={p.resolve().relative_to(Path.cwd()).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        normalization='Channel taps normalized once per profile to unit energy; no per-frame receive-power normalization',
        energy_convention='Expected transmitted steady-state energy per data bit, includes pilots and CP; warm-up is simulation boundary setup',
        uncertainty='SE across independent frames conditional on fixed taps; not iid bit-binomial confidence intervals',
        stopping_rule='fixed frames and scored symbols; no early stopping')
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.plot:
        plot(summary,cfg,args.output)
    checks=[r for r in summary if r['oracle_consistent'] != '']
    print(f'{len(rows)} receiver-run rows; {len(summary)} summary points; {sum(r["oracle_consistent"] for r in checks)}/{len(checks)} oracle theory checks passed')
    if not all(r['oracle_consistent'] for r in checks):
        raise SystemExit('Oracle baseline mismatch: inspect the model, do not tune seeds')


if __name__=='__main__':
    main()
