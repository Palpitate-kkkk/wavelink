"""Length-mismatch experiment. Run: python -m experiments.tapls_ber --plot"""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
from experiments.multipath_ber import run, channel_taps, write_csv
from wavelink.receiver import comb_grid, pilot_noise_variance
from wavelink.estimation import TapDomainLS


def evaluate(cfg):
    if any(cp < max(s['delays']) for cp in cfg['cp_lengths'] for s in cfg['channels']):
        raise ValueError('length-mismatch benchmark requires sufficient CP to isolate estimator bias')
    rows, summary = run(cfg)
    geometry = []
    for spacing in cfg['pilot_spacings']:
        layout = comb_grid(spacing)
        for length in cfg['tap_lengths']:
            m = TapDomainLS(layout,length)
            geometry.append(dict(pilot_spacing=spacing,assumed_length=length,n_pilots=len(layout.pilot_tones),
                                 rank=m.rank,condition=m.condition,total_data_noise_gain=m.noise_gain))
    profiles = {spec['name']: channel_taps(spec) for spec in cfg['channels']}
    for r in summary:
        r.update(bias_nmse='',variance_nmse='',predicted_nmse='',nmse_consistent='')
        if not r['receiver'].startswith('tap_ls_L'):
            continue
        layout = comb_grid(r['pilot_spacing'])
        model = TapDomainLS(layout,int(r['receiver'].split('L')[-1]))
        noise = pilot_noise_variance(r['ebn0_db'],layout,r['cp_len'])
        bias, variance = model.expected_nmse(np.fft.fft(profiles[r['channel']],64),noise)
        predicted = bias+variance
        # Fixed engineering tolerance: six empirical frame SEs plus floating point allowance.
        consistent = abs(r['data_nmse']-predicted) <= 6*r['frame_nmse_se']+1e-10
        r.update(bias_nmse=bias,variance_nmse=variance,predicted_nmse=predicted,nmse_consistent=consistent)
    return rows, summary, geometry


def plot(summary,cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    receivers = [('oracle_zf','Oracle CSI'),('ls_linear_zf','Linear LS')]+[(f'tap_ls_L{l}',f'Tap LS: L={l}') for l in cfg['tap_lengths']]
    for column,spec in enumerate(cfg['channels']):
        for index,(receiver,label) in enumerate(receivers):
            group=[r for r in summary if r['channel']==spec['name'] and r['pilot_spacing']==4 and r['receiver']==receiver]
            x=[r['ebn0_db'] for r in group]
            axes[0,column].semilogy(x,[max(r['ber'],.5/r['bits']) for r in group],marker='o',color=f'C{index}',label=label)
            if receiver!='oracle_zf':
                axes[1,column].semilogy(x,[r['data_nmse'] for r in group],marker='o',color=f'C{index}',label=label)
        axes[0,column].set(title=f"{spec['name']}: true length {max(spec['delays'])+1}",ylabel='Data BER')
        axes[1,column].set(ylabel='Data-channel NMSE')
    for ax in axes.ravel():
        ax.set_xlabel('Total transmitted Eb/N0 (dB)')
        ax.grid(True,which='both',alpha=.25)
        ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Channel-length mismatch: CP=16, pilot spacing=4\nZero-error points displayed at 0.5 / bits; not a confidence bound',fontsize=13)
    fig.savefig(out/'tapls.png',dpi=160)
    fig.savefig(out/'tapls.svg')
    svg=out/'tapls.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('experiments/tapls.json'))
    parser.add_argument('--output',type=Path,default=Path('results/tapls'))
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    rows, summary, geometry=evaluate(cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'runs.csv',rows)
    write_csv(args.output/'summary.csv',summary)
    write_csv(args.output/'geometry.csv',geometry)
    sources=sorted(Path('src').rglob('*.py'))+[Path(__file__),Path('experiments/multipath_ber.py'),args.config]
    meta=dict(config=cfg,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        source_sha256={p.resolve().relative_to(Path.cwd()).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        stopping_rule='fixed frames and scored symbols; no early stopping',
        prior='contiguous causal integer delays 0..L-1; L swept explicitly, not inferred from received data',
        uncertainty='independent-frame standard errors conditional on fixed channel; no paired significance claims',
        energy='total expected transmitted energy per data bit including pilots and CP')
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.plot:
        plot(summary,cfg,args.output)
    checks=[r for r in summary if r['nmse_consistent']!='']
    oracle=[r for r in summary if r['oracle_consistent']!='']
    print(f'{len(rows)} receiver-run rows; {sum(r["nmse_consistent"] for r in checks)}/{len(checks)} bias/variance checks; {sum(r["oracle_consistent"] for r in oracle)}/{len(oracle)} oracle BER checks')
    if not all(r['nmse_consistent'] for r in checks) or not all(r['oracle_consistent'] for r in oracle):
        raise SystemExit('Reference consistency failed; inspect without retuning seeds')


if __name__=='__main__':
    main()
