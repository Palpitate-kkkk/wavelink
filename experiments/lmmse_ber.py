"""Block-fading PDP/noise-mismatch benchmark: python -m experiments.lmmse_ber --plot."""
import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
import numpy as np
from wavelink import qpsk_modulate, qpsk_demodulate, ofdm_modulate, ofdm_demodulate
from wavelink.channel import add_awgn
from wavelink.estimation import TapDomainLS, TapDomainLMMSE, ensemble_nmse
from wavelink.receiver import comb_grid, apply_multipath, ls_channel_estimate, zero_force, pilot_noise_variance
from experiments.multipath_ber import write_csv


def profile(spec):
    p = np.asarray(spec['powers'],float) if 'powers' in spec else np.exp(-np.arange(spec['length'])/spec['decay_samples'])
    if p.ndim != 1 or not len(p) or not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError('invalid channel PDP')
    return p/p.sum()


def models(layout, powers, noise, cfg):
    length = cfg['assumed_length']
    uniform = np.ones(length)/length
    wrong_short = np.zeros(length); wrong_short[:4] = 0.25
    result = {'tap_ls': TapDomainLS(layout,length)}
    for factor in cfg['noise_factors']:
        result[f'lmmse_matched_n{factor:g}'] = TapDomainLMMSE(layout,powers,noise*factor)
    result['lmmse_uniform'] = TapDomainLMMSE(layout,uniform,noise)
    result['lmmse_short'] = TapDomainLMMSE(layout,wrong_short,noise)
    return result


def linear_mapping(layout):
    """Extract the existing real/imag linear interpolator's complex-linear map."""
    n = len(layout.pilot_tones)
    y = np.zeros((n,layout.n_fft),complex)
    y[:,layout.bins(layout.pilot_tones)] = np.eye(n)
    return ls_channel_estimate(y,np.ones((n,n)),layout).T


def se(values):
    return float(np.std(values,ddof=1)/np.sqrt(len(values)))


def evaluate(cfg):
    if cfg['frames_per_seed'] < 2 or cfg['symbols_per_frame'] < 1 or len(set(cfg['seeds'])) != len(cfg['seeds']) or not cfg['seeds']:
        raise ValueError('independent frames and unique seeds required')
    if 1.0 not in cfg['noise_factors'] or len(set(cfg['noise_factors'])) != len(cfg['noise_factors']):
        raise ValueError('unique noise factors including 1.0 required')
    rows, summary, predictions = [], [], []
    for ci,spec in enumerate(cfg['channels']):
        powers = profile(spec)
        if cfg['cp_len'] < len(powers)-1:
            raise ValueError('sufficient CP required to isolate statistical mismatch')
        for spacing in cfg['pilot_spacings']:
            layout = comb_grid(spacing)
            nd,npilot = len(layout.data_tones),len(layout.pilot_tones)
            bins = layout.bins(layout.data_tones)
            for si,db in enumerate(cfg['ebn0_db']):
                noise = pilot_noise_variance(db,layout,cfg['cp_len'])
                fitted = models(layout,powers,noise,cfg)
                mappings = {'ls_linear': linear_mapping(layout), **{k:m.mapping for k,m in fitted.items()}}
                theory = {k:sum(ensemble_nmse(a,layout,powers,noise)) for k,a in mappings.items()}
                for name,a in mappings.items():
                    signal,variance = ensemble_nmse(a,layout,powers,noise)
                    predictions.append(dict(channel=spec['name'],pilot_spacing=spacing,ebn0_db=db,receiver=name,
                                            signal_nmse=signal,noise_nmse=variance,predicted_nmse=signal+variance))
                all_frames = {k:[] for k in ('oracle_zf',*mappings)}
                for seed in cfg['seeds']:
                    rng = np.random.default_rng(np.random.SeedSequence([seed,ci,spacing,si]))
                    frames = {k:[] for k in all_frames}
                    for _ in range(cfg['frames_per_seed']):
                        # No realization-wise normalization: covariance really is diag(powers).
                        taps = (rng.normal(size=len(powers))+1j*rng.normal(size=len(powers)))*np.sqrt(powers/2)
                        truth = np.fft.fft(taps,layout.n_fft)[bins]
                        ns = cfg['symbols_per_frame']+1
                        bits = rng.integers(0,2,(ns,nd*2),dtype=np.uint8)
                        data = qpsk_modulate(bits.ravel()).reshape(ns,nd)
                        pilots = qpsk_modulate(rng.integers(0,2,ns*npilot*2)).reshape(ns,npilot)
                        tx = ofdm_modulate(layout.pack(data,pilots),cfg['cp_len'])
                        propagated,_ = apply_multipath(tx,taps)
                        received = ofdm_demodulate(add_awgn(propagated,noise,rng),layout.n_fft,cfg['cp_len'])[1:]
                        estimates = {'oracle_zf':np.broadcast_to(truth,(ns-1,nd)),
                                     'ls_linear':ls_channel_estimate(received,pilots[1:],layout),
                                     **{k:m.estimate(received,pilots[1:]) for k,m in fitted.items()}}
                        oracle_ber = float(np.mean([.5*math.erfc(abs(h)/math.sqrt(2*noise)) for h in truth]))
                        for name,h in estimates.items():
                            decoded = qpsk_demodulate(zero_force(received[:,bins],h).ravel())
                            errors = int(np.count_nonzero(decoded != bits[1:].ravel()))
                            # Expected channel power is one on every data tone. Do not divide by this frame's fading power.
                            metric = float(np.mean(abs(h-truth)**2))
                            frames[name].append((errors/bits[1:].size,metric,errors,oracle_ber))
                    for name,values in frames.items():
                        v = np.asarray(values)
                        count = cfg['frames_per_seed']*cfg['symbols_per_frame']*nd*2
                        rows.append(dict(channel=spec['name'],pilot_spacing=spacing,cp_len=cfg['cp_len'],ebn0_db=db,
                            receiver=name,seed=seed,frames=len(v),bits=count,errors=int(v[:,2].sum()),
                            ber=float(v[:,0].mean()),ensemble_nmse=float(v[:,1].mean()),frame_ber_se=se(v[:,0]),frame_nmse_se=se(v[:,1])))
                        all_frames[name].extend(values)
                reference = np.asarray(all_frames['lmmse_matched_n1'])[:,1]
                for name,values in all_frames.items():
                    v = np.asarray(values)
                    predicted = 0.0 if name=='oracle_zf' else theory[name]
                    mse = float(v[:,1].mean())
                    consistent = abs(mse-predicted) <= 6*se(v[:,1])+1e-10
                    diff = v[:,1]-reference
                    oracle_diff = v[:,0]-v[:,3]
                    summary.append(dict(channel=spec['name'],pilot_spacing=spacing,cp_len=cfg['cp_len'],ebn0_db=db,
                        receiver=name,frames=len(v),bits=len(v)*cfg['symbols_per_frame']*nd*2,errors=int(v[:,2].sum()),
                        ber=float(v[:,0].mean()),frame_ber_se=se(v[:,0]),ensemble_nmse=mse,frame_nmse_se=se(v[:,1]),
                        predicted_nmse=predicted,nmse_consistent=consistent,
                        paired_nmse_minus_matched=float(diff.mean()),paired_nmse_se=se(diff),
                        oracle_theory_ber=float(v[:,3].mean()) if name=='oracle_zf' else '',
                        oracle_consistent=abs(oracle_diff.mean()) <= 6*se(oracle_diff)+1e-10 if name=='oracle_zf' else ''))
    return rows,summary,predictions


def plot(summary,cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    curves=[('oracle_zf','Oracle CSI'),('ls_linear','Linear LS'),('tap_ls','Tap LS, L=13'),
            ('lmmse_matched_n1','LMMSE: matched'),('lmmse_matched_n0.1','LMMSE: noise x0.1'),
            ('lmmse_matched_n10','LMMSE: noise x10'),('lmmse_uniform','LMMSE: uniform PDP'),('lmmse_short','LMMSE: short PDP')]
    fig,axes=plt.subplots(2,len(cfg['channels']),figsize=(13,9),constrained_layout=True,squeeze=False)
    for col,spec in enumerate(cfg['channels']):
        for i,(name,label) in enumerate(curves):
            group=[r for r in summary if r['channel']==spec['name'] and r['pilot_spacing']==4 and r['receiver']==name]
            x=[r['ebn0_db'] for r in group]
            axes[0,col].semilogy(x,[max(r['ber'],.5/r['bits']) for r in group],marker='o',color=f'C{i}',label=label)
            if name!='oracle_zf':
                axes[1,col].semilogy(x,[r['ensemble_nmse'] for r in group],marker='o',color=f'C{i}',label=label)
        axes[0,col].set_title(spec['name']+' block fading')
        axes[0,col].set_ylabel('Data BER')
        axes[1,col].set_ylabel('Ensemble NMSE (ratio of expectations)')
    for ax in axes.ravel():
        ax.set_xlabel('Total transmitted Eb/N0 (dB)')
        ax.grid(True,which='both',alpha=.25)
        ax.legend(fontsize=7,ncol=2)
    fig.suptitle('LMMSE statistical mismatch: CP=16, pilot spacing=4\nIndependent fading per frame; per-symbol estimation; no channel averaging',fontsize=13)
    fig.savefig(out/'lmmse.png',dpi=160)
    fig.savefig(out/'lmmse.svg')
    svg=out/'lmmse.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('experiments/lmmse.json'))
    parser.add_argument('--output',type=Path,default=Path('results/lmmse'))
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    rows,summary,predictions=evaluate(cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,values in [('runs',rows),('summary',summary),('predictions',predictions)]:
        write_csv(args.output/(name+'.csv'),values)
    sources=sorted(Path('src').rglob('*.py'))+[Path(__file__),Path('experiments/multipath_ber.py'),args.config]
    metadata=dict(config=cfg,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        source_sha256={p.resolve().relative_to(Path.cwd()).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        channel='Independent CN(0,diag(PDP)) per frame; fixed within frame; no per-realization normalization',
        metric='E||Hhat-H||^2 / E||H||^2; unit ensemble power; not mean of instantaneous NMSE ratios',
        priors='Matched PDP supplied as ideal statistics baseline, not estimated; uniform/short PDP and noise factors explicit',
        uncertainty='SE over independent fading frames; paired NMSE differences use shared frames; fixed six-SE regression tolerance',
        energy='Expected transmitted Eb includes pilots and CP; no receive-power or instantaneous-SNR normalization',
        stopping_rule='fixed frames, symbols and seeds; no early stopping or seed retuning')
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.plot: plot(summary,cfg,args.output)
    oracle=[r for r in summary if r['oracle_consistent']!='']
    estimated=[r for r in summary if r['receiver']!='oracle_zf']
    print(f'{len(rows)} run rows; {len(summary)} summary points; {sum(r["nmse_consistent"] for r in estimated)}/{len(estimated)} estimator NMSE checks; {sum(r["oracle_consistent"] for r in oracle)}/{len(oracle)} oracle BER checks')
    if not all(r['nmse_consistent'] for r in summary) or not all(r['oracle_consistent'] for r in oracle):
        raise SystemExit('Consistency failed; inspect model without retuning seeds')


if __name__=='__main__':
    main()
