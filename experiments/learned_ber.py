"""Disjoint pilot-statistics calibration/test: python -m experiments.learned_ber --plot."""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
from wavelink import qpsk_modulate, qpsk_demodulate, ofdm_modulate, ofdm_demodulate
from wavelink.channel import add_awgn
from wavelink.estimation import TapDomainLS, TapDomainLMMSE, ensemble_nmse
from wavelink.statistics import learn_pilot_statistics
from wavelink.receiver import comb_grid, apply_multipath, ls_channel_estimate, zero_force, pilot_noise_variance
from experiments.multipath_ber import write_csv
from experiments.lmmse_ber import profile, linear_mapping, se


def simulate_frame(rng, powers, noise, layout, cp, symbols):
    """Evaluator owns truth; receiver calibration receives only divided pilots."""
    taps=(rng.normal(size=len(powers))+1j*rng.normal(size=len(powers)))*np.sqrt(powers/2)
    nd,npilot=len(layout.data_tones),len(layout.pilot_tones)
    ns=symbols+1
    bits=rng.integers(0,2,(ns,nd*2),dtype=np.uint8)
    pilots=qpsk_modulate(rng.integers(0,2,ns*npilot*2)).reshape(ns,npilot)
    data=qpsk_modulate(bits.ravel()).reshape(ns,nd)
    tx=ofdm_modulate(layout.pack(data,pilots),cp)
    y,_=apply_multipath(tx,taps)
    received=ofdm_demodulate(add_awgn(y,noise,rng),layout.n_fft,cp)[1:]
    truth=np.fft.fft(taps,layout.n_fft)[layout.bins(layout.data_tones)]
    return received,pilots[1:],bits[1:],truth


def rng_for(seed, ci, spacing, si, role):
    # Separate SeedSequence branches, fixed before any frames are generated.
    return np.random.default_rng(np.random.SeedSequence([seed,ci,spacing,si,role]))


def calibrate(cfg, layout, powers, noise, seed, ci, spacing, si):
    rng=rng_for(seed,ci,spacing,si,1)
    observations=[]
    for _ in range(max(cfg['calibration_frames'])):
        y,p,_,_=simulate_frame(rng,powers,noise,layout,cfg['cp_len'],cfg['symbols_per_frame'])
        observations.append(y[:,layout.bins(layout.pilot_tones)]/p)
    return {count:learn_pilot_statistics(np.asarray(observations[:count]),layout,cfg['assumed_length'])
            for count in cfg['calibration_frames']}


def evaluate(cfg):
    budgets=cfg['calibration_frames']
    if not budgets or len(set(budgets))!=len(budgets) or any(isinstance(k,bool) or not isinstance(k,int) or k<2 for k in budgets):
        raise ValueError('unique calibration budgets >=2 required')
    if cfg['frames_per_seed']<2 or cfg['symbols_per_frame']<2 or not cfg['seeds'] or len(set(cfg['seeds']))!=len(cfg['seeds']):
        raise ValueError('independent training/test frames and >=2 symbols required')
    rows,calibration,pdp=[],[],[]
    for ci,spec in enumerate(cfg['channels']):
        powers=profile(spec)
        if cfg['cp_len']<len(powers)-1:
            raise ValueError('sufficient CP required')
        for spacing in cfg['pilot_spacings']:
            layout=comb_grid(spacing)
            bins=layout.bins(layout.data_tones)
            nd=len(bins)
            for si,db in enumerate(cfg['ebn0_db']):
                noise=pilot_noise_variance(db,layout,cfg['cp_len'])
                for seed in cfg['seeds']:
                    fits=calibrate(cfg,layout,powers,noise,seed,ci,spacing,si)
                    estimators={'tap_ls':TapDomainLS(layout,cfg['assumed_length']),
                                'lmmse_matched':TapDomainLMMSE(layout,powers,noise),
                                'lmmse_uniform':TapDomainLMMSE(layout,np.ones(cfg['assumed_length'])/cfg['assumed_length'],noise)}
                    for count,fit in fits.items():
                        estimators[f'learned_{count}']=TapDomainLMMSE(layout,fit.powers,fit.noise_variance)
                        truth_p=np.pad(powers,(0,max(0,len(fit.powers)-len(powers))))
                        fitted_p=np.pad(fit.powers,(0,max(0,len(powers)-len(fit.powers))))
                        calibration.append(dict(channel=spec['name'],pilot_spacing=spacing,ebn0_db=db,seed=seed,
                            calibration_frames=count,symbols_per_frame=fit.symbols,noise_estimate=fit.noise_variance,
                            true_noise=noise,noise_ratio=fit.noise_variance/noise,total_power_estimate=float(fit.powers.sum()),
                            pdp_l1_error=float(np.sum(abs(fitted_p-truth_p))),clipped_taps=fit.clipped_taps))
                        for l,p in enumerate(fit.powers):
                            pdp.append(dict(channel=spec['name'],pilot_spacing=spacing,ebn0_db=db,seed=seed,
                                calibration_frames=count,delay=l,raw_power=float(fit.raw_powers[l]),estimated_power=float(p),
                                true_power=float(powers[l]) if l<len(powers) else 0.))
                    largest=fits[max(budgets)]
                    # Partial-oracle ablations: label the remaining true prior explicitly.
                    estimators['learned_noise_true_pdp']=TapDomainLMMSE(layout,powers,largest.noise_variance)
                    estimators['learned_pdp_true_noise']=TapDomainLMMSE(layout,largest.powers,noise)
                    mappings={'ls_linear':linear_mapping(layout),**{k:m.mapping for k,m in estimators.items()}}
                    predicted={k:sum(ensemble_nmse(a,layout,powers,noise)) for k,a in mappings.items()}
                    metrics={k:[] for k in ('oracle_zf',*mappings)}
                    rng=rng_for(seed,ci,spacing,si,2)
                    for _ in range(cfg['frames_per_seed']):
                        y,p,bits,truth=simulate_frame(rng,powers,noise,layout,cfg['cp_len'],cfg['symbols_per_frame'])
                        fitted={'oracle_zf':np.broadcast_to(truth,(len(y),nd)),
                                'ls_linear':ls_channel_estimate(y,p,layout),
                                **{k:m.estimate(y,p) for k,m in estimators.items()}}
                        theory_ber=float(np.mean([.5*math.erfc(abs(h)/math.sqrt(2*noise)) for h in truth]))
                        for name,h in fitted.items():
                            errors=int(np.count_nonzero(qpsk_demodulate(zero_force(y[:,bins],h).ravel())!=bits.ravel()))
                            metrics[name].append((errors/bits.size,float(np.mean(abs(h-truth)**2)),errors,theory_ber))
                    reference=np.asarray(metrics['lmmse_matched'])[:,1]
                    for name,values in metrics.items():
                        v=np.asarray(values)
                        pred=predicted.get(name,0.)
                        difference=v[:,1]-reference
                        oracle_diff=v[:,0]-v[:,3]
                        rows.append(dict(channel=spec['name'],pilot_spacing=spacing,cp_len=cfg['cp_len'],ebn0_db=db,
                            receiver=name,seed=seed,frames=len(v),bits=len(v)*cfg['symbols_per_frame']*nd*2,
                            errors=int(v[:,2].sum()),ber=float(v[:,0].mean()),frame_ber_se=se(v[:,0]),
                            ensemble_nmse=float(v[:,1].mean()),frame_nmse_se=se(v[:,1]),predicted_nmse=pred,
                            nmse_consistent=abs(v[:,1].mean()-pred)<=6*se(v[:,1])+1e-10,
                            paired_nmse_minus_matched=float(difference.mean()),paired_nmse_se=se(difference),
                            oracle_consistent=abs(oracle_diff.mean())<=6*se(oracle_diff)+1e-10 if name=='oracle_zf' else ''))
    summary=[]
    keys=('channel','pilot_spacing','cp_len','ebn0_db','receiver')
    groups={}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    for key,group in groups.items():
        n=len(group)
        mean=lambda field:float(np.mean([r[field] for r in group]))
        conditional_se=lambda field:float(np.sqrt(sum(r[field]**2 for r in group))/n)
        metric=mean('ensemble_nmse');pred=mean('predicted_nmse')
        summary.append(dict(zip(keys,key),replicates=n,frames=sum(r['frames'] for r in group),bits=sum(r['bits'] for r in group),
            errors=sum(r['errors'] for r in group),ber=mean('ber'),ensemble_nmse=metric,predicted_nmse=pred,
            conditional_test_nmse_se=conditional_se('frame_nmse_se'),
            replicate_nmse_se=se([r['ensemble_nmse'] for r in group]) if n>1 else '',
            paired_nmse_minus_matched=mean('paired_nmse_minus_matched'),conditional_paired_nmse_se=conditional_se('paired_nmse_se'),
            replicate_paired_nmse_se=se([r['paired_nmse_minus_matched'] for r in group]) if n>1 else '',
            nmse_consistent=abs(metric-pred)<=6*conditional_se('frame_nmse_se')+1e-10))
    return rows,summary,calibration,pdp


def plot(summary,cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    curves=[('oracle_zf','Oracle CSI'),('ls_linear','Linear LS'),('tap_ls','Tap LS'),('lmmse_matched','LMMSE: true statistics'),
            ('lmmse_uniform','Uniform PDP, true noise')]+[(f'learned_{k}',f'Learned: {k} frames') for k in cfg['calibration_frames']]
    fig,axes=plt.subplots(2,len(cfg['channels']),figsize=(13,9),constrained_layout=True,squeeze=False)
    for col,spec in enumerate(cfg['channels']):
        for i,(name,label) in enumerate(curves):
            g=[r for r in summary if r['channel']==spec['name'] and r['pilot_spacing']==4 and r['receiver']==name]
            x=[r['ebn0_db'] for r in g]
            axes[0,col].semilogy(x,[max(r['ber'],.5/r['bits']) for r in g],marker='o',color=f'C{i}',label=label)
            if name!='oracle_zf':
                axes[1,col].semilogy(x,[r['ensemble_nmse'] for r in g],marker='o',color=f'C{i}',label=label)
        axes[0,col].set(title=spec['name']+' block fading',ylabel='Data BER')
        axes[1,col].set_ylabel('Ensemble NMSE (ratio of expectations)')
    for ax in axes.ravel():
        ax.set_xlabel('Total transmitted Eb/N0 (dB)');ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Pilot-only calibration on separate frames: CP=16, pilot spacing=4\n3 independent calibration/test repetitions; no test-frame fitting',fontsize=13)
    fig.savefig(out/'learned.png',dpi=160);fig.savefig(out/'learned.svg')
    svg=out/'learned.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('experiments/learned.json'))
    parser.add_argument('--output',type=Path,default=Path('results/learned'))
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    rows,summary,calibration,pdp=evaluate(cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,values in [('runs',rows),('summary',summary),('calibration',calibration),('pdp',pdp)]:write_csv(args.output/(name+'.csv'),values)
    sources=sorted(Path('src').rglob('*.py'))+[Path(__file__),Path('experiments/lmmse_ber.py'),Path('experiments/multipath_ber.py'),args.config]
    meta=dict(config=cfg,command=['python','-m','experiments.learned_ber',*sys.argv[1:]],output_directory=args.output.as_posix(),
        metric='Ensemble E||Hhat-H||^2/E||H||^2, expected true channel power one; no framewise division',python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        source_sha256={p.resolve().relative_to(Path.cwd()).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        split='SeedSequence [seed,channel_index,spacing,snr_index,role]: role 1 calibration; role 2 test; nested calibration prefixes',
        priors='Known contiguous delay bound L=13; block-static channel; zero-mean independent taps; unit-modulus known pilots',
        calibration='Noise from disjoint within-frame pilot differences; PDP from mean-pilot tap LS with estimated noise subtraction and 1e-12 floor; no true-power normalization',
        uncertainty='Frame SE conditional on each fitted model; independent calibration/test replicate SE separately reported; only 3 calibration repetitions',
        energy='Total expected transmitted Eb includes pilots/CP; calibration budget separate from scored test bits; no throughput claim',
        stopping_rule='Fixed train/test budgets, seeds and numerical floors; no test-based model selection')
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.plot:plot(summary,cfg,args.output)
    checks=[r for r in summary if r['receiver']!='oracle_zf'];oracle=[r for r in rows if r['receiver']=='oracle_zf']
    print(f'{len(rows)} run rows; {sum(r["nmse_consistent"] for r in checks)}/{len(checks)} conditional NMSE checks; {sum(r["oracle_consistent"] for r in oracle)}/{len(oracle)} oracle BER checks')
    if not all(r['nmse_consistent'] for r in checks) or not all(r['oracle_consistent'] for r in oracle):raise SystemExit('Consistency failure: inspect without seed retuning')


if __name__=='__main__':main()
