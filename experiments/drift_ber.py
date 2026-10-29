"""Causal statistics-drift benchmark: python -m experiments.drift_ber --plot."""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
from wavelink import qpsk_demodulate
from wavelink.estimation import TapDomainLS, TapDomainLMMSE, ensemble_nmse, _tap_powers
from wavelink.tracking import make_tracker
from wavelink.receiver import comb_grid, ls_channel_estimate, zero_force, pilot_noise_variance
from experiments.learned_ber import simulate_frame
from experiments.lmmse_ber import linear_mapping, se
from experiments.multipath_ber import write_csv


def validate(cfg):
    for key in ('calibration_frames','episode_frames','switch_after_frames','transient_frames','symbols_per_frame','assumed_length','trace_bin_frames'):
        if isinstance(cfg[key],bool) or not isinstance(cfg[key],int) or cfg[key]<1:raise ValueError('positive integer configuration required: '+key)
    if cfg['symbols_per_frame']<2 or cfg['calibration_frames']<2:raise ValueError('at least two calibration frames/symbols required')
    if not 0<cfg['switch_after_frames']<cfg['switch_after_frames']+cfg['transient_frames']<cfg['episode_frames']:raise ValueError('nonempty pre/early/late phases required')
    if cfg['episode_frames']%cfg['trace_bin_frames']:raise ValueError('trace bins must divide episode length')
    if len(cfg['seeds'])<2 or len(set(cfg['seeds']))!=len(cfg['seeds']):raise ValueError('independent unique seeds required')
    initial=_tap_powers(cfg['initial_powers'],64);changed=_tap_powers(cfg['changed_powers'],64)
    if len(initial)!=len(changed) or max(len(initial),len(changed))-1>cfg['cp_len']:raise ValueError('equal channel lengths and sufficient CP required')
    if not np.isclose(initial.sum(),1) or not np.isclose(changed.sum(),1):raise ValueError('unit ensemble power required')
    names=[p['name'] for p in cfg['policies']]
    if len(set(names))!=len(names) or 'frozen' not in names:raise ValueError('unique policies including frozen required')
    if any(not np.isfinite(s['noise_factor']) or s['noise_factor']<=0 for s in cfg['scenarios']):raise ValueError('positive noise factors required')
    return initial,changed


def stream_rng(seed,snr_index,role):
    # Scenarios share latent draws to isolate changed statistics. Independent
    # seeds, rather than scenarios or adaptive frames, are statistical units.
    return np.random.default_rng(np.random.SeedSequence([seed,snr_index,role]))


def episode(cfg, scenario, db, seed, snr_index, trajectory=None, statistics=None):
    initial,changed=validate(cfg)
    if trajectory is not None and statistics is not None:raise ValueError('choose scalar trajectory or explicit statistics, not both')
    if statistics is not None:
        if any(np.iscomplexobj(x) for x in statistics):raise ValueError('statistics must be real powers')
        power_grid,noise_factors=(np.asarray(x,float) for x in statistics)
        if power_grid.ndim!=2 or power_grid.shape[0]!=cfg['episode_frames'] or not 1<=power_grid.shape[1]<=min(64,cfg['cp_len']+1):raise ValueError('explicit PDP shape or CP mismatch')
        if not np.all(np.isfinite(power_grid)) or np.any(power_grid<0) or not np.allclose(power_grid.sum(axis=1),1,rtol=0,atol=1e-12):raise ValueError('finite nonnegative unit-power PDP rows required')
        if noise_factors.shape!=(cfg['episode_frames'],) or not np.all(np.isfinite(noise_factors)) or np.any(noise_factors<=0):raise ValueError('positive per-frame noise factors required')
    if trajectory is not None:
        trajectory=np.asarray(trajectory,float)
        if trajectory.shape!=(cfg['episode_frames'],) or not np.all(np.isfinite(trajectory)) or np.any((trajectory<0)|(trajectory>1)):
            raise ValueError('trajectory must supply one finite weight in [0,1] per frame')
    layout=comb_grid(cfg['pilot_spacing']);bins=layout.bins(layout.data_tones)
    base_noise=pilot_noise_variance(db,layout,cfg['cp_len'])
    rng=stream_rng(seed,snr_index,1)
    calibration=[]
    for _ in range(cfg['calibration_frames']):
        y,p,_,_=simulate_frame(rng,initial,base_noise,layout,cfg['cp_len'],cfg['symbols_per_frame'])
        calibration.append(y[:,layout.bins(layout.pilot_tones)]/p)
    trackers={p['name']:make_tracker(p,np.asarray(calibration),layout,cfg['assumed_length']) for p in cfg['policies']}
    tap=TapDomainLS(layout,cfg['assumed_length']);linear=linear_mapping(layout)
    names=('oracle_zf','matched','ls_linear','tap_ls',*trackers)
    metrics={name:[] for name in names}
    matched_before=TapDomainLMMSE(layout,initial,base_noise)
    post_powers=changed if scenario['change_pdp'] else initial
    post_noise=base_noise*scenario['noise_factor']
    matched_after=TapDomainLMMSE(layout,post_powers,post_noise)
    rng=stream_rng(seed,snr_index,2)
    events=[]
    checkpoints={0,31,63,cfg['switch_after_frames']-1,cfg['switch_after_frames'],127,159,cfg['episode_frames']-2}
    maximum_lead=-1
    matched_cache={}
    for t in range(cfg['episode_frames']):
        post=t>=cfg['switch_after_frames']
        powers=post_powers if post else initial;noise=post_noise if post else base_noise
        matched=matched_after if post else matched_before
        if trajectory is not None:
            alpha=trajectory[t]
            if alpha==0:
                powers,noise,matched=initial,base_noise,matched_before
            elif alpha==1:
                powers,noise,matched=post_powers,post_noise,matched_after
            else:
                powers=(1-alpha)*initial+alpha*post_powers
                noise=base_noise*((1-alpha)+alpha*scenario['noise_factor'])
                matched=TapDomainLMMSE(layout,powers,noise)
        if statistics is not None:
            powers=power_grid[t];noise=base_noise*noise_factors[t]
            key=(powers.tobytes(),float(noise))
            if key not in matched_cache:matched_cache[key]=TapDomainLMMSE(layout,powers,noise)
            matched=matched_cache[key]
        y,p,bits,truth=simulate_frame(rng,powers,noise,layout,cfg['cp_len'],cfg['symbols_per_frame'])
        # Snapshot mappings BEFORE scoring or ingesting any current-frame pilots.
        mappings={'matched':matched.mapping,'ls_linear':linear,'tap_ls':tap.mapping,**{k:v.model.mapping for k,v in trackers.items()}}
        predicted={k:sum(ensemble_nmse(a,layout,powers,noise)) for k,a in mappings.items()}
        for tracker in trackers.values():
            if tracker.frames_observed!=t or tracker.last_update_after_frame>=t:raise AssertionError('noncausal state')
            maximum_lead=max(maximum_lead,tracker.last_update_after_frame-t)
        estimates={'oracle_zf':np.broadcast_to(truth,(len(y),len(bins))), 'matched':matched.estimate(y,p),
                   'ls_linear':ls_channel_estimate(y,p,layout),'tap_ls':tap.estimate(y,p),
                   **{k:v.estimate(y,p) for k,v in trackers.items()}}
        oracle=float(np.mean([.5*math.erfc(abs(h)/math.sqrt(2*noise)) for h in truth]))
        for name,h in estimates.items():
            errors=int(np.count_nonzero(qpsk_demodulate(zero_force(y[:,bins],h).ravel())!=bits.ravel()))
            metrics[name].append((errors/bits.size,float(np.mean(abs(h-truth)**2)),predicted.get(name,0.),errors,oracle))
        # Current frame has now been scored. It can only change future estimates.
        # Do not perform a final update that would never be used in this episode.
        if t<cfg['episode_frames']-1:
            for name,tracker in trackers.items():
                updated=tracker.advance()
                if updated and (t in checkpoints or hasattr(tracker,'trigger_log')):
                    fit=tracker.statistics
                    events.append(dict(receiver=name,after_frame=t,applies_from_frame=t+1,noise_ratio=fit.noise_variance/noise,
                        pdp_l1_error=float(np.sum(abs(np.pad(fit.powers,(0,max(0,len(powers)-len(fit.powers))))-np.pad(powers,(0,max(0,len(fit.powers)-len(powers))))))),updates=tracker.update_count))
                    if hasattr(tracker,'trigger_log'):events[-1].update(tracker.trigger_log[-1])
    costs=[dict(receiver=name,updates=v.update_count,fit_frame_visits=v.fit_frame_visits,
                initial_fit_ms=v.initial_fit_ns/1e6,total_update_ms=v.update_ns/1e6,
                max_history_frames=v.history.maxlen,history_complex_values=v.history.maxlen*v.symbols*len(layout.pilot_tones),
                last_update_after_frame=v.last_update_after_frame,maximum_training_lead=maximum_lead)
           for name,v in trackers.items()]
    for row in costs:
        tracker=trackers[row['receiver']]
        if hasattr(tracker,'trigger_log'):
            row.update(monitor_checks=tracker.monitor_checks,feature_frames=tracker.feature_frames,feature_ms=tracker.feature_ns/1e6,monitor_ms=tracker.monitor_ns/1e6)
    return {k:np.asarray(v) for k,v in metrics.items()},costs,events


def phases(cfg):
    switch=cfg['switch_after_frames'];late=switch+cfg['transient_frames'];end=cfg['episode_frames']
    return {'pre':(0,switch),'post_early':(switch,late),'post_late':(late,end),'post_all':(switch,end)}


def evaluate(cfg):
    validate(cfg)
    runs,costs,events=[],[],[]
    trace_groups={}
    for scenario in cfg['scenarios']:
        for si,db in enumerate(cfg['ebn0_db']):
            for seed in cfg['seeds']:
                values,episode_costs,episode_events=episode(cfg,scenario,db,seed,si)
                context=dict(scenario=scenario['name'],ebn0_db=db,seed=seed)
                costs.extend(dict(context,**c) for c in episode_costs)
                events.extend(dict(context,**e) for e in episode_events)
                reference=values['frozen']
                for name,v in values.items():
                    for phase,(start,stop) in phases(cfg).items():
                        g=v[start:stop];ref=reference[start:stop]
                        runs.append(dict(context,receiver=name,phase=phase,frames=len(g),bits=len(g)*cfg['symbols_per_frame']*len(comb_grid(cfg['pilot_spacing']).data_tones)*2,
                            errors=int(g[:,3].sum()),ber=float(g[:,0].mean()),ensemble_nmse=float(g[:,1].mean()),predicted_nmse=float(g[:,2].mean()),
                            theory_residual=float(np.mean(g[:,1]-g[:,2])),oracle_ber_residual=float(np.mean(g[:,0]-g[:,4])) if name=='oracle_zf' else '',
                            paired_ber_minus_frozen=float(np.mean(g[:,0]-ref[:,0])),paired_nmse_minus_frozen=float(np.mean(g[:,1]-ref[:,1]))))
                    width=cfg['trace_bin_frames']
                    for start in range(0,len(v),width):
                        trace_groups.setdefault((scenario['name'],db,name,start),[]).append(v[start:start+width,:3].mean(axis=0))
    grouped={}
    for r in runs:grouped.setdefault((r['scenario'],r['ebn0_db'],r['receiver'],r['phase']),[]).append(r)
    summary=[]
    for (scenario,db,name,phase),group in grouped.items():
        mean=lambda key:float(np.mean([r[key] for r in group]))
        stderr=lambda key:se([r[key] for r in group])
        consistent=abs(mean('theory_residual'))<=6*stderr('theory_residual')+1e-10
        summary.append(dict(scenario=scenario,ebn0_db=db,receiver=name,phase=phase,episodes=len(group),bits=sum(r['bits'] for r in group),errors=sum(r['errors'] for r in group),
            ber=mean('ber'),episode_ber_se=stderr('ber'),ensemble_nmse=mean('ensemble_nmse'),episode_nmse_se=stderr('ensemble_nmse'),
            predicted_nmse=mean('predicted_nmse'),theory_residual=mean('theory_residual'),episode_theory_residual_se=stderr('theory_residual'),nmse_consistent=consistent,
            paired_ber_minus_frozen=mean('paired_ber_minus_frozen'),paired_episode_ber_se=stderr('paired_ber_minus_frozen'),
            paired_nmse_minus_frozen=mean('paired_nmse_minus_frozen'),paired_episode_nmse_se=stderr('paired_nmse_minus_frozen'),
            oracle_consistent=abs(mean('oracle_ber_residual'))<=6*stderr('oracle_ber_residual')+1e-10 if name=='oracle_zf' else ''))
    trace=[]
    for (scenario,db,name,start),g in trace_groups.items():
        a=np.asarray(g)
        trace.append(dict(scenario=scenario,ebn0_db=db,receiver=name,start_frame=start,stop_frame=start+cfg['trace_bin_frames']-1,
            ber=float(a[:,0].mean()),episode_ber_se=se(a[:,0]),ensemble_nmse=float(a[:,1].mean()),episode_nmse_se=se(a[:,1]),predicted_nmse=float(a[:,2].mean())))
    return runs,summary,trace,costs,events


def plot(trace,cfg,out,db):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    curves=[('matched','True statistics'),('tap_ls','Tap LS'),('frozen','Frozen'),('periodic_64_32','Window 64, stride 32'),('rolling_16','Window 16, stride 1'),('rolling_64','Window 64, stride 1')]
    for ax,scenario in zip(axes.ravel(),cfg['scenarios']):
        for i,(name,label) in enumerate(curves):
            g=[r for r in trace if r['scenario']==scenario['name'] and r['ebn0_db']==db and r['receiver']==name]
            x=np.array([(r['start_frame']+r['stop_frame'])/2 for r in g]);y=np.array([r['ensemble_nmse'] for r in g]);err=np.array([r['episode_nmse_se'] for r in g])
            ax.semilogy(x,y,color=f'C{i}',label=label)
            ax.fill_between(x,np.maximum(y-err,1e-12),y+err,color=f'C{i}',alpha=.10)
        ax.axvline(cfg['switch_after_frames']-.5,color='black',linestyle=':',label='Scheduled boundary')
        ax.set(title=scenario['name'],xlabel='Scored frame index (0 based)',ylabel='Ensemble NMSE')
        ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=7,ncol=2)
    fig.suptitle(f'Causal pilot-statistics tracking: initial Eb/N0={db:g} dB\n8-frame bins; mean ±1 SE across independent episodes; update after scoring',fontsize=13)
    fig.savefig(out/f'drift_{db:g}.png',dpi=160);fig.savefig(out/f'drift_{db:g}.svg')
    svg=out/f'drift_{db:g}.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('experiments/drift.json'));parser.add_argument('--output',type=Path,default=Path('results/drift'));parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    runs,summary,trace,costs,events=evaluate(cfg);args.output.mkdir(parents=True,exist_ok=True)
    for name,values in [('runs',runs),('summary',summary),('trace',trace),('costs',costs),('update_events',events)]:write_csv(args.output/(name+'.csv'),values)
    sources=sorted(Path('src').rglob('*.py'))+[Path(__file__),Path('experiments/learned_ber.py'),Path('experiments/lmmse_ber.py'),Path('experiments/multipath_ber.py'),args.config]
    meta=dict(config=cfg,command=['python','-m','experiments.drift_ber',*sys.argv[1:]],output_directory=args.output.as_posix(),python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        source_sha256={p.resolve().relative_to(Path.cwd()).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        random_streams='SeedSequence [seed,snr_index,role], role 1 calibration, role 2 episode; scenarios reuse latent draws',
        causality='Estimate and score frame t using statistics updated at most through t-1; advance only after scoring; no final unused update',
        uncertainty='Independent whole episodes are units, including theory residual and paired comparisons; frames and overlapping windows not treated as independent',
        energy='Fixed transmitter Eb; post-change noise factor 4 lowers actual Eb/N0 by 6.0206 dB; initial Eb/N0 labels are not post-change SNR',
        priors='Known L and block-static uncorrelated zero-mean taps; policy never receives scenario, true statistics or change time',
        timing='perf_counter_ns diagnostic for each fit, includes statistics fitting and LMMSE construction; no formal latency benchmark',
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        stopping_rule='Fixed scenarios, change schedule, windows, strides, seeds and episode lengths; no test-driven selection')
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.plot:
        for db in cfg['ebn0_db']:plot(trace,cfg,args.output,db)
    checks=[r for r in summary if r['receiver']!='oracle_zf'];oracle=[r for r in summary if r['receiver']=='oracle_zf']
    print(f'{len(runs)} episode-phase rows; {sum(r["nmse_consistent"] for r in checks)}/{len(checks)} NMSE residual checks; {sum(r["oracle_consistent"] for r in oracle)}/{len(oracle)} oracle BER checks')
    if not all(r['nmse_consistent'] for r in checks) or not all(r['oracle_consistent'] for r in oracle):raise SystemExit('Consistency failure; inspect without seed retuning')


if __name__=='__main__':main()
