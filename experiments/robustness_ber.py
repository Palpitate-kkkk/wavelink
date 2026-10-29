"""Balanced random onset and gradual statistics: python -m experiments.robustness_ber --plot."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
from experiments.drift_ber import episode, validate as validate_episode
from experiments.lmmse_ber import se
from experiments.multipath_ber import write_csv
from wavelink.receiver import comb_grid


def episode_config(cfg):
    # Fixed-boundary fields only satisfy legacy validation/diagnostic checkpoints.
    # Actual channel statistics are supplied by the explicit per-frame trajectory.
    return dict(cfg,switch_after_frames=cfg['onset_min'],transient_frames=cfg['early_frames'])


def validate(cfg):
    validate_episode(episode_config(cfg))
    keys=('schedule_seed','onset_min','phase_cycle','pre_frames','early_frames','late_start_offset','late_frames')
    if any(isinstance(cfg[k],bool) or not isinstance(cfg[k],int) or cfg[k]<0 for k in keys):raise ValueError('nonnegative integer schedule parameters required')
    if cfg['phase_cycle']<1 or len(cfg['seeds'])!=cfg['phase_cycle']:raise ValueError('one independent episode seed per balanced phase required')
    if not cfg['pre_frames'] or not cfg['early_frames'] or not cfg['late_frames']:raise ValueError('nonempty evaluation windows required')
    if cfg['onset_min']%cfg['phase_cycle'] or cfg['onset_min']<cfg['pre_frames']:raise ValueError('aligned onset origin and sufficient prefix required')
    if cfg['late_start_offset']<cfg['early_frames']:raise ValueError('late window must follow early window')
    if cfg['onset_min']+cfg['phase_cycle']-1+cfg['late_start_offset']+cfg['late_frames']>cfg['episode_frames']:raise ValueError('insufficient episode length for all onset phases')
    if any(cfg[k]%cfg['trace_bin_frames'] for k in ('pre_frames','early_frames','late_start_offset','late_frames')):raise ValueError('aligned trace windows required')
    names=[p['name'] for p in cfg['policies']]
    if 'periodic_64_32' not in names:raise ValueError('periodic reference required')
    policy=next(p for p in cfg['policies'] if p['name']=='periodic_64_32')
    if policy['update_every']!=cfg['phase_cycle']:raise ValueError('phase cycle must match periodic update stride')
    if next(p for p in cfg['policies'] if p['name']=='frozen')['update_every'] is not None:raise ValueError('frozen reference must remain frozen')
    if len({s['name'] for s in cfg['scenarios']})!=len(cfg['scenarios']):raise ValueError('unique scenario names required')
    for spec in cfg['scenarios']:
        r=spec['ramp_frames']
        if isinstance(r,bool) or not isinstance(r,int) or r<0 or r>cfg['early_frames']:raise ValueError('ramp must fit the predefined early window')
        settling=r+max(p['window'] for p in cfg['policies'])+max(p['update_every'] or 0 for p in cfg['policies'])
        if cfg['late_start_offset']<settling:raise ValueError('late window must follow ramp and window/update settling bound')
    return cfg


def onset_schedule(cfg):
    validate(cfg)
    phases=np.random.default_rng(cfg['schedule_seed']).permutation(cfg['phase_cycle'])
    return {seed:cfg['onset_min']+int(phase) for seed,phase in zip(cfg['seeds'],phases)}


def transition_weights(total_frames,start,ramp_frames):
    if isinstance(start,bool) or not isinstance(start,(int,np.integer)) or not 0<=start<total_frames:raise ValueError('valid integer onset required')
    if isinstance(ramp_frames,bool) or not isinstance(ramp_frames,(int,np.integer)) or ramp_frames<0 or start+ramp_frames>total_frames:raise ValueError('valid integer ramp required')
    frames=np.arange(total_frames)
    return (frames>=start).astype(float) if ramp_frames==0 else np.clip((frames-start+1)/ramp_frames,0.,1.)


def phases(cfg,start):
    early=start+cfg['early_frames'];late=start+cfg['late_start_offset'];end=late+cfg['late_frames']
    return {'pre':(start-cfg['pre_frames'],start),'post_early':(start,early),'post_late':(late,end),'post_all':(start,end)}


def summarize_runs(runs):
    groups={}
    for row in runs:groups.setdefault((row['scenario'],row['ebn0_db'],row['receiver'],row['phase']),[]).append(row)
    summary=[]
    for (scenario,db,name,phase),group in groups.items():
        mean=lambda key:float(np.mean([r[key] for r in group]))
        stderr=lambda key:se([r[key] for r in group])
        summary.append(dict(scenario=scenario,ebn0_db=db,receiver=name,phase=phase,episodes=len(group),frames_per_episode=group[0]['frames'],
            bits=sum(r['bits'] for r in group),errors=sum(r['errors'] for r in group),ber=mean('ber'),episode_ber_se=stderr('ber'),
            ensemble_nmse=mean('ensemble_nmse'),episode_nmse_se=stderr('ensemble_nmse'),predicted_nmse=mean('predicted_nmse'),
            theory_residual=mean('theory_residual'),episode_theory_residual_se=stderr('theory_residual'),
            nmse_consistent=abs(mean('theory_residual'))<=6*stderr('theory_residual')+1e-10,
            paired_ber_minus_frozen=mean('paired_ber_minus_frozen'),paired_episode_ber_se=stderr('paired_ber_minus_frozen'),
            paired_nmse_minus_frozen=mean('paired_nmse_minus_frozen'),paired_episode_nmse_se=stderr('paired_nmse_minus_frozen'),
            paired_ber_minus_periodic=mean('paired_ber_minus_periodic'),paired_periodic_ber_se=stderr('paired_ber_minus_periodic'),
            paired_nmse_minus_periodic=mean('paired_nmse_minus_periodic'),paired_periodic_nmse_se=stderr('paired_nmse_minus_periodic'),
            oracle_consistent=abs(mean('oracle_ber_residual'))<=6*stderr('oracle_ber_residual')+1e-10 if name=='oracle_zf' else ''))
    return summary


def evaluate(cfg):
    schedule=onset_schedule(cfg)
    runs,costs,schedules=[],[],[]
    traces={}
    bit_count=cfg['symbols_per_frame']*len(comb_grid(cfg['pilot_spacing']).data_tones)*2
    for spec in cfg['scenarios']:
        for si,db in enumerate(cfg['ebn0_db']):
            for seed in cfg['seeds']:
                start=schedule[seed];ramp=spec['ramp_frames']
                weights=transition_weights(cfg['episode_frames'],start,ramp)
                values,counts,_=episode(episode_config(cfg),spec,db,seed,si,trajectory=weights)
                context=dict(scenario=spec['name'],ebn0_db=db,seed=seed,onset_frame=start,onset_phase=start%cfg['phase_cycle'])
                schedules.append(dict(context,ramp_frames=ramp,full_change_frame=start+max(ramp-1,0),
                    early_start=start,early_stop=start+cfg['early_frames']-1,late_start=start+cfg['late_start_offset'],
                    late_stop=start+cfg['late_start_offset']+cfg['late_frames']-1))
                costs.extend(dict(context,**x) for x in counts)
                reference=values['frozen'];periodic=values['periodic_64_32']
                for name,v in values.items():
                    for phase,(a,b) in phases(cfg,start).items():
                        g=v[a:b];ref=reference[a:b];per=periodic[a:b]
                        runs.append(dict(context,receiver=name,phase=phase,start_frame=a,stop_frame=b-1,frames=len(g),bits=len(g)*bit_count,
                            errors=int(g[:,3].sum()),ber=float(g[:,0].mean()),ensemble_nmse=float(g[:,1].mean()),predicted_nmse=float(g[:,2].mean()),
                            theory_residual=float(np.mean(g[:,1]-g[:,2])),oracle_ber_residual=float(np.mean(g[:,0]-g[:,4])) if name=='oracle_zf' else '',
                            paired_ber_minus_frozen=float(np.mean(g[:,0]-ref[:,0])),paired_nmse_minus_frozen=float(np.mean(g[:,1]-ref[:,1])),
                            paired_ber_minus_periodic=float(np.mean(g[:,0]-per[:,0])),paired_nmse_minus_periodic=float(np.mean(g[:,1]-per[:,1]))))
                    width=cfg['trace_bin_frames']
                    for offset in range(-cfg['pre_frames'],cfg['late_start_offset']+cfg['late_frames'],width):
                        traces.setdefault((spec['name'],db,name,offset),[]).append(v[start+offset:start+offset+width,:3].mean(axis=0))
    trace=[]
    for (scenario,db,name,offset),group in traces.items():
        g=np.asarray(group)
        trace.append(dict(scenario=scenario,ebn0_db=db,receiver=name,relative_start=offset,relative_stop=offset+cfg['trace_bin_frames']-1,
            ber=float(g[:,0].mean()),episode_ber_se=se(g[:,0]),ensemble_nmse=float(g[:,1].mean()),episode_nmse_se=se(g[:,1]),predicted_nmse=float(g[:,2].mean())))
    return runs,summarize_runs(runs),trace,costs,schedules


def plot(trace,cfg,out,db):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    curves=[('matched','True statistics'),('tap_ls','Tap LS'),('frozen','Frozen'),('periodic_64_32','Window 64, stride 32'),('rolling_16','Window 16, stride 1'),('rolling_64','Window 64, stride 1')]
    for ax,spec in zip(axes.ravel(),cfg['scenarios']):
        for i,(name,label) in enumerate(curves):
            g=[r for r in trace if r['scenario']==spec['name'] and r['ebn0_db']==db and r['receiver']==name]
            x=np.array([(r['relative_start']+r['relative_stop'])/2 for r in g]);y=np.array([r['ensemble_nmse'] for r in g]);err=np.array([r['episode_nmse_se'] for r in g])
            ax.semilogy(x,y,color=f'C{i}',label=label);ax.fill_between(x,np.maximum(y-err,1e-12),y+err,color=f'C{i}',alpha=.10)
        if spec['name']!='stationary' and spec['ramp_frames']:
            ax.axvspan(-.5,spec['ramp_frames']-.5,color='grey',alpha=.10,label='Statistics ramp')
        ax.axvline(-.5,color='black',linestyle=':')
        ax.set(title=spec['name'],xlabel='Frames relative to hidden onset',ylabel='Ensemble NMSE');ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=7,ncol=2)
    fig.suptitle(f'Random onset and gradual statistics: initial Eb/N0={db:g} dB\nAll 32 update phases; aligned 8-frame bins; mean ±1 SE conditional on schedule',fontsize=13)
    for ext in ('png','svg'):fig.savefig(out/f'robustness_{db:g}.{ext}',**({'dpi':160} if ext=='png' else {}))
    p=out/f'robustness_{db:g}.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('experiments/robustness.json'));parser.add_argument('--output',type=Path,default=Path('results/robustness'));parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    runs,summary,trace,costs,schedules=evaluate(cfg);args.output.mkdir(parents=True,exist_ok=True)
    for name,values in [('runs',runs),('summary',summary),('trace',trace),('costs',costs),('schedules',schedules)]:write_csv(args.output/(name+'.csv'),values)
    sources=sorted(Path('src').rglob('*.py'))+[Path(__file__),Path('experiments/drift_ber.py'),Path('experiments/learned_ber.py'),Path('experiments/lmmse_ber.py'),Path('experiments/multipath_ber.py'),args.config]
    meta=dict(config=cfg,command=['python','-m','experiments.robustness_ber',*sys.argv[1:]],output_directory=args.output.as_posix(),python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        source_sha256={p.resolve().relative_to(Path.cwd()).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        schedule='Random permutation of every phase 0..31 assigned once to 32 independent channel/noise seeds; same assignment across scenarios and SNRs',
        trajectory='PDP and noise POWERS interpolate linearly by alpha=clip((t-start+1)/ramp,0,1); step uses t>=start; block static within each frame',
        uncertainty='Independent episode errors conditional on fixed stratified schedule; SE is conservative stratified mean estimate, not an unconditional random-onset confidence interval',
        random_streams='Channel/noise role streams inherited from drift episode; schedule seed independent; scenarios reuse latent draws',
        causality='Trajectory and onset enter evaluator only; current frame scored before statistics update; no final unused update',
        energy='Fixed transmitter Eb; true instantaneous ensemble Eb/N0 = initial dB - 10log10(1+3alpha) in changing cases',
        timing='Inherited per-fit wall-time diagnostics; not a formal latency benchmark',
        stopping_rule='Fixed stratified phases, ramps, horizons, seeds and policies; no result-driven selection')
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.plot:
        for db in cfg['ebn0_db']:plot(trace,cfg,args.output,db)
    checks=[r for r in summary if r['receiver']!='oracle_zf'];oracle=[r for r in summary if r['receiver']=='oracle_zf']
    print(f'{len(runs)} episode-phase rows; {sum(r["nmse_consistent"] for r in checks)}/{len(checks)} NMSE checks; {sum(r["oracle_consistent"] for r in oracle)}/{len(oracle)} oracle BER checks')
    if not all(r['nmse_consistent'] for r in checks) or not all(r['oracle_consistent'] for r in oracle):raise SystemExit('Consistency failed; inspect without seed or schedule retuning')


if __name__=='__main__':main()
