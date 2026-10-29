"""Frozen-policy distribution-shift audit: python -m experiments.generalization_ber --plot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
from experiments.drift_ber import episode
from experiments.triggered_ber import policies_for
from experiments.robustness_ber import summarize_runs
from experiments.lmmse_ber import se
from experiments.multipath_ber import write_csv
from wavelink.estimation import TapDomainLS
from wavelink.receiver import comb_grid


def frozen_protocol(cfg):
    path=Path(cfg['source_selection'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=cfg['source_selection_sha256']:raise ValueError('frozen selection digest mismatch')
    lock=json.loads(path.read_text())
    protocol=Path(cfg['source_protocol'])
    if hashlib.sha256(protocol.read_bytes()).hexdigest()!=lock['config_sha256']:raise ValueError('source protocol changed')
    for name,digest in lock['source_sha256'].items():
        if name.startswith('src/') and hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('frozen receiver implementation changed: '+name)
    parent=json.loads(protocol.read_text())
    return parent,lock


def validate(cfg):
    parent,lock=frozen_protocol(cfg)
    previous=set(parent['development']['seeds'])|set(parent['test']['seeds'])
    if len(cfg['seeds'])!=cfg['phase_cycle'] or len(set(cfg['seeds']))!=len(cfg['seeds']) or previous&set(cfg['seeds']):raise ValueError('new unique seeds covering all phases required')
    for key in ('sample_rate_hz','cp_len','pilot_spacing','calibration_frames','symbols_per_frame','assumed_length','phase_cycle'):
        if cfg[key]!=parent[key]:raise ValueError('frozen receiver/resource convention changed: '+key)
    for key in ('episode_frames','onset_min','trace_bin_frames','pre_frames','early_frames','late_start_offset','late_frames','post_frames'):
        if isinstance(cfg[key],bool) or not isinstance(cfg[key],int) or cfg[key]<1:raise ValueError('positive integer schedule values required')
    if cfg['onset_min']%cfg['phase_cycle'] or cfg['onset_min']<cfg['pre_frames']:raise ValueError('invalid onset origin')
    end=cfg['onset_min']+cfg['phase_cycle']-1+cfg['post_frames']
    if end>cfg['episode_frames'] or cfg['late_start_offset']+cfg['late_frames']!=cfg['post_frames']:raise ValueError('evaluation windows exceed episode')
    if any(cfg[k]%cfg['trace_bin_frames'] for k in ('episode_frames','pre_frames','early_frames','late_start_offset','late_frames')):raise ValueError('trace bins must align')
    length=len(cfg['states']['initial'])
    if length>cfg['cp_len']+1 or length>64:raise ValueError('true support must fit CP')
    initial=np.asarray(cfg['states']['initial'])
    expected=np.pad(parent['initial_powers'],(0,length-len(parent['initial_powers'])))
    if not np.array_equal(initial,expected):raise ValueError('initial calibration distribution changed')
    for p in cfg['states'].values():
        if np.iscomplexobj(p):raise ValueError('real PDP required')
        p=np.asarray(p,float)
        if p.shape!=(length,) or not np.all(np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1,rtol=0,atol=1e-12):raise ValueError('equal-length unit-power states required')
    if len({s['name'] for s in cfg['scenarios']})!=len(cfg['scenarios']):raise ValueError('unique scenario names required')
    for spec in cfg['scenarios']:
        changes=spec['changes'];offsets=[c['offset'] for c in changes]
        if not offsets or offsets[0]!=0 or offsets!=sorted(set(offsets)):raise ValueError('changes must start at zero and have unique sorted offsets')
        for c in changes:
            if isinstance(c['offset'],bool) or not isinstance(c['offset'],int) or not 0<=c['offset']<cfg['post_frames']:raise ValueError('invalid change offset')
            if c['state'] not in cfg['states'] or not np.isfinite(c['noise_factor']) or c['noise_factor']<=0:raise ValueError('invalid change state/noise')
    return parent,lock


def timeline(cfg,spec,start):
    n=cfg['episode_frames']
    p=np.tile(np.asarray(cfg['states']['initial'],float),(n,1));noise=np.ones(n)
    for change in spec['changes']:
        t=start+change['offset']
        p[t:]=cfg['states'][change['state']];noise[t:]=change['noise_factor']
    return p,noise


def update_response(times,start,stop):
    """A rebuild must take effect inside this state's interval, not the next."""
    valid=[t-start+1 for t in times if t>=start and t+1<stop]
    delay=min(valid) if valid else ''
    return dict(first_update_delay=delay,no_effective_update=int(not valid),missed_within_64=int(delay=='' or delay>64),
                observation_horizon=min(64,stop-start-1))


def subspace_floor(layout,length,powers):
    """Oracle data-space projection lower bound for length-constrained outputs."""
    assumed=TapDomainLS(layout,length).data_design
    true=np.exp(-2j*np.pi*np.outer(layout.data_tones,np.arange(len(powers)))/layout.n_fft)
    residual=true-assumed @ (np.linalg.pinv(assumed) @ true)
    return float(np.sum(abs(residual)**2*np.asarray(powers))/len(layout.data_tones))


def evaluate(cfg):
    parent,lock=validate(cfg);policies=policies_for(parent,lock['threshold'])
    phases=np.random.default_rng(cfg['schedule_seed']).permutation(cfg['phase_cycle'])
    starts={seed:cfg['onset_min']+int(phase) for seed,phase in zip(cfg['seeds'],phases)}
    core=dict(parent,seeds=cfg['seeds'],episode_frames=cfg['episode_frames'],switch_after_frames=cfg['onset_min'],transient_frames=cfg['early_frames'],policies=policies)
    layout=comb_grid(cfg['pilot_spacing']);bits_per_frame=cfg['symbols_per_frame']*len(layout.data_tones)*2
    runs,costs,schedules,detection,events=[],[],[],[],[];traces={}
    for spec in cfg['scenarios']:
        adapter=dict(name=spec['name'],change_pdp=False,noise_factor=1.)
        for si,db in enumerate(cfg['ebn0_db']):
            for seed in cfg['seeds']:
                start=starts[seed];powers,noise=timeline(cfg,spec,start)
                values,counts,all_events=episode(core,adapter,db,seed,si,statistics=(powers,noise))
                context=dict(scenario=spec['name'],ebn0_db=db,seed=seed,onset_frame=start,onset_phase=start%cfg['phase_cycle'])
                event_times=[e['after_frame'] for e in all_events if e['receiver']=='triggered']
                events.extend(dict(context,**e) for e in all_events if e['receiver']=='triggered')
                for row in counts:
                    row=dict(context,**row)
                    for k in ('monitor_checks','feature_frames','feature_ms','monitor_ms'):row.setdefault(k,0)
                    costs.append(row)
                for name,stride in [('periodic_64_32',32),('rolling_64',1),('triggered',None)]:
                    times=event_times if stride is None else [k-1 for k in range(stride,cfg['episode_frames'],stride)]
                    actual=next(r['updates'] for r in counts if r['receiver']==name)
                    if len(times)!=actual:raise AssertionError('rebuild schedule/count disagreement')
                    for j,change in enumerate(spec['changes']):
                        a=start+change['offset']
                        b=start+spec['changes'][j+1]['offset'] if j+1<len(spec['changes']) else cfg['episode_frames']
                        response=update_response(times,a,b)
                        detection.append(dict(context,receiver=name,change_index=j,change_frame=a,next_change_or_end=b,
                            target_state=change['state'],noise_factor=change['noise_factor'],**response))
                for j,change in enumerate(spec['changes']):
                    schedules.append(dict(context,change_index=j,change_frame=start+change['offset'],target_state=change['state'],noise_factor=change['noise_factor']))
                reference=values['frozen'];periodic=values['periodic_64_32']
                windows={'pre':(start-cfg['pre_frames'],start),'post_early':(start,start+cfg['early_frames']),
                         'post_late':(start+cfg['late_start_offset'],start+cfg['late_start_offset']+cfg['late_frames']),
                         'post_all':(start,start+cfg['post_frames'])}
                for name,v in values.items():
                    for phase,(a,b) in windows.items():
                        g=v[a:b];ref=reference[a:b];per=periodic[a:b]
                        runs.append(dict(context,receiver=name,phase=phase,frames=len(g),bits=len(g)*bits_per_frame,errors=int(g[:,3].sum()),
                            ber=float(g[:,0].mean()),ensemble_nmse=float(g[:,1].mean()),predicted_nmse=float(g[:,2].mean()),
                            theory_residual=float(np.mean(g[:,1]-g[:,2])),oracle_ber_residual=float(np.mean(g[:,0]-g[:,4])) if name=='oracle_zf' else '',
                            paired_ber_minus_frozen=float(np.mean(g[:,0]-ref[:,0])),paired_nmse_minus_frozen=float(np.mean(g[:,1]-ref[:,1])),
                            paired_ber_minus_periodic=float(np.mean(g[:,0]-per[:,0])),paired_nmse_minus_periodic=float(np.mean(g[:,1]-per[:,1]))))
                    width=cfg['trace_bin_frames']
                    for offset in range(-cfg['pre_frames'],cfg['post_frames'],width):
                        traces.setdefault((spec['name'],db,name,offset),[]).append(v[start+offset:start+offset+width,:3].mean(axis=0))
    summary=summarize_runs(runs)
    reference={(r['scenario'],r['ebn0_db']):r['ensemble_nmse'] for r in summary if r['phase']=='post_all' and r['receiver']=='periodic_64_32'}
    gates=[dict(scenario=r['scenario'],ebn0_db=r['ebn0_db'],nmse_ratio=r['ensemble_nmse']/reference[(r['scenario'],r['ebn0_db'])],
                observed_mean_within_reference=r['ensemble_nmse']<=1.05*reference[(r['scenario'],r['ebn0_db'])])
           for r in summary if r['phase']=='post_all' and r['receiver']=='triggered']
    trace=[]
    for (scenario,db,name,offset),group in traces.items():
        g=np.asarray(group)
        trace.append(dict(scenario=scenario,ebn0_db=db,receiver=name,relative_start=offset,relative_stop=offset+cfg['trace_bin_frames']-1,
            ber=float(g[:,0].mean()),episode_ber_se=se(g[:,0]),ensemble_nmse=float(g[:,1].mean()),episode_nmse_se=se(g[:,1]),predicted_nmse=float(g[:,2].mean())))
    capacity=[dict(state=k,assumed_length=cfg['assumed_length'],true_max_delay=int(np.flatnonzero(p)[-1]),
                   oracle_data_subspace_nmse_floor=subspace_floor(layout,cfg['assumed_length'],p)) for k,p in cfg['states'].items()]
    return dict(runs=runs,summary=summary,trace=trace,costs=costs,schedules=schedules,detection=detection,events=events,gates=gates,capacity=capacity)


def plot(trace,cfg,out,db):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    curves=[('matched','True statistics'),('frozen','Frozen'),('periodic_64_32','Periodic 32'),('rolling_64','Rolling 64'),('triggered','Frozen trigger rule')]
    for ax,spec in zip(axes.ravel(),cfg['scenarios']):
        for i,(name,label) in enumerate(curves):
            g=[r for r in trace if r['scenario']==spec['name'] and r['ebn0_db']==db and r['receiver']==name]
            x=np.array([(r['relative_start']+r['relative_stop'])/2 for r in g]);y=np.array([r['ensemble_nmse'] for r in g]);err=np.array([r['episode_nmse_se'] for r in g])
            ax.semilogy(x,y,color=f'C{i}',label=label);ax.fill_between(x,np.maximum(y-err,1e-12),y+err,color=f'C{i}',alpha=.10)
        for change in spec['changes']:ax.axvline(change['offset']-.5,color='black',linestyle=':',alpha=.35)
        ax.set(title=spec['name'],xlabel='Frames relative to first hidden change',ylabel='Ensemble NMSE');ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=7,ncol=2)
    fig.suptitle(f'Unseen shifts with unchanged trigger policy: initial Eb/N0={db:g} dB\nNew seeds, all 32 phases; mean ±1 episode SE; no retuning',fontsize=13)
    for ext in ('png','svg'):fig.savefig(out/f'generalization_{db:g}.{ext}',**({'dpi':160} if ext=='png' else {}))
    p=out/f'generalization_{db:g}.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('experiments/generalization.json'));parser.add_argument('--output',type=Path,default=Path('results/generalization'));parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());parent,lock=validate(cfg)
    sources=sorted(Path('src').rglob('*.py'))+[Path(__file__),Path('experiments/drift_ber.py'),Path('experiments/robustness_ber.py'),Path('experiments/triggered_ber.py'),Path('experiments/learned_ber.py'),Path('experiments/lmmse_ber.py'),Path('experiments/multipath_ber.py'),args.config]
    hashes={p.resolve().relative_to(Path.cwd()).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    results=evaluate(cfg);args.output.mkdir(parents=True,exist_ok=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise SystemExit('Sources changed during audit')
    frozen_protocol(cfg)
    for name,values in results.items():
        if values:write_csv(args.output/(name+'.csv'),values)
    meta=dict(config=cfg,command=['python','-m','experiments.generalization_ber',*sys.argv[1:]],output_directory=args.output.as_posix(),
        finished_at_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        source_sha256=hashes,frozen_selection_sha256=cfg['source_selection_sha256'],frozen_threshold=lock['threshold'],
        frozen_parameters={k:parent['trigger'][k] for k in ('window','monitor_window','check_every','cooldown')},
        split='New channel/noise seeds disjoint from prior development/test; policies and all receiver sources verified against original selection',
        channel='All scored frames use 16-length true PDP arrays, padded consistently across scenarios; CP=16 covers delay 15; assumed L remains 13',
        randomness='SeedSequence role streams reused across scenarios with equal true-array lengths; per-frame fades independent, statistics piecewise constant',
        uncertainty='Episode SE conditional on balanced phase assignment; repeated changes and overlapping windows are not independent replicates',
        detection='Each update must take effect before the next change to count for a segment; periodic times reconstructed from fixed schedule and checked against counts',
        reference='1.05 mean-NMSE ratio reused descriptively on post_all=352 frames; no claim of original 224-frame constraint certification',
        stopping_rule='Frozen threshold and implementation, predeclared unseen shifts; failures retained without retuning')
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.plot:
        for db in cfg['ebn0_db']:plot(results['trace'],cfg,args.output,db)
    checks=[r for r in results['summary'] if r['receiver']!='oracle_zf'];oracle=[r for r in results['summary'] if r['receiver']=='oracle_zf']
    print(f'{sum(r["nmse_consistent"] for r in checks)}/{len(checks)} NMSE checks; {sum(r["oracle_consistent"] for r in oracle)}/{len(oracle)} oracle BER checks; {sum(r["observed_mean_within_reference"] for r in results["gates"])}/{len(results["gates"])} descriptive reference cells')
    if not all(r['nmse_consistent'] for r in checks) or not all(r['oracle_consistent'] for r in oracle):raise SystemExit('Consistency failure; no retuning')


if __name__=='__main__':main()
