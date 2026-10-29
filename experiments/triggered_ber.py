"""Development-locked observation trigger: python -m experiments.triggered_ber --plot."""
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
from experiments.robustness_ber import episode_config, phases, transition_weights, summarize_runs, validate as validate_robustness
from experiments.lmmse_ber import se
from experiments.multipath_ber import write_csv
from wavelink.receiver import comb_grid


def validate(cfg):
    dev,test=set(cfg['development']['seeds']),set(cfg['test']['seeds'])
    if dev&test:raise ValueError('development and test seeds must be disjoint')
    if len(dev)!=len(cfg['development']['seeds']) or len(dev)<2 or len(dev)>cfg['phase_cycle']:raise ValueError('unique development seeds within phase cycle required')
    validate_robustness(dict(cfg,**cfg['test']))
    candidates=cfg['trigger']['threshold_candidates']
    if not candidates or len(set(candidates))!=len(candidates) or any(not np.isfinite(x) or x<=0 for x in candidates):raise ValueError('positive unique thresholds required')
    if cfg['selection']['phase']!='post_all' or cfg['selection']['max_nmse_ratio_per_cell']<1 or cfg['selection']['max_stationary_mean_updates']<0:raise ValueError('invalid predeclared development constraints')


def trigger_policy(cfg,threshold,name):
    keys=('window','monitor_window','check_every','cooldown')
    return dict(name=name,kind='triggered',update_every=None,threshold=threshold,**{k:cfg['trigger'][k] for k in keys})


def policies_for(cfg,threshold=None):
    if threshold is None:
        base=[p for p in cfg['policies'] if p['name'] in ('frozen','periodic_64_32')]
        return base+[trigger_policy(cfg,x,f'trigger_{x:g}') for x in cfg['trigger']['threshold_candidates']]
    return cfg['policies']+[trigger_policy(cfg,threshold,'triggered')]


def evaluate_dataset(cfg,dataset,policies):
    cycle=cfg['phase_cycle']
    permutation=np.random.default_rng(dataset['schedule_seed']).permutation(cycle)
    starts={seed:cfg['onset_min']+int(p) for seed,p in zip(dataset['seeds'],permutation)}
    core=dict(episode_config(cfg),seeds=dataset['seeds'],policies=policies)
    runs,costs,schedules,events,detection=[],[],[],[],[]
    traces={};trigger_names=[p['name'] for p in policies if p.get('kind')=='triggered']
    bits_per_frame=cfg['symbols_per_frame']*len(comb_grid(cfg['pilot_spacing']).data_tones)*2
    for spec in cfg['scenarios']:
        for si,db in enumerate(cfg['ebn0_db']):
            for seed in dataset['seeds']:
                start=starts[seed];weights=transition_weights(cfg['episode_frames'],start,spec['ramp_frames'])
                values,counts,episode_events=episode(core,spec,db,seed,si,weights)
                context=dict(scenario=spec['name'],ebn0_db=db,seed=seed,onset_frame=start,onset_phase=start%cycle)
                schedules.append(dict(context,ramp_frames=spec['ramp_frames']))
                for row in counts:
                    row=dict(context,**row)
                    for key in ('monitor_checks','feature_frames','feature_ms','monitor_ms'):row.setdefault(key,0)
                    costs.append(row)
                for name in trigger_names:
                    g=[e for e in episode_events if e['receiver']==name]
                    events.extend(dict(context,**e) for e in g)
                    changing=spec['change_pdp'] or spec['noise_factor']!=1
                    after=[e['after_frame']-start+1 for e in g if e['after_frame']>=start] if changing else []
                    delay=min(after) if after else ''
                    detection.append(dict(context,receiver=name,total_triggers=len(g),prechange_false_triggers=sum(e['after_frame']<start for e in g),
                        stationary_false_updates=len(g) if not changing else '',first_update_delay=delay,
                        missed_within_64=int(delay=='' or delay>64) if changing else '',no_postchange_update=int(not after) if changing else ''))
                reference=values['frozen'];periodic=values['periodic_64_32']
                for name,v in values.items():
                    for phase,(a,b) in phases(cfg,start).items():
                        g=v[a:b];ref=reference[a:b];per=periodic[a:b]
                        runs.append(dict(context,receiver=name,phase=phase,frames=len(g),bits=len(g)*bits_per_frame,errors=int(g[:,3].sum()),
                            ber=float(g[:,0].mean()),ensemble_nmse=float(g[:,1].mean()),predicted_nmse=float(g[:,2].mean()),
                            theory_residual=float(np.mean(g[:,1]-g[:,2])),oracle_ber_residual=float(np.mean(g[:,0]-g[:,4])) if name=='oracle_zf' else '',
                            paired_ber_minus_frozen=float(np.mean(g[:,0]-ref[:,0])),paired_nmse_minus_frozen=float(np.mean(g[:,1]-ref[:,1])),
                            paired_ber_minus_periodic=float(np.mean(g[:,0]-per[:,0])),paired_nmse_minus_periodic=float(np.mean(g[:,1]-per[:,1]))))
                    width=cfg['trace_bin_frames']
                    for offset in range(-cfg['pre_frames'],cfg['late_start_offset']+cfg['late_frames'],width):
                        traces.setdefault((spec['name'],db,name,offset),[]).append(v[start+offset:start+offset+width,:3].mean(axis=0))
    trace=[]
    for (scenario,db,name,offset),group in traces.items():
        a=np.asarray(group)
        trace.append(dict(scenario=scenario,ebn0_db=db,receiver=name,relative_start=offset,relative_stop=offset+cfg['trace_bin_frames']-1,
            ber=float(a[:,0].mean()),episode_ber_se=se(a[:,0]),ensemble_nmse=float(a[:,1].mean()),episode_nmse_se=se(a[:,1]),predicted_nmse=float(a[:,2].mean())))
    return dict(runs=runs,summary=summarize_runs(runs),trace=trace,costs=costs,schedules=schedules,detection=detection,events=events)


def select_threshold(cfg,development_summary,development_costs):
    """Only development results enter this function; test results cannot enter."""
    phase=cfg['selection']['phase']
    reference={(r['scenario'],r['ebn0_db']):r['ensemble_nmse'] for r in development_summary if r['phase']==phase and r['receiver']=='periodic_64_32'}
    candidates=[]
    for threshold in cfg['trigger']['threshold_candidates']:
        name=f'trigger_{threshold:g}'
        ratios=[r['ensemble_nmse']/reference[(r['scenario'],r['ebn0_db'])] for r in development_summary if r['phase']==phase and r['receiver']==name]
        counts=[r['updates'] for r in development_costs if r['receiver']==name]
        stable=[r['updates'] for r in development_costs if r['receiver']==name and r['scenario']=='stationary']
        if len(ratios)!=len(reference) or not counts or not stable:raise ValueError('complete development cells and stationary control required')
        worst=max(ratios);stable_mean=float(np.mean(stable));mean_updates=float(np.mean(counts))
        feasible=worst<=cfg['selection']['max_nmse_ratio_per_cell'] and stable_mean<=cfg['selection']['max_stationary_mean_updates']
        candidates.append(dict(threshold=threshold,worst_nmse_ratio=worst,mean_updates=mean_updates,stationary_mean_updates=stable_mean,feasible=feasible))
    feasible=[r for r in candidates if r['feasible']]
    if feasible:
        chosen=min(feasible,key=lambda r:(r['mean_updates'],r['worst_nmse_ratio'],r['threshold']))
        reason='fewest mean updates among candidates satisfying every development NMSE cell and stationary-update constraints'
    else:
        chosen=min(candidates,key=lambda r:(r['worst_nmse_ratio'],r['mean_updates'],r['threshold']))
        reason='no feasible candidate; accuracy-first fallback, constraints explicitly not achieved'
    return dict(threshold=chosen['threshold'],development_constraints_met=bool(feasible),reason=reason,candidates=candidates)


def test_gates(cfg,summary,costs):
    reference={(r['scenario'],r['ebn0_db']):r['ensemble_nmse'] for r in summary if r['phase']=='post_all' and r['receiver']=='periodic_64_32'}
    gates=[]
    for row in summary:
        if row['phase']=='post_all' and row['receiver']=='triggered':
            ratio=row['ensemble_nmse']/reference[(row['scenario'],row['ebn0_db'])]
            gates.append(dict(scenario=row['scenario'],ebn0_db=row['ebn0_db'],nmse_ratio=ratio,
                              observed_mean_within_tolerance=ratio<=cfg['selection']['max_nmse_ratio_per_cell']))
    return gates


def plot(trace,cfg,out,db):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    curves=[('matched','True statistics'),('frozen','Frozen'),('periodic_64_32','Periodic 32'),('rolling_64','Rolling 64'),('triggered','Observation trigger')]
    for ax,spec in zip(axes.ravel(),cfg['scenarios']):
        for i,(name,label) in enumerate(curves):
            g=[r for r in trace if r['scenario']==spec['name'] and r['ebn0_db']==db and r['receiver']==name]
            x=np.array([(r['relative_start']+r['relative_stop'])/2 for r in g]);y=np.array([r['ensemble_nmse'] for r in g]);err=np.array([r['episode_nmse_se'] for r in g])
            ax.semilogy(x,y,color=f'C{i}',label=label);ax.fill_between(x,np.maximum(y-err,1e-12),y+err,color=f'C{i}',alpha=.10)
        ax.axvline(-.5,color='black',linestyle=':');ax.set(title=spec['name'],xlabel='Frames relative to hidden onset',ylabel='Ensemble NMSE');ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=7,ncol=2)
    fig.suptitle(f'Development-locked observation trigger: initial Eb/N0={db:g} dB\nIndependent test seeds, all 32 phases; mean ±1 episode SE',fontsize=13)
    for ext in ('png','svg'):fig.savefig(out/f'triggered_{db:g}.{ext}',**({'dpi':160} if ext=='png' else {}))
    p=out/f'triggered_{db:g}.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)


def manifest(config):
    sources=sorted(Path('src').rglob('*.py'))+[Path(__file__),Path('experiments/drift_ber.py'),Path('experiments/robustness_ber.py'),Path('experiments/learned_ber.py'),Path('experiments/lmmse_ber.py'),Path('experiments/multipath_ber.py'),config]
    return {p.resolve().relative_to(Path.cwd()).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('experiments/triggered.json'));parser.add_argument('--output',type=Path,default=Path('results/triggered'));parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());validate(cfg);args.output.mkdir(parents=True,exist_ok=True)
    sources=manifest(args.config);started=datetime.now(timezone.utc).isoformat()
    development=evaluate_dataset(cfg,cfg['development'],policies_for(cfg))
    for name in ('runs','summary','costs','detection'):write_csv(args.output/f'development_{name}.csv',development[name])
    selection=select_threshold(cfg,development['summary'],development['costs'])
    selection.update(locked_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,
                     config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                     development_summary_sha256=hashlib.sha256((args.output/'development_summary.csv').read_bytes()).hexdigest(),
                     development_costs_sha256=hashlib.sha256((args.output/'development_costs.csv').read_bytes()).hexdigest())
    lock=args.output/'selection.json';lock.write_text(json.dumps(selection,indent=2)+'\n');digest=hashlib.sha256(lock.read_bytes()).hexdigest()
    print(f'Development lock saved: threshold={selection["threshold"]}, constraints_met={selection["development_constraints_met"]}; starting independent test',flush=True)
    test_started=datetime.now(timezone.utc).isoformat()
    test=evaluate_dataset(cfg,cfg['test'],policies_for(cfg,selection['threshold']))
    if hashlib.sha256(lock.read_bytes()).hexdigest()!=digest or manifest(args.config)!=sources:raise SystemExit('Selection or sources changed during evaluation')
    for name in ('runs','summary','trace','costs','schedules','detection'):
        write_csv(args.output/(name+'.csv'),test[name])
    if test['events']:write_csv(args.output/'trigger_events.csv',test['events'])
    gates=test_gates(cfg,test['summary'],test['costs']);write_csv(args.output/'test_gates.csv',gates)
    meta=dict(config=cfg,command=['python','-m','experiments.triggered_ber',*sys.argv[1:]],output_directory=args.output.as_posix(),
        started_at_utc=started,test_started_at_utc=test_started,selection_sha256=digest,source_sha256=sources,
        python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        split='Disjoint development/test channel/noise seeds and separate schedule seeds; runtime policies receive pilots only',
        protocol='Every development post_all NMSE mean <=1.05x periodic; stationary mean updates <=1; minimize updates if feasible, otherwise explicit accuracy-first fallback',
        uncertainty='Episode SE conditional on fixed schedule assignment; mean-ratio gates are descriptive criteria, not statistical non-inferiority tests',
        cost='Trigger uses per-frame cached LS moments and checks every 8 frames; feature/monitor work counted separately from matrix rebuilds',
        causality='Score current frame before updating statistics; no change time supplied to detector; no final unused update',
        stopping_rule='Fixed candidate grid, datasets, scenarios and selection rule; no test-based retuning')
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.plot:
        for db in cfg['ebn0_db']:plot(test['trace'],cfg,args.output,db)
    for label,result in [('development',development),('test',test)]:
        checks=[r for r in result['summary'] if r['receiver']!='oracle_zf'];oracle=[r for r in result['summary'] if r['receiver']=='oracle_zf']
        print(f'{label}: {sum(r["nmse_consistent"] for r in checks)}/{len(checks)} NMSE checks; {sum(r["oracle_consistent"] for r in oracle)}/{len(oracle)} oracle BER checks')
        if not all(r['nmse_consistent'] for r in checks) or not all(r['oracle_consistent'] for r in oracle):raise SystemExit('Theoretical consistency failure; do not retune test seeds')
    print(f'{sum(r["observed_mean_within_tolerance"] for r in gates)}/{len(gates)} descriptive test NMSE gates; threshold remains frozen')


if __name__=='__main__':main()
