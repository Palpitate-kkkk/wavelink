"""Development-locked pilot-subspace guard: python -m experiments.guard_ber --plot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from experiments.drift_ber import episode
from experiments.generalization_ber import frozen_protocol, timeline, update_response
from experiments.triggered_ber import policies_for
from experiments.robustness_ber import summarize_runs
from experiments.multipath_ber import write_csv
from wavelink.receiver import comb_grid
from wavelink.guard import SupportGuard
from wavelink.tracking import make_tracker


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(cfg):
    prior=json.loads(Path('experiments/generalization.json').read_text())
    parent,lock=frozen_protocol(prior)
    used=set(prior['seeds']+parent['development']['seeds']+parent['test']['seeds'])
    dev,test=cfg['development']['seeds'],cfg['test']['seeds']
    if not 2<=len(dev)<=32 or len(test)!=32 or len(set(dev+test))!=len(dev+test) or used&set(dev+test):raise ValueError('new disjoint unique development/test seeds required')
    for key in ('calibration_frames','symbols_per_frame','sample_rate_hz'):
        if cfg[key]!=parent[key]:raise ValueError('fixed calibration/resource convention required')
    if cfg['episode_frames']<64+31+cfg['post_frames']:raise ValueError('episode too short')
    if cfg['cp_len']!=16 or cfg['pilot_spacing']!=4 or cfg['assumed_length']!=13:raise ValueError('fixed resource/support convention required')
    for powers in cfg['states'].values():
        p=np.asarray(powers,float)
        if p.shape!=(17,) or not np.all(np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1,rtol=0,atol=1e-12):raise ValueError('unit-power CP-covered states required')
    if not np.array_equal(cfg['states']['initial'],np.pad(parent['initial_powers'],(0,4))):raise ValueError('initial prior changed')
    if any(isinstance(x,bool) or not np.isfinite(x) or x<=0 for x in cfg['thresholds']):raise ValueError('positive thresholds required')
    if len(set(cfg['thresholds']))!=len(cfg['thresholds']):raise ValueError('unique thresholds required')
    for s in cfg['scenarios']:
        offsets=[c['offset'] for c in s['changes']]
        if offsets!=sorted(set(offsets)) or offsets[0]!=0:raise ValueError('ordered changes from zero required')
        for c in s['changes']:
            if c['state'] not in cfg['states'] or not 0<=c['offset']<cfg['post_frames'] or c['noise_factor']<=0:raise ValueError('invalid change')
    return parent,lock


def evaluate(cfg,split,thresholds):
    parent,lock=validate(cfg)
    policies=policies_for(parent,lock['threshold'])
    # rolling_64 is costly and not relevant to the support repair comparison.
    policies=[p for p in policies if p['name']!='rolling_64']
    policies += [dict(name=f'guard_{t:g}',kind='guard',threshold=t) for t in thresholds]
    core=dict(parent,seeds=cfg[split]['seeds'],episode_frames=cfg['episode_frames'],switch_after_frames=64,transient_frames=64,policies=policies)
    def factory(spec,cal,layout,length):
        if spec.get('kind')=='guard':
            return SupportGuard(cal,layout,length,cfg['cp_len'],threshold=spec['threshold'],**cfg['guard'])
        return make_tracker(spec,cal,layout,length)
    phases=np.random.default_rng(cfg[split]['schedule_seed']).permutation(32)
    layout=comb_grid(4);bpf=2*len(layout.data_tones)*cfg['symbols_per_frame']
    runs=[];events=[];costs=[];detection=[];schedules=[]
    for spec in cfg['scenarios']:
        for si,db in enumerate(cfg['ebn0_db']):
            for seed,phase in zip(cfg[split]['seeds'],phases):
                start=64+int(phase)
                powers,noise=timeline(cfg,spec,start)
                values,counts,logs=episode(core,dict(name=spec['name'],change_pdp=False,noise_factor=1.),db,seed,si,statistics=(powers,noise),tracker_factory=factory)
                context=dict(scenario=spec['name'],ebn0_db=db,seed=seed,onset_frame=start,onset_phase=int(phase))
                schedules.append(context)
                events.extend(dict(context,**dict(e,event_kind=e.get('event_kind','primary'),score=e.get('score',0.),noise_score=e.get('noise_score',0.),pdp_score=e.get('pdp_score',0.))) for e in logs)
                for c in counts:
                    for k in ('monitor_checks','feature_frames','feature_ms','monitor_ms'):c.setdefault(k,0)
                    costs.append(dict(context,**c))
                for t in thresholds:
                    name=f'guard_{t:g}'
                    alarms=[e['after_frame'] for e in logs if e['receiver']==name and e.get('event_kind')=='alarm']
                    true_changes=[start+c['offset'] for c in spec['changes'] if np.any(np.asarray(cfg['states'][c['state']])[cfg['assumed_length']:]>0)]
                    if true_changes:
                        stop=next((start+c['offset'] for c in spec['changes'] if c['offset']>true_changes[0]-start),cfg['episode_frames'])
                        response=update_response(alarms,true_changes[0],stop)
                    else:response=dict(first_update_delay='',no_effective_update='',missed_within_64='',observation_horizon='')
                    detection.append(dict(context,receiver=name,alarm_after_frame=alarms[0] if alarms else '',
                        alarm_before_outside=int(bool(alarms) and (not true_changes or alarms[0]<true_changes[0])),
                        supported_only=int(not true_changes),any_alarm=int(bool(alarms)),**response))
                windows={'pre':(start-32,start),'post_early':(start,start+64),'post_late':(start+cfg['post_frames']-64,start+cfg['post_frames']),'post_all':(start,start+cfg['post_frames'])}
                for name,v in values.items():
                    for phase,(a,b) in windows.items():
                        g=v[a:b];ref=values['frozen'][a:b];per=values['periodic_64_32'][a:b]
                        runs.append(dict(context,receiver=name,phase=phase,frames=len(g),bits=len(g)*bpf,errors=int(g[:,3].sum()),
                            ber=float(g[:,0].mean()),ensemble_nmse=float(g[:,1].mean()),predicted_nmse=float(g[:,2].mean()),
                            theory_residual=float(np.mean(g[:,1]-g[:,2])),oracle_ber_residual=float(np.mean(g[:,0]-g[:,4])) if name=='oracle_zf' else '',
                            paired_ber_minus_frozen=float(np.mean(g[:,0]-ref[:,0])),paired_nmse_minus_frozen=float(np.mean(g[:,1]-ref[:,1])),
                            paired_ber_minus_periodic=float(np.mean(g[:,0]-per[:,0])),paired_nmse_minus_periodic=float(np.mean(g[:,1]-per[:,1]))))
    return dict(runs=runs,summary=summarize_runs(runs),events=events,costs=costs,detection=detection,schedules=schedules)


def select(results,cfg):
    rows=[]
    for t in cfg['thresholds']:
        name=f'guard_{t:g}'
        supported={s['name'] for s in cfg['scenarios'] if all(not np.any(np.asarray(cfg['states'][c['state']])[13:]>0) for c in s['changes'])}
        ratios=[];outside=[]
        for r in results['summary']:
            if r['receiver']!=name or r['phase']!='post_all':continue
            ref=next(x for x in results['summary'] if x['scenario']==r['scenario'] and x['ebn0_db']==r['ebn0_db'] and x['phase']==r['phase'] and x['receiver']=='triggered')
            (ratios if r['scenario'] in supported else outside).append(r['ensemble_nmse']/ref['ensemble_nmse'])
        false=sum(r['any_alarm'] for r in results['detection'] if r['receiver']==name and r['supported_only'])
        passed=false==0 and max(ratios)<=1.05 and max(outside)<=.5
        rows.append(dict(threshold=t,supported_alarm_episodes=false,worst_supported_nmse_ratio=max(ratios),worst_outside_nmse_ratio=max(outside),eligible=passed))
    eligible=[r for r in rows if r['eligible']]
    # Fixed fallback: largest threshold if no candidate passes. Retain failure.
    chosen=min(r['threshold'] for r in eligible) if eligible else max(cfg['thresholds'])
    return chosen,rows,bool(eligible)


def plot(summary,cfg,out,threshold):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    names=[s['name'] for s in cfg['scenarios']];x=np.arange(len(names));db=max(cfg['ebn0_db'])
    for i,(method,label) in enumerate([('triggered','Original L=13'),(f'guard_{threshold:g}','Residual guard + CP prior'),('matched','True statistics')]):
        rows=[next(r for r in summary if r['scenario']==s and r['ebn0_db']==db and r['receiver']==method and r['phase']=='post_all') for s in names]
        for ax,key,err in zip(axes,['ensemble_nmse','ber'],['episode_nmse_se','episode_ber_se']):
            y=[r[key] for r in rows];e=[r[err] for r in rows]
            ax.errorbar(x+(i-1)*.15,y,yerr=e,fmt='o',capsize=3,label=label)
    for ax,ylabel in zip(axes,['Ensemble NMSE','BER']):
        ax.set(yscale='log',xticks=x,xticklabels=names,ylabel=ylabel);ax.tick_params(axis='x',rotation=30);ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=8)
    fig.suptitle(f'Independent test, initial Eb/N0={db:g} dB; post-change 224 frames; mean ±1 episode SE')
    for ext in ['svg','png']:fig.savefig(out/f'guard.{ext}',**({'dpi':160} if ext=='png' else {}))
    path=out/'guard.svg';path.write_text('\n'.join(l.rstrip() for l in path.read_text().splitlines())+'\n');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('experiments/guard.json'));parser.add_argument('--output',type=Path,default=Path('results/guard'));parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());validate(cfg)
    sources=sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))+[args.config,Path('experiments/generalization.json'),Path('experiments/triggered.json')]
    hashes={p.as_posix():digest(p) for p in sources}
    dev=evaluate(cfg,'development',cfg['thresholds']);chosen,rows,passed=select(dev,cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    for k,v in dev.items():
        if v:write_csv(args.output/f'development_{k}.csv',v)
    lock=dict(threshold=chosen,eligible_candidate_found=passed,candidates=rows,locked_before_test_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=digest(args.config),source_sha256=hashes,development_runs_sha256=digest(args.output/'development_runs.csv'),
        rule='Zero supported-only alarm episodes; all supported mean NMSE ratios <=1.05 and outside <=0.5 of original trigger; choose smallest eligible threshold, else largest candidate with failure retained')
    selection=args.output/'selection.json';selection.write_text(json.dumps(lock,indent=2)+'\n');selection_hash=digest(selection)
    test_start=datetime.now(timezone.utc).isoformat()
    test=evaluate(cfg,'test',[chosen])
    if any(digest(p)!=h for p,h in hashes.items()) or digest(selection)!=selection_hash:raise SystemExit('Sources or lock changed during experiment')
    for k,v in test.items():
        if v:write_csv(args.output/(k+'.csv'),v)
    meta=dict(config=cfg,command=['python','-m','experiments.guard_ber',*sys.argv[1:]],source_sha256=hashes,selection_sha256=selection_hash,test_started_at_utc=test_start,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        causality='Current mapping and score precede current-frame ingestion; alarm/fallback usable only from next frame; latched until episode end',
        priors='Frame-static, independent uncorrelated taps, ideal sync, integer delays; unit total ensemble power; fallback uniform over 0..CP, no true statistics input',
        statistics='Independent whole episodes and episode SE conditional on balanced phase assignment; repeated scenarios reuse latent streams; not independent frames or pooled scenarios',
        scope='Development reuses known failure structures with new seeds; test is independent-seed replication, not unseen-scenario generalization',
        stopping_rule='Fixed development rule, lock persisted before test; no test retuning; heuristic alarms, not calibrated hypothesis test')
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.plot:plot(test['summary'],cfg,args.output,chosen)
    checks=[r for r in test['summary'] if r['receiver']!='oracle_zf'];oracle=[r for r in test['summary'] if r['receiver']=='oracle_zf']
    print(json.dumps(dict(threshold=chosen,eligible=passed,nmse_checks=sum(r['nmse_consistent'] for r in checks),nmse_total=len(checks),oracle_checks=sum(r['oracle_consistent'] for r in oracle),oracle_total=len(oracle))))
    if not all(r['nmse_consistent'] for r in checks) or not all(r['oracle_consistent'] for r in oracle):raise SystemExit('Consistency failure; retain results without tuning')


if __name__=='__main__':main()
