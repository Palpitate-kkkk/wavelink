"""Causal guard release: python -m experiments.recovery_ber --plot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from experiments.guard_ber import digest, validate as guard_validate
from experiments.drift_ber import episode
from experiments.generalization_ber import timeline, update_response
from experiments.triggered_ber import policies_for
from experiments.robustness_ber import summarize_runs
from experiments.multipath_ber import write_csv
from wavelink.guard import SupportGuard
from wavelink.recovery import RecoveringSupportGuard
from wavelink.tracking import make_tracker
from wavelink.receiver import comb_grid


def validate(cfg):
    parent,lock=guard_validate(dict(cfg,thresholds=[cfg['entry_threshold']]))
    if digest(cfg['baseline_metadata'])!=cfg['baseline_metadata_sha256']:raise ValueError('baseline metadata changed')
    baseline=json.loads(Path(cfg['baseline_metadata']).read_text())
    for p,h in baseline['source_sha256'].items():
        if digest(p)!=h:raise ValueError('baseline implementation/protocol changed: '+p)
    used=set(baseline['config']['development']['seeds']+baseline['config']['test']['seeds'])
    if used&set(cfg['development']['seeds']+cfg['test']['seeds']):raise ValueError('prior guard seeds reused')
    if cfg['entry_threshold']!=4. or not 0<cfg['exit_threshold']<4:raise ValueError('fixed entry and lower exit required')
    if not cfg['release_checks'] or len(set(cfg['release_checks']))!=len(cfg['release_checks']) or any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in cfg['release_checks']):raise ValueError('positive unique release check counts required')
    if len({s['name'] for s in cfg['scenarios']})!=len(cfg['scenarios']):raise ValueError('unique scenario names required')
    return parent,lock


def audit_modes(logs,outside):
    """Evaluation labels only: model transitions take effect on t+1."""
    active=np.zeros(len(outside),bool);mode=False;events=[]
    by_time={e['after_frame']:e for e in logs if e.get('event_kind') in ('alarm','release')}
    for t in range(len(outside)):
        active[t]=mode
        if t in by_time:
            e=by_time[t];mode=e['event_kind']=='alarm'
            events.append(dict(after_frame=t,event_kind=e['event_kind'],applies_from_frame=t+1,
                observed_frame_outside=int(outside[t]),next_frame_outside=int(outside[t+1]) if t+1<len(outside) else '',
                release_while_outside=int(e['event_kind']=='release' and outside[t])))
    return active,events


def evaluate(cfg,split,candidates):
    parent,lock=validate(cfg)
    policies=[p for p in policies_for(parent,lock['threshold']) if p['name']!='rolling_64']
    policies.append(dict(name='latched',kind='latched'))
    policies += [dict(name=f'recovery_{k}',kind='recovery',release_checks=k) for k in candidates]
    core=dict(parent,seeds=cfg[split]['seeds'],episode_frames=cfg['episode_frames'],switch_after_frames=64,transient_frames=64,policies=policies)
    def factory(spec,cal,layout,length):
        if spec.get('kind')=='latched':return SupportGuard(cal,layout,length,cfg['cp_len'],threshold=cfg['entry_threshold'],**cfg['guard'])
        if spec.get('kind')=='recovery':
            return RecoveringSupportGuard(cal,layout,length,cfg['cp_len'],threshold=cfg['entry_threshold'],exit_threshold=cfg['exit_threshold'],release_checks=spec['release_checks'],**cfg['guard'])
        return make_tracker(spec,cal,layout,length)
    phases=np.random.default_rng(cfg[split]['schedule_seed']).permutation(32)
    bpf=2*len(comb_grid(4).data_tones)*cfg['symbols_per_frame']
    runs=[];events=[];costs=[];mode_audit=[];transitions=[];responses=[];schedules=[]
    for spec in cfg['scenarios']:
        for si,db in enumerate(cfg['ebn0_db']):
            for seed,phase in zip(cfg[split]['seeds'],phases):
                start=64+int(phase);powers,noise=timeline(cfg,spec,start);outside=np.any(powers[:,cfg['assumed_length']:] > 0,axis=1)
                values,counts,logs=episode(core,dict(name=spec['name'],change_pdp=False,noise_factor=1.),db,seed,si,statistics=(powers,noise),tracker_factory=factory)
                context=dict(scenario=spec['name'],ebn0_db=db,seed=seed,onset_frame=start,onset_phase=int(phase))
                schedules.append(context)
                events.extend(dict(context,**dict(e,event_kind=e.get('event_kind','primary'),score=e.get('score',0.),noise_score=e.get('noise_score',0.),pdp_score=e.get('pdp_score',0.))) for e in logs)
                for c in counts:
                    for key in ('monitor_checks','feature_frames','feature_ms','monitor_ms'):c.setdefault(key,0)
                    costs.append(dict(context,**c))
                windows={'pre':(start-32,start),'post_early':(start,start+64),'post_late':(start+cfg['post_frames']-64,start+cfg['post_frames']),'post_all':(start,start+cfg['post_frames'])}
                for name,v in values.items():
                    for phase,(a,b) in windows.items():
                        g=v[a:b];ref=values['frozen'][a:b];per=values['periodic_64_32'][a:b];lat=values['latched'][a:b]
                        runs.append(dict(context,receiver=name,phase=phase,frames=len(g),bits=len(g)*bpf,errors=int(g[:,3].sum()),
                            ber=float(g[:,0].mean()),ensemble_nmse=float(g[:,1].mean()),predicted_nmse=float(g[:,2].mean()),
                            theory_residual=float(np.mean(g[:,1]-g[:,2])),oracle_ber_residual=float(np.mean(g[:,0]-g[:,4])) if name=='oracle_zf' else '',
                            paired_ber_minus_frozen=float(np.mean(g[:,0]-ref[:,0])),paired_nmse_minus_frozen=float(np.mean(g[:,1]-ref[:,1])),
                            paired_ber_minus_periodic=float(np.mean(g[:,0]-per[:,0])),paired_nmse_minus_periodic=float(np.mean(g[:,1]-per[:,1])),
                            paired_ber_minus_latched=float(np.mean(g[:,0]-lat[:,0])),paired_nmse_minus_latched=float(np.mean(g[:,1]-lat[:,1]))))
                for name in ['latched']+[f'recovery_{k}' for k in candidates]:
                    own=[e for e in logs if e['receiver']==name]
                    active,ts=audit_modes(own,outside)
                    transitions.extend(dict(context,receiver=name,**t) for t in ts)
                    a,b=windows['post_all'];support=~outside[a:b]
                    mode_audit.append(dict(context,receiver=name,alarms=sum(t['event_kind']=='alarm' for t in ts),releases=sum(t['event_kind']=='release' for t in ts),
                        release_while_outside=sum(t['release_while_outside'] for t in ts),supported_only=int(not np.any(outside)),
                        fallback_supported_frames=int(np.count_nonzero(active[a:b]&support)),supported_frames=int(np.count_nonzero(support)),
                        primary_outside_frames=int(np.count_nonzero(~active[a:b]&~support)),outside_frames=int(np.count_nonzero(~support))))
                    for j,c in enumerate(spec['changes']):
                        state_outside=bool(np.any(np.asarray(cfg['states'][c['state']])[cfg['assumed_length']:]>0))
                        # Only support changes are alarm/release response targets.
                        if j==0 and not state_outside:continue
                        if j and state_outside==bool(np.any(np.asarray(cfg['states'][spec['changes'][j-1]['state']])[cfg['assumed_length']:]>0)):continue
                        a=start+c['offset'];b=start+spec['changes'][j+1]['offset'] if j+1<len(spec['changes']) else cfg['episode_frames']
                        kind='alarm' if state_outside else 'release'
                        times=[t['after_frame'] for t in ts if t['event_kind']==kind]
                        response=update_response(times,a,b)
                        responses.append(dict(context,receiver=name,change_frame=a,next_change_or_end=b,target=kind,already_in_target_mode=int(active[a]==state_outside),**response))
    summary=summarize_runs(runs)
    # Preserve paired comparisons against the ACTUAL latched baseline as well.
    for row in summary:
        group=[r for r in runs if all(r[k]==row[k] for k in ('scenario','ebn0_db','receiver','phase'))]
        from experiments.lmmse_ber import se
        for key in ('ber','nmse'):
            vals=[r[f'paired_{key}_minus_latched'] for r in group]
            row[f'paired_{key}_minus_latched']=float(np.mean(vals));row[f'paired_latched_{key}_se']=se(vals)
    return dict(runs=runs,summary=summary,events=events,costs=costs,mode_audit=mode_audit,transitions=transitions,responses=responses,schedules=schedules)


def select(results,cfg):
    rows=[]
    for k in cfg['release_checks']:
        name=f'recovery_{k}';ratios=[];late=[]
        for r in results['summary']:
            if r['receiver']!=name:continue
            if r['phase'] not in ('post_all','post_late'):continue
            ref=next(x for x in results['summary'] if all(x[q]==r[q] for q in ('scenario','ebn0_db','phase')) and x['receiver']=='latched')
            ratio=r['ensemble_nmse']/ref['ensemble_nmse']
            if r['phase']=='post_all':ratios.append(ratio)
            elif r['scenario'] in ('return15','return_new'):late.append(ratio)
        audit=[r for r in results['mode_audit'] if r['receiver']==name]
        false=sum(r['alarms'] for r in audit if r['supported_only'])
        premature=sum(r['release_while_outside'] for r in audit if r['scenario']=='persistent15')
        passed=false==0 and premature==0 and max(ratios)<=1.05 and max(late)<=.5
        rows.append(dict(release_checks=k,supported_alarms=false,persistent_premature_releases=premature,worst_post_nmse_ratio=max(ratios),worst_return_late_nmse_ratio=max(late),eligible=passed))
    eligible=[r['release_checks'] for r in rows if r['eligible']]
    return (min(eligible) if eligible else max(cfg['release_checks'])),rows,bool(eligible)


def plot(summary,cfg,out,chosen):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    scenarios=[s['name'] for s in cfg['scenarios']]
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ri,db in enumerate(cfg['ebn0_db']):
        for ci,key in enumerate(('ensemble_nmse','ber')):
            ax=axes[ri,ci]
            for i,(method,label) in enumerate([('latched','Latched guard'),(f'recovery_{chosen}','Releasing guard'),('triggered','Original L=13'),('matched','True statistics')]):
                rows=[next(r for r in summary if r['scenario']==s and r['ebn0_db']==db and r['receiver']==method and r['phase']=='post_all') for s in scenarios]
                err='episode_nmse_se' if key=='ensemble_nmse' else 'episode_ber_se'
                ax.errorbar(np.arange(len(scenarios))+(i-1.5)*.12,[r[key] for r in rows],yerr=[r[err] for r in rows],fmt='o',capsize=2,label=label,markersize=4)
            ax.set(yscale='log',xticks=np.arange(len(scenarios)),xticklabels=scenarios,ylabel=key,title=f'Initial Eb/N0={db:g} dB')
            ax.tick_params(axis='x',rotation=30);ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Independent-seed replication: 224 post-change frames, mean ±1 episode SE')
    for ext in ('svg','png'):fig.savefig(out/f'recovery.{ext}',**({'dpi':160} if ext=='png' else {}))
    p=out/'recovery.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('experiments/recovery.json'));parser.add_argument('--output',type=Path,default=Path('results/recovery'));parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());validate(cfg)
    sources=sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))+[args.config,Path('experiments/generalization.json'),Path('experiments/triggered.json')]
    hashes={p.as_posix():digest(p) for p in sources}
    dev=evaluate(cfg,'development',cfg['release_checks']);chosen,rows,passed=select(dev,cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    for k,v in dev.items():
        if v:write_csv(args.output/f'development_{k}.csv',v)
    lock=dict(release_checks=chosen,eligible_candidate_found=passed,candidates=rows,locked_before_test_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
        config_sha256=digest(args.config),development_runs_sha256=digest(args.output/'development_runs.csv'),
        rule='Zero supported alarms and persistent15 premature releases; all post_all mean NMSE ratios <=1.05 of latched, both return scenarios post_late <=0.5; smallest eligible confirmation count, else largest with failure retained')
    path=args.output/'selection.json';path.write_text(json.dumps(lock,indent=2)+'\n');lock_hash=digest(path);test_start=datetime.now(timezone.utc).isoformat()
    test=evaluate(cfg,'test',[chosen])
    if any(digest(p)!=h for p,h in hashes.items()) or digest(path)!=lock_hash:raise SystemExit('Sources or selection changed')
    for k,v in test.items():
        if v:write_csv(args.output/(k+'.csv'),v)
    meta=dict(config=cfg,source_sha256=hashes,selection_sha256=lock_hash,command=['python','-m','experiments.recovery_ber',*sys.argv[1:]],test_started_at_utc=test_start,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        causality='Score current frame before advance; alarm and release apply from next frame; release refits latest 64 observed frames and resets primary local update clock',
        scope='Independent-seed replication on declared known/extended synthetic scenarios, not hardware or unseen-scenario certification',
        uncertainty='Whole independent episodes; phase assignment balanced; scenarios reuse latent streams; paired SE vs latched saved explicitly',
        priors='Integer delays, ideal sync, frame-static uncorrelated taps, sufficient CP, unit ensemble total-power fallback prior',
        costs='updates count effective next-frame replacements; overwritten same-frame rebuild time retained; release full-history refit visits and initialization work counted as online update cost',
        stopping_rule='Development selection persisted before independent test, no test-driven changes; all failed gates and scenarios retained')
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.plot:plot(test['summary'],cfg,args.output,chosen)
    checks=[r for r in test['summary'] if r['receiver']!='oracle_zf'];oracle=[r for r in test['summary'] if r['receiver']=='oracle_zf']
    print(json.dumps(dict(release_checks=chosen,eligible=passed,nmse_pass=sum(r['nmse_consistent'] for r in checks),nmse_total=len(checks),oracle_pass=sum(r['oracle_consistent'] for r in oracle),oracle_total=len(oracle))))
    if not all(r['nmse_consistent'] for r in checks) or not all(r['oracle_consistent'] for r in oracle):raise SystemExit('Consistency failure: no retuning')


if __name__=='__main__':main()
