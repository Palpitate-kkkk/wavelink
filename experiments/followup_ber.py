"""Development-locked delayed refit: python -m experiments.followup_ber --plot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import math
import numpy as np
from wavelink import qpsk_demodulate
from wavelink.receiver import comb_grid, pilot_noise_variance, zero_force
from wavelink.estimation import TapDomainLMMSE, ensemble_nmse
from wavelink.tracking import TriggeredPilotLMMSE, CausalPilotLMMSE
from wavelink.recovery import RecoveringSupportGuard
from wavelink.innovation import InnovationTriggeredLMMSE
from wavelink.followup import FollowupPilotLMMSE
from experiments.density_ber import coupled_frame, digest
from experiments.generalization_ber import timeline
from experiments.lmmse_ber import se
from experiments.multipath_ber import write_csv


def validate(cfg):
    if digest(cfg['baseline_metadata'])!=cfg['baseline_metadata_sha256']:raise ValueError('baseline metadata changed')
    old=json.loads(Path(cfg['baseline_metadata']).read_text())
    for p,h in old['source_sha256'].items():
        if digest(p)!=h:raise ValueError('frozen baseline changed: '+p)
    used=set()
    for name in ('triggered','generalization','guard','recovery','density','innovation'):
        c=json.loads(Path(f'experiments/{name}.json').read_text());used.update(c.get('seeds',[]))
        for split in ('development','test'):used.update(c.get(split,{}).get('seeds',[]))
    dev,test=cfg['development']['seeds'],cfg['test']['seeds']
    if not 2<=len(dev)<=32 or len(test)!=32 or len(set(dev+test))!=len(dev+test) or used&set(dev+test):raise ValueError('new disjoint seeds required')
    parent=old['config']
    for k in ('sample_rate_hz','cp_len','calibration_frames','symbols_per_frame','episode_frames','post_frames','guard','entry_threshold','exit_threshold','release_checks'):
        if cfg[k]!=parent[k]:raise ValueError('fixed baseline convention changed: '+k)
    if cfg['pilot_spacing']!=2:raise ValueError('full-length identifiable dense grid required')
    if not np.array_equal(cfg['states']['initial'],parent['states']['initial']):raise ValueError('initial state changed')
    if cfg['innovation_threshold']!=2.:raise ValueError('frozen innovation threshold required')
    if not cfg['refit_delays'] or len(set(cfg['refit_delays']))!=len(cfg['refit_delays']) or any(isinstance(x,bool) or not isinstance(x,int) or x<32 or x%8 for x in cfg['refit_delays']):raise ValueError('unique aligned delays >=32 required')
    for p in cfg['states'].values():
        if np.iscomplexobj(p):raise ValueError('real powers required')
        p=np.asarray(p,float)
        if p.shape!=(17,) or not np.all(np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1,rtol=0,atol=1e-12):raise ValueError('unit-power CP-covered states required')
    if len({s['name'] for s in cfg['scenarios']})!=len(cfg['scenarios']):raise ValueError('unique scenarios required')
    for s in cfg['scenarios']:
        offsets=[x['offset'] for x in s['changes']]
        if not offsets or offsets[0]!=0 or offsets!=sorted(set(offsets)):raise ValueError('ordered changes from zero required')
        for c in s['changes']:
            if isinstance(c['offset'],bool) or not isinstance(c['offset'],int) or not 0<=c['offset']<cfg['post_frames'] or c['state'] not in cfg['states'] or not np.isfinite(c['noise_factor']) or c['noise_factor']<=0:raise ValueError('invalid change')


def episode(cfg,spec,db,si,seed,start,delays):
    layout=comb_grid(2);noise0=pilot_noise_variance(db,layout,cfg['cp_len']);bins=layout.bins(layout.data_tones)
    calibration=[]
    for t in range(cfg['calibration_frames']):
        y,p,_,_=coupled_frame(seed,si,1,t,cfg['states']['initial'],noise0,layout,cfg['cp_len'],cfg['symbols_per_frame'])
        calibration.append(y[:,layout.bins(layout.pilot_tones)]/p)
    cal=np.asarray(calibration)
    trackers=dict(legacy17=TriggeredPilotLMMSE(cal,layout,17,threshold=.75),periodic17=CausalPilotLMMSE(cal,layout,17,64,32),
        recovery13=RecoveringSupportGuard(cal,layout,13,cfg['cp_len'],threshold=cfg['entry_threshold'],exit_threshold=cfg['exit_threshold'],release_checks=cfg['release_checks'],**cfg['guard']))
    trackers['innovation']=InnovationTriggeredLMMSE(cal,layout,17,innovation_threshold=cfg['innovation_threshold'])
    trackers.update({f'followup_{v}':FollowupPilotLMMSE(cal,layout,17,innovation_threshold=cfg['innovation_threshold'],refit_delay=v) for v in delays})
    metrics={name:[] for name in ['oracle_zf','matched',*trackers]};events=[];cache={}
    powers,noises=timeline(cfg,spec,start)
    for t in range(cfg['episode_frames']):
        noise=noise0*noises[t];power=powers[t];key=(power.tobytes(),float(noise))
        if key not in cache:cache[key]=TapDomainLMMSE(layout,power,noise)
        matched=cache[key]
        y,p,bits,truth=coupled_frame(seed,si,2,t,power,noise,layout,cfg['cp_len'],cfg['symbols_per_frame'])
        maps={'matched':matched.mapping,**{k:v.model.mapping for k,v in trackers.items()}}
        predicted={k:sum(ensemble_nmse(m,layout,power,noise)) for k,m in maps.items()}
        estimates={'oracle_zf':np.broadcast_to(truth,(len(y),len(bins))),'matched':matched.estimate(y,p)}
        for name,tracker in trackers.items():
            if tracker.frames_observed!=t or tracker.last_update_after_frame>=t:raise AssertionError('noncausal model')
            estimates[name]=tracker.estimate(y,p)
        oracle=float(np.mean([.5*math.erfc(abs(h)/math.sqrt(2*noise)) for h in truth]))
        for name,h in estimates.items():
            error=int(np.count_nonzero(qpsk_demodulate(zero_force(y[:,bins],h).ravel())!=bits.ravel()))
            metrics[name].append((error/bits.size,float(np.mean(abs(h-truth)**2)),predicted.get(name,0.),error,oracle))
        if t<cfg['episode_frames']-1:
            for name,tracker in trackers.items():
                if tracker.advance():
                    log=tracker.trigger_log[-1] if hasattr(tracker,'trigger_log') else {}
                    events.append(dict(receiver=name,after_frame=t,applies_from_frame=t+1,event_kind=log.get('event_kind','statistics' if name=='legacy17' else 'periodic'),
                        score=log.get('score',0.),innovation_score=log.get('innovation_score',0.)))
    costs=[dict(receiver=name,updates=v.update_count,fit_frame_visits=v.fit_frame_visits,initial_fit_ms=v.initial_fit_ns/1e6,total_update_ms=v.update_ns/1e6,
        feature_frames=getattr(v,'feature_frames',0),feature_ms=getattr(v,'feature_ns',0)/1e6,monitor_checks=getattr(v,'monitor_checks',0),monitor_ms=getattr(v,'monitor_ns',0)/1e6,
        innovation_checks=getattr(v,'innovation_checks',0),innovation_cached_complex_values=len(v.pilot_means)*len(layout.pilot_tones) if hasattr(v,'pilot_means') else 0,
        followups=getattr(v,'followup_count',0),armed_plans=getattr(v,'armed_plans',0),cancelled_plans=getattr(v,'cancelled_plans',0),pending_plan=int(getattr(v,'pending_refit_after_frame',None) is not None),
        last_update_after_frame=v.last_update_after_frame,maximum_training_lead=-1) for name,v in trackers.items()]
    plans=[dict(receiver=name,**row) for name,v in trackers.items() for row in getattr(v,'schedule_log',[])]
    return {k:np.asarray(v) for k,v in metrics.items()},events,costs,plans


def summarize(runs):
    grouped={}
    for r in runs:grouped.setdefault(tuple(r[k] for k in ('scenario','ebn0_db','receiver','phase')),[]).append(r)
    summary=[]
    metrics=['ber','ensemble_nmse','predicted_nmse','theory_residual']+[f'paired_{m}_minus_{ref}' for ref in ('legacy17','recovery13','periodic17','innovation') for m in ('ber','nmse')]
    for keys,g in grouped.items():
        r=dict(zip(('scenario','ebn0_db','receiver','phase'),keys));r['episodes']=len(g)
        for k in metrics:r[k]=float(np.mean([x[k] for x in g]));r[k+'_se']=se([x[k] for x in g])
        r['bits']=sum(x['bits'] for x in g);r['errors']=sum(x['errors'] for x in g)
        r['nmse_consistent']=abs(r['theory_residual'])<=6*r['theory_residual_se']+1e-10
        ores=[x['oracle_residual'] for x in g]
        r['oracle_consistent']=bool(abs(np.mean(ores))<=6*se(ores)+1e-10) if keys[2]=='oracle_zf' else ''
        summary.append(r)
    return summary


def evaluate(cfg,split,delays):
    validate(cfg);phases=np.random.default_rng(cfg[split]['schedule_seed']).permutation(32)
    runs=[];events=[];costs=[];responses=[];plans=[]
    for spec in cfg['scenarios']:
        for si,db in enumerate(cfg['ebn0_db']):
            for seed,onset_phase in zip(cfg[split]['seeds'],phases):
                start=64+int(onset_phase);values,logs,counts,schedule=episode(cfg,spec,db,si,seed,start,delays)
                context=dict(scenario=spec['name'],ebn0_db=db,seed=seed,onset_frame=start,onset_phase=int(onset_phase))
                events.extend(dict(context,**e) for e in logs);costs.extend(dict(context,**c) for c in counts);plans.extend(dict(context,**p) for p in schedule)
                for j,change in enumerate(spec['changes']):
                    a=start+change['offset'];b=start+spec['changes'][j+1]['offset'] if j+1<len(spec['changes']) else cfg['episode_frames']
                    for name in ('innovation',*[f'followup_{v}' for v in delays]):
                        response_delays=[e['applies_from_frame']-a for e in logs if e['receiver']==name and a<=e['after_frame'] and e['applies_from_frame']<b]
                        responses.append(dict(context,receiver=name,change_frame=a,next_change_or_end=b,first_update_delay=min(response_delays) if response_delays else '',no_effective_update=int(not response_delays)))
                windows={'pre':(start-32,start),'post_early':(start,start+64),'post_late':(start+cfg['post_frames']-64,start+cfg['post_frames']),'post_all':(start,start+cfg['post_frames'])}
                for name,v in values.items():
                    for phase,(a,b) in windows.items():
                        g=v[a:b];row=dict(context,receiver=name,phase=phase,frames=len(g),bits=len(g)*cfg['symbols_per_frame']*48,errors=int(g[:,3].sum()),
                            ber=float(g[:,0].mean()),ensemble_nmse=float(g[:,1].mean()),predicted_nmse=float(g[:,2].mean()),theory_residual=float(np.mean(g[:,1]-g[:,2])),
                            oracle_residual=float(np.mean(g[:,0]-g[:,4])) if name=='oracle_zf' else 0.)
                        for ref in ('legacy17','recovery13','periodic17','innovation'):
                            row[f'paired_ber_minus_{ref}']=float(np.mean(g[:,0]-values[ref][a:b,0]));row[f'paired_nmse_minus_{ref}']=float(np.mean(g[:,1]-values[ref][a:b,1]))
                        runs.append(row)
    return dict(runs=runs,summary=summarize(runs),events=events,costs=costs,responses=responses,plans=plans)


def select(results,cfg):
    candidates=[];index={(r['scenario'],r['ebn0_db'],r['receiver'],r['phase']):r for r in results['summary']}
    for delay in cfg['refit_delays']:
        name=f'followup_{delay}';ratios=[];extra=[]
        for r in results['summary']:
            if r['receiver']==name and r['phase']=='post_all':
                ratios.append(r['ensemble_nmse']/index[r['scenario'],r['ebn0_db'],'innovation','post_all']['ensemble_nmse'])
                def mean_updates(method):return float(np.mean([x['updates'] for x in results['costs'] if x['receiver']==method and x['scenario']==r['scenario'] and x['ebn0_db']==r['ebn0_db']]))
                extra.append(mean_updates(name)-mean_updates('innovation'))
        weak=index['weak15',18,name,'post_late']['ensemble_nmse']/index['weak15',18,'innovation','post_late']['ensemble_nmse']
        static=[x['updates'] for x in results['costs'] if x['receiver']==name and x['scenario']=='stationary']
        control=[x['updates'] for x in results['costs'] if x['receiver']=='innovation' and x['scenario']=='stationary']
        stationary_extra=float(np.mean(static)-np.mean(control))
        mean_updates=float(np.mean([x['updates'] for x in results['costs'] if x['receiver']==name]))
        passed=max(ratios)<=1.05 and weak<=.75 and max(extra)<=1.5 and stationary_extra==0.
        candidates.append(dict(refit_delay=delay,worst_post_nmse_ratio=max(ratios),weak18_late_nmse_ratio=weak,max_cell_extra_mean_updates=max(extra),stationary_extra_mean_updates=stationary_extra,mean_updates=mean_updates,eligible=passed))
    eligible=[x for x in candidates if x['eligible']]
    chosen=min(eligible,key=lambda x:(x['mean_updates'],x['refit_delay']))['refit_delay'] if eligible else max(cfg['refit_delays'])
    return chosen,candidates,bool(eligible)


def plot(summary,cfg,out,delay):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    names=[s['name'] for s in cfg['scenarios']];fig,axes=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
    for ri,db in enumerate(cfg['ebn0_db']):
        for ci,metric in enumerate(('ber','ensemble_nmse')):
            ax=axes[ri,ci]
            for i,(method,label) in enumerate([('innovation','Innovation L17'),('recovery13','Recovery L13'),('periodic17','Periodic L17'),(f'followup_{delay}','Delayed refit L17'),('matched','True statistics')]):
                rows=[next(r for r in summary if r['scenario']==s and r['ebn0_db']==db and r['receiver']==method and r['phase']=='post_all') for s in names]
                ax.errorbar(np.arange(len(names))+(i-2)*.12,[r[metric] for r in rows],yerr=[r[metric+'_se'] for r in rows],fmt='o',markersize=3,capsize=2,label=label)
            ax.set(yscale='log',xticks=np.arange(len(names)),xticklabels=names,title=f'Initial Eb/N0={db:g} dB',ylabel=metric)
            ax.tick_params(axis='x',rotation=30);ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=6,ncol=2)
    fig.suptitle('28 pilots; fixed Eb/N0 including pilots/CP; 224 post-change frames; mean ±1 episode SE')
    for ext in ('svg','png'):fig.savefig(out/f'followup.{ext}',**({'dpi':160} if ext=='png' else {}))
    p=out/'followup.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('experiments/followup.json'));parser.add_argument('--output',type=Path,default=Path('results/followup'));parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());validate(cfg)
    sources=sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))+[args.config]
    hashes={p.as_posix():digest(p) for p in sources};dev=evaluate(cfg,'development',cfg['refit_delays']);chosen,candidates,passed=select(dev,cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    for k,v in dev.items():
        if v:write_csv(args.output/f'development_{k}.csv',v)
    lock=dict(refit_delay=chosen,candidates=candidates,eligible_candidate_found=passed,source_sha256=hashes,config_sha256=digest(args.config),
        development_runs_sha256=digest(args.output/'development_runs.csv'),locked_before_test_utc=datetime.now(timezone.utc).isoformat(),
        rule='Every post_all mean NMSE <=1.05 innovation; weak15 at18dB post_late <=0.75 innovation; max cell extra mean updates <=1.5 and stationary extra=0; fewest mean updates among eligible, tie shorter delay; else largest delay with failure retained')
    path=args.output/'selection.json';path.write_text(json.dumps(lock,indent=2)+'\n');lock_hash=digest(path);started=datetime.now(timezone.utc).isoformat()
    result=evaluate(cfg,'test',[chosen])
    if any(digest(p)!=h for p,h in hashes.items()) or digest(path)!=lock_hash:raise SystemExit('Sources or lock changed')
    for k,v in result.items():
        if v:write_csv(args.output/(k+'.csv'),v)
    meta=dict(config=cfg,source_sha256=hashes,selection_sha256=lock_hash,test_started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=['python','-m','experiments.followup_ber',*sys.argv[1:]],base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        causality='Current estimate and score precede ingestion; all trigger mechanisms share cooldown; one effective rebuild per frame, usable next frame; no final unused update',
        scope='New-seed replication with fixed dense geometry/energy, synthetic independent block-static taps; no hardware or unseen-scenario certification',
        uncertainty='Whole independent episodes, paired differences vs shared-data baselines; scenarios reuse latent streams; no pooling frames as replicates',
        cost='All schedule bookkeeping is timed; follow-ups reuse cached moments, record probe/model solve time and count effective rebuilds; pending plans at episode end not executed',
        stopping_rule='Development selection persisted before test; no test retuning; failed cells retained')
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.plot:plot(result['summary'],cfg,args.output,chosen)
    checks=[r for r in result['summary'] if r['receiver']!='oracle_zf'];oracle=[r for r in result['summary'] if r['receiver']=='oracle_zf']
    print(json.dumps(dict(refit_delay=chosen,eligible=passed,nmse_pass=sum(r['nmse_consistent'] for r in checks),nmse_total=len(checks),oracle_pass=sum(r['oracle_consistent'] for r in oracle),oracle_total=len(oracle))))
    if not all(r['nmse_consistent'] for r in checks) or not all(r['oracle_consistent'] for r in oracle):raise SystemExit('Consistency failure; retain without retuning')


if __name__=='__main__':main()
