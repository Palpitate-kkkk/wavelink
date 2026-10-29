"""Frozen-policy pilot-density/energy audit: python -m experiments.density_ber --plot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
from wavelink import qpsk_modulate, qpsk_demodulate, ofdm_modulate, ofdm_demodulate
from wavelink.receiver import comb_grid, pilot_noise_variance, apply_multipath, zero_force
from wavelink.estimation import TapDomainLS, TapDomainLMMSE, ensemble_nmse
from wavelink.recovery import RecoveringSupportGuard
from wavelink.tracking import TriggeredPilotLMMSE
from experiments.generalization_ber import timeline
from experiments.recovery_ber import audit_modes
from experiments.lmmse_ber import se
from experiments.multipath_ber import write_csv


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(cfg):
    if digest(cfg['baseline_metadata'])!=cfg['baseline_metadata_sha256']:raise ValueError('baseline metadata changed')
    meta=json.loads(Path(cfg['baseline_metadata']).read_text())
    for p,h in meta['source_sha256'].items():
        if digest(p)!=h:raise ValueError('frozen source changed: '+p)
    # Include all earlier held-out and development seeds, not just last test.
    used=set()
    for name in ['triggered','generalization','guard','recovery']:
        old=json.loads(Path(f'experiments/{name}.json').read_text())
        used.update(old.get('seeds',[]))
        for split in ('development','test'):used.update(old.get(split,{}).get('seeds',[]))
    if len(cfg['seeds'])!=32 or len(set(cfg['seeds']))!=32 or used&set(cfg['seeds']):raise ValueError('new unique seeds for all phases required')
    if cfg['pilot_spacings']!=[4,2] or cfg['budgets']!=['fixed_eb','fixed_noise']:raise ValueError('fixed factorial geometry/budget design required')
    parent=meta['config']
    for k in ('cp_len','sample_rate_hz','calibration_frames','symbols_per_frame','episode_frames','post_frames','guard','exit_threshold','entry_threshold'):
        if cfg[k]!=parent[k]:raise ValueError('frozen protocol changed: '+k)
    if cfg['release_checks']!=8:raise ValueError('frozen recovery confirmation required')
    if len({s['name'] for s in cfg['scenarios']})!=len(cfg['scenarios']):raise ValueError('unique scenarios required')
    for p in cfg['states'].values():
        if np.iscomplexobj(p):raise ValueError('real powers required')
        p=np.asarray(p,float)
        if p.shape!=(17,) or not np.all(np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1,rtol=0,atol=1e-12):raise ValueError('unit-power CP-covered states required')
    for s in cfg['scenarios']:
        ts=[x['offset'] for x in s['changes']]
        if not ts or ts[0]!=0 or ts!=sorted(set(ts)):raise ValueError('ordered changes from zero required')
        for c in s['changes']:
            if isinstance(c['offset'],bool) or not isinstance(c['offset'],int) or not 0<=c['offset']<cfg['post_frames'] or c['state'] not in cfg['states'] or not np.isfinite(c['noise_factor']) or c['noise_factor']<=0:raise ValueError('invalid change')
    return meta


def effective_db(db,layout,cp,budget):
    if budget=='fixed_eb':return float(db)
    if budget!='fixed_noise':raise ValueError('unknown energy budget')
    ratio=pilot_noise_variance(0.,layout,cp)/pilot_noise_variance(0.,comb_grid(4),cp)
    return float(db+10*np.log10(ratio))


def budget_noise(db,layout,cp,budget):
    # Use the very same float across grids at fixed noise, rather than a
    # log/exp round trip whose last bits can depend on the resource ratio.
    if budget=='fixed_noise':return pilot_noise_variance(db,comb_grid(4),cp)
    if budget=='fixed_eb':return pilot_noise_variance(db,layout,cp)
    raise ValueError('unknown energy budget')


def coupled_frame(seed,si,role,frame,powers,noise,layout,cp,symbols):
    """Common TX waveform, taps and standard-noise draws across grids.

    Independent substreams prevent pilot/data allocation from advancing the
    channel or noise RNG differently. Truth remains evaluator-only.
    """
    def rng(part):return np.random.default_rng(np.random.SeedSequence([seed,si,role,frame,part]))
    r=rng(0);h=(r.normal(size=len(powers))+1j*r.normal(size=len(powers)))*np.sqrt(np.asarray(powers)/2)
    active=np.asarray(layout.active_tones);ns=symbols+1
    raw=rng(1).integers(0,2,(ns,len(active),2),dtype=np.uint8)
    qpsk=qpsk_modulate(raw.ravel()).reshape(ns,-1)
    pi=np.searchsorted(active,layout.pilot_tones);di=np.searchsorted(active,layout.data_tones)
    tx=ofdm_modulate(layout.pack(qpsk[:,di],qpsk[:,pi]),cp)
    clean,_=apply_multipath(tx,h)
    r=rng(2);standard=(r.normal(size=clean.shape)+1j*r.normal(size=clean.shape))/np.sqrt(2)
    y=ofdm_demodulate(clean+np.sqrt(noise)*standard,layout.n_fft,cp)[1:]
    truth=np.fft.fft(h,layout.n_fft)[layout.bins(layout.data_tones)]
    return y,qpsk[1:,pi],raw[1:,di,:].reshape(symbols,-1),truth


def geometry(spacing,cp=16):
    layout=comb_grid(spacing);length=cp+1
    fp=np.exp(-2j*np.pi*np.outer(layout.pilot_tones,np.arange(length))/layout.n_fft)
    singular=np.linalg.svd(fp,compute_uv=False);rank=int(np.linalg.matrix_rank(fp))
    identified=rank==length
    return dict(pilot_spacing=spacing,pilots=len(layout.pilot_tones),data_tones=len(layout.data_tones),full_length=length,
        full_rank=rank,full_nullity=length-rank,full_identifiable=identified,
        full_condition=float(singular[0]/singular[-1]) if identified else '',
        full_ls_noise_gain=TapDomainLS(layout,length).noise_gain if identified else '',
        pilot_fraction=len(layout.pilot_tones)/len(layout.active_tones),bits_per_ofdm_symbol=2*len(layout.data_tones),
        expected_energy_per_symbol=len(layout.active_tones)*(1+cp/layout.n_fft))


def run_episode(cfg,spec,db,si,seed,start,spacing,budget):
    layout=comb_grid(spacing);actual_db=effective_db(db,layout,cfg['cp_len'],budget)
    noise0=budget_noise(db,layout,cfg['cp_len'],budget);bins=layout.bins(layout.data_tones)
    common=comb_grid(2).data_tones;common_index=[layout.data_tones.index(k) for k in common]
    calibration=[]
    for t in range(cfg['calibration_frames']):
        y,p,_,_=coupled_frame(seed,si,1,t,cfg['states']['initial'],noise0,layout,cfg['cp_len'],cfg['symbols_per_frame'])
        calibration.append(y[:,layout.bins(layout.pilot_tones)]/p)
    trackers={'recovery13':RecoveringSupportGuard(np.asarray(calibration),layout,13,cfg['cp_len'],threshold=cfg['entry_threshold'],exit_threshold=cfg['exit_threshold'],release_checks=cfg['release_checks'],**cfg['guard'])}
    if spacing==2:trackers['full17']=TriggeredPilotLMMSE(np.asarray(calibration),layout,17,threshold=.75)
    powers,noises=timeline(cfg,spec,start);outside=np.any(powers[:,13:]>0,axis=1)
    metrics={name:[] for name in ['oracle_zf','matched',*trackers]};events=[];cache={}
    for t in range(cfg['episode_frames']):
        noise=noise0*noises[t];power=powers[t]
        key=(power.tobytes(),float(noise))
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
            decoded=qpsk_demodulate(zero_force(y[:,bins],h).ravel()).reshape(len(y),-1,2)
            error=decoded!=bits.reshape(len(y),-1,2);count=int(error.sum())
            metrics[name].append((count/bits.size,float(np.mean(abs(h-truth)**2)),predicted.get(name,0.),count,oracle,float(error[:,common_index,:].mean())))
        if t<cfg['episode_frames']-1:
            for name,tracker in trackers.items():
                if tracker.advance():
                    log=tracker.trigger_log[-1]
                    events.append(dict(receiver=name,after_frame=t,applies_from_frame=t+1,event_kind=log.get('event_kind','primary'),score=log['score']))
    costs=[dict(receiver=name,updates=v.update_count,fit_frame_visits=v.fit_frame_visits,initial_fit_ms=v.initial_fit_ns/1e6,
        total_update_ms=v.update_ns/1e6,feature_frames=v.feature_frames,feature_ms=v.feature_ns/1e6,monitor_checks=v.monitor_checks,monitor_ms=v.monitor_ns/1e6,
        last_update_after_frame=v.last_update_after_frame,maximum_training_lead=-1) for name,v in trackers.items()]
    return {k:np.asarray(v) for k,v in metrics.items()},events,costs,outside


def summarize(runs):
    grouped={}
    for r in runs:grouped.setdefault(tuple(r[k] for k in ('scenario','reference_ebn0_db','budget','pilot_spacing','receiver','phase')),[]).append(r)
    summary=[]
    for key,g in grouped.items():
        row=dict(zip(('scenario','reference_ebn0_db','budget','pilot_spacing','receiver','phase'),key));row['episodes']=len(g)
        for name in ('ber','ensemble_nmse','predicted_nmse','theory_residual','common_data_ber','nominal_correct_raw_bps','benchmark_correct_raw_bps','paired_ber_minus_recovery13','paired_nmse_minus_recovery13'):
            vals=[r[name] for r in g];row[name]=float(np.mean(vals));row['episode_'+name+'_se']=se(vals)
        row['actual_initial_ebn0_db']=g[0]['actual_initial_ebn0_db'];row['initial_noise_variance']=g[0]['initial_noise_variance']
        row['bits']=sum(r['bits'] for r in g);row['errors']=sum(r['errors'] for r in g)
        row['nmse_consistent']=abs(row['theory_residual'])<=6*row['episode_theory_residual_se']+1e-10
        ores=[r['oracle_residual'] for r in g]
        row['oracle_consistent']=bool(abs(np.mean(ores))<=6*se(ores)+1e-10) if key[4]=='oracle_zf' else ''
        summary.append(row)
    return summary


def evaluate(cfg):
    validate(cfg);phases=np.random.default_rng(cfg['schedule_seed']).permutation(32)
    runs=[];events=[];costs=[];responses=[];mode_audit=[]
    for spec in cfg['scenarios']:
        for si,db in enumerate(cfg['ebn0_db']):
            for seed,phase in zip(cfg['seeds'],phases):
                start=64+int(phase)
                shared_sparse=None
                for budget in cfg['budgets']:
                    for spacing in cfg['pilot_spacings']:
                        layout=comb_grid(spacing)
                        if spacing==4 and shared_sparse is not None:
                            values,logs,counts,outside=shared_sparse
                        else:
                            values,logs,counts,outside=run_episode(cfg,spec,db,si,seed,start,spacing,budget)
                            if spacing==4:shared_sparse=(values,logs,counts,outside)
                        context=dict(scenario=spec['name'],reference_ebn0_db=db,budget=budget,pilot_spacing=spacing,seed=seed,onset_frame=start,onset_phase=int(phase))
                        events.extend(dict(context,**e) for e in logs);costs.extend(dict(context,**c) for c in counts)
                        active,ts=audit_modes([e for e in logs if e['receiver']=='recovery13'],outside)
                        mode_audit.append(dict(context,alarms=sum(t['event_kind']=='alarm' for t in ts),releases=sum(t['event_kind']=='release' for t in ts),
                            release_while_outside=sum(t['release_while_outside'] for t in ts),supported_only=int(not np.any(outside))))
                        for j,c in enumerate(spec['changes']):
                            if j==0 or np.any(np.asarray(cfg['states'][c['state']])[13:]>0):continue
                            a=start+c['offset'];b=start+spec['changes'][j+1]['offset'] if j+1<len(spec['changes']) else cfg['episode_frames']
                            delays=[t['after_frame']+1-a for t in ts if t['event_kind']=='release' and a<=t['after_frame']<b-1]
                            responses.append(dict(context,change_frame=a,next_change_or_end=b,release_delay=min(delays) if delays else '',no_effective_release=int(not delays)))
                        windows={'pre':(start-32,start),'post_early':(start,start+64),'post_late':(start+cfg['post_frames']-64,start+cfg['post_frames']),'post_all':(start,start+cfg['post_frames'])}
                        for name,v in values.items():
                            for window_phase,(a,b) in windows.items():
                                g=v[a:b];ref=values['recovery13'][a:b];bits=len(g)*cfg['symbols_per_frame']*2*len(layout.data_tones)
                                rate=2*len(layout.data_tones)*cfg['sample_rate_hz']/(layout.n_fft+cfg['cp_len'])
                                runs.append(dict(context,receiver=name,phase=window_phase,frames=len(g),bits=bits,errors=int(g[:,3].sum()),ber=float(g[:,0].mean()),
                                    ensemble_nmse=float(g[:,1].mean()),predicted_nmse=float(g[:,2].mean()),theory_residual=float(np.mean(g[:,1]-g[:,2])),
                                    oracle_residual=float(np.mean(g[:,0]-g[:,4])) if name=='oracle_zf' else 0.,common_data_ber=float(g[:,5].mean()),
                                    nominal_correct_raw_bps=rate*(1-float(g[:,0].mean())),benchmark_correct_raw_bps=rate*(1-float(g[:,0].mean()))*cfg['symbols_per_frame']/(cfg['symbols_per_frame']+1),
                                    paired_ber_minus_recovery13=float(np.mean(g[:,0]-ref[:,0])),paired_nmse_minus_recovery13=float(np.mean(g[:,1]-ref[:,1])),
                                    actual_initial_ebn0_db=effective_db(db,layout,cfg['cp_len'],budget),initial_noise_variance=budget_noise(db,layout,cfg['cp_len'],budget)))
    summary=summarize(runs)
    # Across geometries compare COMMON data tones with whole-seed paired SE.
    pairs=[]
    sparse_index={tuple(r[k] for k in ('scenario','reference_ebn0_db','budget','seed','phase')):r for r in runs if r['pilot_spacing']==4 and r['receiver']=='recovery13'}
    for r in runs:
        if r['pilot_spacing']!=2 or r['receiver'] not in ('recovery13','full17'):continue
        other=sparse_index[tuple(r[k] for k in ('scenario','reference_ebn0_db','budget','seed','phase'))]
        pairs.append(dict(**{k:r[k] for k in ('scenario','reference_ebn0_db','budget','seed','phase','receiver')},
            common_ber_difference=r['common_data_ber']-other['common_data_ber'],raw_rate_difference=r['nominal_correct_raw_bps']-other['nominal_correct_raw_bps']))
    grouped={}
    for r in pairs:grouped.setdefault(tuple(r[k] for k in ('scenario','reference_ebn0_db','budget','phase','receiver')),[]).append(r)
    paired=[]
    for keys,group in grouped.items():
        row=dict(zip(('scenario','reference_ebn0_db','budget','phase','receiver'),keys));row['episodes']=len(group)
        for key in ('common_ber_difference','raw_rate_difference'):
            row[key]=float(np.mean([r[key] for r in group]));row[key+'_se']=se([r[key] for r in group])
        paired.append(row)
    return dict(runs=runs,summary=summary,paired=paired,events=events,costs=costs,responses=responses,mode_audit=mode_audit,
        geometry=[geometry(s,cfg['cp_len']) for s in cfg['pilot_spacings']])


def plot(summary,cfg,out):
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.fonttype']='none'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,8),constrained_layout=True);names=[s['name'] for s in cfg['scenarios']]
    curves=[(4,'recovery13','16 pilots / recovery L13'),(2,'recovery13','28 pilots / recovery L13'),(2,'full17','28 pilots / learned L17')]
    for ri,budget in enumerate(cfg['budgets']):
        for j,(spacing,method,label) in enumerate(curves):
            rows=[next(r for r in summary if r['scenario']==s and r['reference_ebn0_db']==18 and r['budget']==budget and r['pilot_spacing']==spacing and r['receiver']==method and r['phase']=='post_all') for s in names]
            for ci,key in enumerate(('common_data_ber','nominal_correct_raw_bps')):
                scale=1 if ci==0 else 1e-6
                axes[ri,ci].errorbar(np.arange(len(names))+(j-1)*.15,[r[key]*scale for r in rows],yerr=[r['episode_'+key+'_se']*scale for r in rows],fmt='o',capsize=2,label=label,markersize=4)
        for ci in range(2):
            ax=axes[ri,ci];ax.set(xticks=np.arange(len(names)),xticklabels=names,title=budget,ylabel='BER on common data tones' if ci==0 else 'Nominal correct raw Mbit/s')
            if ci==0:ax.set_yscale('log')
            ax.tick_params(axis='x',rotation=30);ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('Frozen policies, reference Eb/N0=18 dB; post_all mean ±1 whole-episode SE')
    for ext in ('svg','png'):fig.savefig(out/f'density.{ext}',**({'dpi':160} if ext=='png' else {}))
    p=out/'density.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('experiments/density.json'));parser.add_argument('--output',type=Path,default=Path('results/density'));parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());validate(cfg)
    sources=sorted(Path('src').rglob('*.py'))+sorted(Path('experiments').glob('*.py'))+[args.config]
    hashes={p.as_posix():digest(p) for p in sources};started=datetime.now(timezone.utc).isoformat()
    result=evaluate(cfg)
    if any(digest(p)!=h for p,h in hashes.items()):raise SystemExit('Sources changed during audit')
    args.output.mkdir(parents=True,exist_ok=True)
    for k,v in result.items():
        if v:write_csv(args.output/(k+'.csv'),v)
    meta=dict(config=cfg,source_sha256=hashes,started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=['python','-m','experiments.density_ber',*sys.argv[1:]],base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        selection='No new tuning: recovery 8 checks and trigger 0.75 frozen; all declared policies/budgets retained',
        coupling='SeedSequence [seed,SNRindex,role,frame,part]; part 0 channel, 1 all-active-tone QPSK, 2 standard AWGN; same TX waveform across grids; data sets differ',
        energy='All active tones unit QPSK: expected 65 energy units per CP-included symbol; fixed_eb uses per-grid data-bit energy, fixed_noise shares sparse-reference noise',
        rates='Expected correctly decided uncoded raw bits, not coded goodput. Nominal rate includes CP/pilots; benchmark rate also includes per-frame warmup, excludes initial calibration',
        uncertainty='Whole seed/SNR episodes; all grids, budgets and scenarios share latent streams, not independent replications; cross-grid comparisons use common data tones',
        causality='All model updates follow scoring and apply to next frame; last unused update skipped',
        scope='Synthetic independent block-static uncorrelated taps, integer delays, ideal sync, unit ensemble power; no hardware or general safety claim')
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.plot:plot(result['summary'],cfg,args.output)
    checks=[r for r in result['summary'] if r['receiver']!='oracle_zf'];oracle=[r for r in result['summary'] if r['receiver']=='oracle_zf']
    print(json.dumps(dict(nmse_pass=sum(r['nmse_consistent'] for r in checks),nmse_total=len(checks),oracle_pass=sum(r['oracle_consistent'] for r in oracle),oracle_total=len(oracle))))
    if not all(r['nmse_consistent'] for r in checks) or not all(r['oracle_consistent'] for r in oracle):raise SystemExit('Consistency failure, retain results without retuning')


if __name__=='__main__':main()
