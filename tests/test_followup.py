import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from wavelink.followup import FollowupPilotLMMSE
from wavelink.innovation import InnovationTriggeredLMMSE
from wavelink.statistics import learn_pilot_statistics
from experiments.density_ber import coupled_frame
from experiments.followup_ber import validate, episode, evaluate, select


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads(Path('experiments/followup.json').read_text())
        from wavelink.receiver import comb_grid
        self.layout=comb_grid(2);self.cal=[]
        for t in range(64):
            y,p,_,_=coupled_frame(191001,0,1,t,self.cfg['states']['initial'],.02,self.layout,16,4)
            self.cal.append(y[:,self.layout.bins(self.layout.pilot_tones)]/p)
        self.cal=np.asarray(self.cal)

    def frame(self,t):return coupled_frame(191001,0,2,t,self.cfg['states']['weak15'],.02,self.layout,16,4)[:2]

    def scheduled(self,delay=32,observed=(8,)):
        g=FollowupPilotLMMSE(self.cal,self.layout,17,refit_delay=delay)
        # Isolate the scheduling state machine from stochastic detector scores.
        g.threshold=1e20;g._innovation_score=lambda noise:3. if g.frames_observed in observed else 1.
        return g

    def step(self,g,t):
        y,p=self.frame(t);expected=g.model.estimate(y,p);before=g.model.mapping.copy()
        np.testing.assert_array_equal(g.estimate(y,p),expected)
        np.testing.assert_array_equal(g.model.mapping,before)
        return g.advance()

    def test_delayed_fit_is_once_causal_and_uses_recent_history(self):
        g=self.scheduled()
        for t in range(40):
            self.step(g,t)
            if t<39:self.assertEqual(g.followup_count,0)
        self.assertEqual(g.followup_count,1);self.assertIsNone(g.pending_refit_after_frame)
        self.assertEqual([e['after_frame'] for e in g.trigger_log],[7,39])
        self.assertEqual(g.trigger_log[-1]['event_kind'],'followup')
        independent=learn_pilot_statistics(np.asarray(g.history),self.layout,17)
        np.testing.assert_allclose(g.statistics.powers,independent.powers,rtol=1e-10,atol=1e-12)
        for t in range(40,128):self.step(g,t)
        self.assertEqual(g.followup_count,1);self.assertEqual(g.armed_plans,1)
        self.assertEqual(g.update_count,2);self.assertEqual(g.fit_frame_visits,0)

    def test_new_innovation_at_due_time_replaces_old_plan(self):
        g=self.scheduled(observed=(8,40))
        for t in range(40):self.step(g,t)
        self.assertEqual(g.followup_count,0);self.assertEqual(g.cancelled_plans,1)
        self.assertEqual(g.pending_refit_after_frame,71);self.assertEqual(g.pending_origin,39)
        self.assertEqual(g.update_count,2)
        for t in range(40,72):self.step(g,t)
        self.assertEqual(g.followup_count,1);self.assertEqual(g.update_count,3)
        self.assertEqual(g.schedule_log[-1]['origin_after_frame'],39)
        self.assertEqual(g.schedule_log[-1]['due_after_frame'],71)

    def test_statistics_only_update_cancels_without_rearming(self):
        g=self.scheduled(delay=64)
        for t in range(96):
            g.threshold=1e-12 if t==39 else 1e20
            self.step(g,t)
        self.assertEqual(g.cancelled_plans,1);self.assertEqual(g.armed_plans,1)
        self.assertEqual(g.followup_count,0);self.assertIsNone(g.pending_refit_after_frame)
        self.assertEqual(g.trigger_log[-1]['event_kind'],'statistics')

    def test_end_of_episode_can_retain_unexecuted_plan(self):
        g=self.scheduled(delay=64)
        for t in range(40):self.step(g,t)
        self.assertEqual(g.pending_refit_after_frame,71);self.assertEqual(g.followup_count,0)
        self.assertEqual(g.armed_plans,g.cancelled_plans+g.followup_count+1)
        self.assertEqual(g.update_count,len(g.trigger_log))

    def test_followup_refreshes_covariance_probe_and_shared_cooldown(self):
        g=self.scheduled()
        for t in range(64):self.step(g,t)
        p=g.statistics.powers;fp=g.model.pilot_design;lam=g.statistics.noise_variance/g.symbols
        np.testing.assert_allclose(g.innovation_matrix@((fp*p)@fp.conj().T+lam*np.eye(28)),lam*np.eye(28),atol=1e-13)
        self.assertEqual(g.last_trigger_observed,40)
        self.assertTrue(all(b-a>=32 for a,b in zip([r['after_frame'] for r in g.trigger_log],[r['after_frame'] for r in g.trigger_log][1:])))

    def test_unarmed_method_is_exact_innovation_ablation(self):
        a=FollowupPilotLMMSE(self.cal,self.layout,17,innovation_threshold=1e20)
        b=InnovationTriggeredLMMSE(self.cal,self.layout,17,innovation_threshold=1e20)
        for t in range(64):
            y,p=self.frame(t);np.testing.assert_array_equal(a.estimate(y,p),b.estimate(y,p))
            self.assertEqual(a.advance(),b.advance());np.testing.assert_array_equal(a.model.mapping,b.model.mapping)
        self.assertEqual(a.armed_plans,0)

    def test_future_suffix_cannot_change_prefix(self):
        a=self.scheduled();b=self.scheduled()
        for t in range(64):
            y,p=self.frame(t);other=y.copy()
            if t>=48:other[:]=1e3
            left=a.estimate(y,p);right=b.estimate(other,p)
            if t<48:np.testing.assert_array_equal(left,right)
            a.advance();b.advance()

    def test_invalid_delay_and_timing_arguments(self):
        for delay in [True,0,31,33,32.5,'32',None]:
            with self.assertRaises(ValueError):FollowupPilotLMMSE(self.cal,self.layout,17,refit_delay=delay)
        for kw in [dict(check_every=0),dict(cooldown=True),dict(cooldown=64,refit_delay=32)]:
            with self.assertRaises(ValueError):FollowupPilotLMMSE(self.cal,self.layout,17,**kw)
        g=self.scheduled()
        with self.assertRaises(RuntimeError):g.advance()

    def test_old_seed_and_baseline_isolation(self):
        validate(self.cfg)
        for seed in [161001,171001,181001]:
            bad=copy.deepcopy(self.cfg);bad['test']['seeds'][0]=seed
            with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(self.cfg);bad['innovation_threshold']=4.
        with self.assertRaises(ValueError):validate(bad)

    def test_selection_cost_tie_break_and_failed_gate(self):
        cfg=self.cfg;rows=[];costs=[]
        for s in cfg['scenarios']:
            for db in cfg['ebn0_db']:
                for phase in ('post_all','post_late'):
                    for name,value in [('innovation',1.),('followup_32',.5),('followup_64',.5)]:rows.append(dict(scenario=s['name'],ebn0_db=db,receiver=name,phase=phase,ensemble_nmse=value))
                for name in ['innovation','followup_32','followup_64']:
                    count=2 if name=='followup_32' and s['name']!='stationary' else 1
                    costs.append(dict(scenario=s['name'],ebn0_db=db,receiver=name,updates=count))
        r=dict(summary=rows,costs=costs)
        self.assertTrue(select(r,cfg)[2]);self.assertEqual(select(r,cfg)[0],64)
        for x in costs:x['updates']=1
        self.assertEqual(select(r,cfg)[0],32)
        for x in rows:
            if x['scenario']=='weak15' and x['receiver'].startswith('followup'):x['ensemble_nmse']=.9
        self.assertFalse(select(r,cfg)[2]);self.assertEqual(select(r,cfg)[0],64)

    def test_evaluation_and_small_real_episode(self):
        cfg=copy.deepcopy(self.cfg);cfg['scenarios']=cfg['scenarios'][:2]
        def fake(cfg,spec,db,si,seed,start,delays):
            names=['oracle_zf','matched','legacy17','recovery13','periodic17','innovation']+[f'followup_{v}' for v in delays]
            return {n:np.zeros((320,5)) for n in names},[],[],[]
        with patch('experiments.followup_ber.episode',side_effect=fake):r=evaluate(cfg,'development',[32])
        self.assertEqual(len(r['runs']),2*8*2*7*4)
        json.dumps(dict(oracle_pass=sum(x['oracle_consistent'] for x in r['summary'] if x['receiver']=='oracle_zf')))
        cfg.update(episode_frames=16,symbols_per_frame=2)
        values,events,counts,plans=episode(cfg,cfg['scenarios'][0],6,0,191001,4,[32])
        self.assertEqual(values['followup_32'].shape,(16,5))
        self.assertTrue(all(e['applies_from_frame']==e['after_frame']+1<16 for e in events))
        self.assertTrue(all(c['last_update_after_frame']<15 for c in counts))


if __name__=='__main__':unittest.main()
