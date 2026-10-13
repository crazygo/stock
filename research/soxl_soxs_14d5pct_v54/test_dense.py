"""Clock coverage, nested calibration, date weight and causal-boundary tests."""
import unittest,copy,tempfile
from types import SimpleNamespace
from dense_learning import *
from audit_dense_calibration import solve,model_check,daily_check
from hierarchical_error import fit_map as old_fit_map
from run_dense import authorize_cases
from run_planning import json_ready
import json


def fixture(n=64):
    days=pd.bdate_range('2025-01-02',periods=n);cal={};rows=[];history={};rng=np.random.default_rng(882)
    for i,day in enumerate(days):
        half=i%15==0;closing='13:00' if half else '16:00';cal[day]={'open_at':pd.Timestamp(str(day.date())+' 09:30',tz=core.ET),'close_at':pd.Timestamp(str(day.date())+' '+closing,tz=core.ET)}
        for c in CLOCKS:
            at=pd.Timestamp(str(day.date())+' '+c,tz=core.ET).tz_convert('UTC')
            if at>=cal[day]['close_at']:continue
            rows.append({'at':at,'date':day,'clock':c,'feature_ready':True,'terminal_release':at+pd.Timedelta(minutes=5),'C':rng.normal(0,.05),'D':rng.normal(0,.04)})
            history[at]=rng.normal(0,.03,2)
    return pd.DataFrame(rows).set_index('at'),SimpleNamespace(calendar=cal),history


class Tests(unittest.TestCase):
    def test_json_unknown_payoff_remains_unknown(self):
        from preflight_inputs import pair_targets
        self.assertTrue(np.isnan(pair_targets([{'net_return':None},{'net_return':.05}])).all())
    def arrays(self,k):
        r=np.random.default_rng(57);X=np.column_stack([r.normal(size=(320,2)),np.tile(np.arange(k),320//k)]);return X,r.normal(size=(320,2)),r.uniform(.1,2,320)
    def test_four_clock_original_solver_exact(self):
        X,Y,w=self.arrays(4);self.assertEqual(fit_map(X,Y,w,80,4),old_fit_map(X,Y,w,'partial',80))
    def test_eight_clock_independent_joint_equations(self):
        X,Y,w=self.arrays(8);model_check(fit_map(X,Y,w,40),solve(X,Y,w,40,8))
    def test_penalty_has_same_mean_strength(self):
        b=np.array([.4,-.7])
        for k in [1,4,8]:
            v=np.tile(b,k);self.assertAlmostEqual(float(v@penalty(k)@v),20*sum(b*b));self.assertGreater(np.linalg.eigvalsh(penalty(k)).min(),0)
    def test_penalty_matches_explicit_deviations(self):
        r=np.random.default_rng(91);b=r.normal(size=(8,2));a=b.mean(axis=0)
        self.assertAlmostEqual(float(b.ravel()@penalty(8)@b.ravel()),20*sum(a*a)+20*np.sum((b-a)**2))
    def test_duplicate_rows_do_not_create_weight(self):
        X,Y,w=self.arrays(8);a=fit_map(X,Y,w,40);b=fit_map(np.repeat(X,2,axis=0),np.repeat(Y,2,axis=0),np.repeat(w/2,2),40);model_check(a,b)
    def test_zero_weight_rows_do_not_affect_fit(self):
        X,Y,w=self.arrays(8);a=fit_map(X,Y,w,40);model_check(a,fit_map(np.r_[X,[[1e12,-1e12,0]]],np.r_[Y,[[1e9,-1e9]]],np.r_[w,0],40))
    def test_missing_weighted_clock_rejected(self):
        X,Y,w=self.arrays(8);w[X[:,2]==7]=0
        with self.assertRaises(AssertionError):fit_map(X,Y,w,40)
    def test_future_labels_do_not_change_daily_model(self):
        p,m,h=fixture();at=p[p.date==p.date.max()].index[0];a=fit_day(p,h,registry(p,m),at);q=p.copy();q.loc[q.date>=p.loc[at,'date'],['C','D']]=1e8
        b=fit_day(q,h,registry(q,m),at);self.assertEqual(json_ready(a),json_ready(b))
    def test_current_forecasts_not_used_in_training(self):
        p,m,h=fixture();at=p[p.date==p.date.max()].index[0];a=fit_day(p,h,registry(p,m),at);hh=copy.deepcopy(h)
        for t in p[p.date>=p.loc[at,'date']].index:hh[t]=np.array([1e8,-1e8])
        self.assertEqual(json_ready(a),json_ready(fit_day(p,hh,registry(p,m),at)))
    def test_whole_dates_weights_and_199_bootstraps(self):
        p,m,h=fixture();at=p[p.date==p.date.max()].index[0];a=fit_day(p,h,registry(p,m),at);saved=json.loads(json.dumps(json_ready(a),default=str));_,n=daily_check(p,m,h,at,saved)
        self.assertEqual(n,400);self.assertEqual(len(a['model']['bootstrap_models']),199)
        for i,p0 in enumerate(a['parents']):self.assertAlmostEqual(sum(np.array(a['row_weights'])[np.array(a['row_groups'])==i]),a['day_weights'][i])
    def test_halfday_requires_five_not_eight(self):
        p,m,h=fixture();r=registry(p,m);self.assertEqual(len(r[0]['members']),5);self.assertTrue(r[0]['feature_ready'])
    def test_missing_actual_clock_not_imputed(self):
        p,m,h=fixture();t=p.index[3];day=p.loc[t,'date'];q=p.drop(t);r=next(x for x in registry(q,m) if x['date']==day);self.assertFalse(r['feature_ready'])
    def test_unknown_prefix_blocks_later_mature_dates(self):
        p,m,h=fixture();first=p.date.min();p.loc[p.date==first,'terminal_release']=pd.NaT;at=p.index[-1];a=fit_day(p,h,registry(p,m),at)
        self.assertFalse(a['ready']);self.assertTrue(a['feedback']['expired_unresolved_prefix']);self.assertEqual(a['members'],[])
    def test_before_sixty_date_models_abstain(self):
        p,m,h=fixture(59);at=p.index[-1];d=fit_day(p,h,registry(p,m),at);info=issue(p,at,h[at],d);self.assertFalse(info['ready'])
    def test_query_missing_abstains(self):
        p,m,h=fixture();at=p.index[-1];d=fit_day(p,h,registry(p,m),at);p.loc[at,'feature_ready']=False;self.assertFalse(issue(p,at,h[at],d)['ready'])
    def test_new_and_old_schedule_routes(self):
        info={'ready':True,'point':[.1,.02],'lower':[.06,-.02]};fb={'expired_unresolved_prefix':False};day=pd.Timestamp('2026-01-05')
        for c in CLOCKS:
            at=pd.Timestamp('2026-01-05 '+c,tz=core.ET);old={'ready':True,'action':'SOXS','allocation':1.};e=events(at,day,c,info,fb,old)
            self.assertEqual(e['dense_band']['action'],'SOXL');self.assertEqual(e['old_four_band']['action'],'SOXS' if c in OLD_CLOCKS else 'CASH');self.assertEqual(e['dense_four_only']['action'],'SOXL' if c in OLD_CLOCKS else 'CASH');self.assertEqual(e['added_only']['action'],'CASH' if c in OLD_CLOCKS else 'SOXL')
    def test_base_training_uses_only_same_clock_and_past(self):
        p,m,h=fixture(123);r=np.random.default_rng(98)
        for c in set(sum([columns(clock,'full') for clock in CLOCKS],[])):p[c]=r.normal(size=len(p))
        p['release']=p.terminal_release;p['M']=abs(p.D);at=p.index[-1];_,a,_=fit_base(p,at);ids=pd.to_datetime(a['fit']['members'],utc=True)
        self.assertEqual(set(p.loc[ids,'clock']),{p.loc[at,'clock']});self.assertTrue((p.loc[ids,'date']<p.loc[at,'date']).all());q=p.copy();q.loc[q.index>=at,'D']=999
        _,b,_=fit_base(q,at);self.assertEqual(json_ready(a),json_ready(b))
    def test_nonpositive_net_estimate_cash(self):
        info={'ready':True,'point':[-.01,-.02],'lower':[-.02,-.03]};self.assertEqual(decide(info,{'expired_unresolved_prefix':False},True)['action'],'CASH')
    def test_primary_failure_not_rescued_by_control(self):
        p=read(SOURCE/'protocol.json');s={'completed_trades':30,'monthly':[{'month':m,'return':v,'mark_complete':True} for m,v in zip(p['development']['months'],[.3,.3,-.01,.3])]};self.assertFalse(development_gate(s,True,p)['passed'])

if __name__=='__main__':unittest.main()
