import unittest,copy
from types import SimpleNamespace
from shared_error import *
from audit_shared import solve,model_check,daily_check


def fixture(n=63):
    days=pd.bdate_range('2025-01-02',periods=n);records=[];history={};calendar={}
    rng=np.random.default_rng(14)
    for i,d in enumerate(days):
        close=pd.Timestamp(str(d.date())+' '+('13:00' if i==20 else '16:00'),tz=core.ET).tz_convert('UTC');calendar[d]={'close_at':close}
        for c in CLOCKS:
            at=pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC')
            if at>=close:continue
            state=rng.normal(size=6);raw=rng.normal(0,.02,size=2);C,D=rng.normal(0,.02,size=2)
            records.append({'at':at,'date':d,'clock':c,'feature_ready':True,'terminal_release':close+pd.Timedelta(seconds=1),'C':C,'D':D,**dict(zip(STATE_COLUMNS,state))})
            history[at]=raw
    panel=pd.DataFrame(records).set_index('at');return panel,history,SimpleNamespace(calendar=calendar)


class SharedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p,cls.h,cls.m=fixture();cls.at=cls.p.index[-4];cls.reg=registry(cls.p,cls.m);cls.daily=fit_day(cls.p,cls.h,cls.reg,cls.at)

    def test_halfday_has_three_rows_but_one_date_weight(self):
        r=self.daily;self.assertTrue(r['ready']);self.assertIn(3,[len(p['members']) for p in r['parents']])
        for i,w in enumerate(r['day_weights']):self.assertAlmostEqual(sum(r['row_weights'][r['row_groups']==i]),w)
        self.assertAlmostEqual(sum(r['row_weights']),len(r['parents']))

    def test_independent_all_bootstrap_equations(self):
        from run_planning import json_ready
        import json
        saved=json.loads(json.dumps(json_ready(self.daily),default=str));models,n=daily_check(self.p,self.m,self.h,self.at,saved)
        self.assertEqual(n,1200);self.assertEqual(set(models),set(VARIANTS))

    def test_current_day_not_in_training(self):
        parents,_=pool(self.reg,self.h,self.at);self.assertTrue(all(r['date']<self.p.loc[self.at,'date'] for r in parents))

    def test_one_pending_clock_stops_entire_prefix(self):
        reg=copy.deepcopy(self.reg);reg[10]['release']=None;parents,fb=pool(reg,self.h,self.at)
        self.assertEqual(len(parents),10);self.assertTrue(fb['expired_unresolved_prefix'])

    def test_exact_cutoff_excluded(self):
        reg=copy.deepcopy(self.reg);reg[10]['release']=day_cutoff(self.at);parents,_=pool(reg,self.h,self.at);self.assertEqual(len(parents),10)

    def test_missing_forecast_does_not_become_zero(self):
        h=self.h.copy();h[self.reg[10]['members'][0]]=None;parents,fb=pool(self.reg,h,self.at)
        self.assertNotIn(self.reg[10]['date'],[r['date'] for r in parents]);self.assertEqual(fb['selected_early_dates'],62)

    def test_future_changes_cannot_change_members(self):
        p=self.p.copy();p.loc[p.date>=p.loc[self.at,'date'],['C','D']]=999
        parents,_=pool(registry(p,self.m),self.h,self.at);old,_=pool(self.reg,self.h,self.at)
        self.assertEqual(parents,old)

    def test_120_dates_not_120_clock_rows(self):
        p,h,m=fixture(150);parents,fb=pool(registry(p,m),h,p.index[-4]);self.assertEqual(len(parents),120);self.assertGreater(fb['candidate_rows'],470)

    def test_incomplete_feature_date_excluded_as_date(self):
        p=self.p.copy();p.loc[self.reg[10]['members'][0],'feature_ready']=False
        parents,_=pool(registry(p,self.m),self.h,self.at);self.assertNotIn(self.reg[10]['date'],[r['date'] for r in parents])

    def test_zero_weight_rows_cannot_affect_scale_or_fit(self):
        rng=np.random.default_rng(31);X=rng.normal(size=(20,11));Y=rng.normal(size=(20,2));w=np.ones(20);w[-1]=0
        a=fit_map(X,Y,w,'interaction',10);X[-1]=np.nan;Y[-1]=np.nan;b=fit_map(X,Y,w,'interaction',10)
        self.assertEqual(a,b)

    def test_constant_columns_have_finite_scale(self):
        X=np.ones((80,11));Y=np.column_stack((np.arange(80)/1000,-np.arange(80)/1000));m=fit_map(X,Y,np.ones(80),'interaction',20)
        np.testing.assert_allclose(forecast(m,X[:1])[0],Y.mean(axis=0))

    def test_interaction_represents_changing_slope(self):
        rng=np.random.default_rng(4);X=rng.normal(size=(1000,11));y=X[:,0]*X[:,2];Y=np.column_stack((y,-y))
        a=fit_map(X,Y,np.ones(1000),'interaction',250);b=fit_map(X,Y,np.ones(1000),'additive',250)
        self.assertLess(np.mean((forecast(a,X)-Y)**2),np.mean((forecast(b,X)-Y)**2)/10)

    def test_common_band_and_abstention(self):
        d=issue_shared(self.p,self.at,self.h[self.at],self.daily)
        for k in VARIANTS:np.testing.assert_allclose(np.array(d[k]['point'])-d[k]['lower'],[d[k]['radius']]*2)
        e=decide(d['interaction'],{'expired_unresolved_prefix':True},True);self.assertEqual(e['action'],'CASH')
        self.assertFalse(issue_shared(self.p,self.at,None,self.daily)['interaction']['ready'])


if __name__=='__main__':unittest.main()
