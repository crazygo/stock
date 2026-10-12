"""Causal inputs, matched model fitting and as-issued calibration boundaries."""
import unittest,json
from coherence_learning import *
from audit_features import reference_values,audit_features
from audit_calibration import daily_check
from audit_models import model_equations
from test_shared import fixture as old_fixture
from run_planning import json_ready


def wire(x):return json.loads(json.dumps(json_ready(x),default=str))


def market_fixture():
    rng=np.random.default_rng(744);dates=pd.bdate_range('2025-01-02',periods=35);d=dates[25]
    at=pd.Timestamp(str(d.date())+' 09:25:30',tz=core.ET).tz_convert('UTC');bars={};daily={};row={'date':d,'clock':CLOCKS[0]}
    for j,s in enumerate(PANEL):
        prices=100*np.cumprod(1+rng.normal(.001,.012,len(dates)));av=pd.DatetimeIndex([pd.Timestamp(str(x.date())+' 16:00:01',tz=core.ET).tz_convert('UTC') for x in dates])
        daily[s]=pd.DataFrame({'close':prices,'complete':True,'available_at':av},index=dates)
        t=at-pd.Timedelta(seconds=30);value=prices[24]*(1+(j-3)*.002)
        bars[s]=pd.DataFrame({'econ_close':[value],'end_at':[t],'available_at':[t+pd.Timedelta(seconds=1)],'date':[d]},index=pd.DatetimeIndex([t]))
        row['leader_'+s+'_today']=value/prices[24]-1;row['leader_'+s+'_ret5']=prices[24]/prices[19]-1
    return pd.DataFrame([row],index=pd.DatetimeIndex([at])),(bars,daily,{}),dates


def learning_fixture(n=126):
    p,h,m=old_fixture(n);rng=np.random.default_rng(345);p['release']=p.index+pd.Timedelta(days=14)
    for c in sorted(set(sum((columns(clock,'full') for clock in CLOCKS),[]))):
        if c not in p:p[c]=rng.normal(0,.5,len(p))
    return p,h,m


class CoherenceTests(unittest.TestCase):
    def test_broad_rise_vs_single_leader(self):
        self.assertEqual(participation(np.ones(8)),1.);self.assertEqual(participation([8,0,0,0,0,0,0,0]),.125)
        self.assertEqual(participation([1]*4+[-1]*4),0.)

    def test_participation_bounds_sign_and_scale(self):
        rng=np.random.default_rng(5)
        for z in rng.normal(size=(100,8)):
            self.assertLessEqual(abs(participation(z)),1+1e-12);self.assertAlmostEqual(participation(z),-participation(-z));self.assertAlmostEqual(participation(z),participation(7*z))
        self.assertEqual(participation(np.zeros(8)),0.)

    def test_synchrony_and_opposing_groups(self):
        x=np.sin(np.arange(20));a=np.repeat(x[:,None],8,axis=1)
        self.assertAlmostEqual(correlation_mean(a),1.);a[:,4:]*=-1;self.assertAlmostEqual(correlation_mean(a),-1/7)

    def test_constant_history_unavailable(self):
        R=np.zeros((20,8));self.assertIsNone(measurements(R,np.ones(8)))

    def test_independent_measurements(self):
        rng=np.random.default_rng(82);R=rng.normal(.001,.01,(20,8));current=rng.normal(0,.01,8)
        prices=np.vstack([np.ones(8),np.cumprod(1+R,axis=0)]);a=measurements(R,current);b=reference_values(prices,current)
        np.testing.assert_allclose(list(a.values()),list(b.values()),atol=1e-12)

    def test_laggards_catching_up(self):
        R=np.sin(np.arange(20))[:,None]*.01+np.arange(8)[None,:]*.001
        now=np.array([.02]*4+[0.]*4);self.assertGreater(measurements(R,now)['laggard_catchup'],0)

    def test_feature_audit_on_known_price_paths(self):
        f,l,dates=market_fixture();x,r=build(f,l,dates);out=audit_features(f,x,wire(r),l,dates)
        self.assertTrue(out['passed']);self.assertEqual(out['ready_ticks'],1)

    def test_current_day_daily_close_and_future_data_cannot_leak(self):
        f,l,dates=market_fixture();a,r=build(f,l,dates);day=f.date.iloc[0]
        for s in PANEL:l[1][s].loc[l[1][s].index>=day,'close']=999999
        b,q=build(f,l,dates);pd.testing.assert_frame_equal(a,b);self.assertEqual(wire(r),wire(q))

    def test_delayed_or_stale_bar_abstains(self):
        f,l,dates=market_fixture();at=f.index[0];s=PANEL[0];l[0][s]['available_at']=at+pd.Timedelta(seconds=1)
        a,r=build(f,l,dates);self.assertFalse(a.coherence_ready.iloc[0]);self.assertIn('unavailable',r[0]['reason'])

    def test_incomplete_history_not_filled(self):
        f,l,dates=market_fixture();l[1][PANEL[0]].loc[dates[10],'complete']=False
        a,_=build(f,l,dates);self.assertFalse(a.coherence_ready.iloc[0]);self.assertTrue(a[COHERENCE].isna().all().all())

    def test_each_stock_price_units_do_not_change_features(self):
        f,l,dates=market_fixture();a,_=build(f,l,dates)
        for i,s in enumerate(PANEL):l[0][s]['econ_close']*=i+2;l[1][s]['close']*=i+2
        b,_=build(f,l,dates);np.testing.assert_allclose(a[COHERENCE],b[COHERENCE],atol=1e-12)

    def test_raw_model_equations_and_future_isolation(self):
        p,h,m=learning_fixture();at=p.index[-8];models,records,raw=fit_bases(p,at);same=p[p.clock==p.loc[at,'clock']]
        for k in VARIANTS:
            ids=records[k]['fit']['members'];want,n=model_equations(same,ids,at,models[k],records[k]['fit'],columns(p.loc[at,'clock'],k))
            self.assertEqual(n,3);np.testing.assert_allclose(raw[k],[want[s] for s in core.ETFS])
        p.loc[p.date>p.loc[at,'date'],COHERENCE+['C','D']]=99999
        p.loc[p.date>=p.loc[at,'date'],['C','D']]=99999
        _,new,newraw=fit_bases(p,at);self.assertEqual(wire(records),wire(new))
        for k in VARIANTS:np.testing.assert_array_equal(raw[k],newraw[k])

    def test_calibration_uses_old_forecasts_and_joint_date_blocks(self):
        p,h,m=learning_fixture(63);at=p.index[-4];hist={k:{t:v+(j-1)*.005 for t,v in h.items()} for j,k in enumerate(VARIANTS)}
        d=fit_calibration(p,hist,registry(p,m),at);_,n=daily_check(p,m,hist,at,wire(d));self.assertEqual(n,1200)
        for k in VARIANTS:hist[k][at]=np.array([999.,999.])
        e=fit_calibration(p,hist,registry(p,m),at);self.assertEqual(wire(d),wire(e))

    def test_unknown_earlier_terminal_blocks_later_dates(self):
        p,h,m=learning_fixture(63);at=p.index[-4];p.loc[p.index[20],'terminal_release']=pd.NaT
        hist={k:h.copy() for k in VARIANTS};d=fit_calibration(p,hist,registry(p,m),at)
        self.assertFalse(d['ready']);self.assertEqual(len(d['parents']),5);self.assertTrue(d['feedback']['expired_unresolved_prefix'])


if __name__=='__main__':unittest.main()
