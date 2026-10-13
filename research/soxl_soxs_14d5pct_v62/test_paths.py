import unittest,copy
from types import SimpleNamespace
from path_learning import *
from audit_paths import reference_values,reference_kernel
from risk import first_exit


def fixture(days=4):
    n=6;b={'ohlc':np.full((n,78,2,4),100.),'quotes':np.full((n,8,2),100.),'previous_close':np.full((n,2),100.),'split_ratio':np.ones((n,2)),'dividend_ratio':np.zeros((n,2)),'blocks':np.array([[i,i+1] for i in range(n-1)])}
    plan={'dates':list(pd.bdate_range('2026-01-05',periods=days)),'bar_counts':[78]*days,'first_offset':0};selected=np.tile(np.arange((days+1)//2),(5,1));return b,plan,selected


def check(b,p,s):
    r=simulate(b,s,CLOCKS[0],np.array([100.,100.]),p);v=reference_values(b,s,CLOCKS[0],np.array([100.,100.]),p)
    for k in r:np.testing.assert_allclose(r[k],v[k],atol=1e-10)
    return r

class PathTests(unittest.TestCase):
    def test_expiry_costs(self):
        b,p,s=fixture();r=check(b,p,s);self.assertTrue((r['reason']==3).all());np.testing.assert_allclose(r['net'],(.9995**2)/(1.0005**2)-1)
    def test_same_bar_stop_first(self):
        b,p,s=fixture();b['ohlc'][0,4,:,1]=110;b['ohlc'][0,4,:,2]=90;r=check(b,p,s);self.assertTrue((r['reason']==2).all());self.assertTrue((r['exit_bar']==4).all())
    def test_adverse_gap(self):
        b,p,s=fixture();b['ohlc'][1,0]=[[80,85,75,81],[80,85,75,81]];r=check(b,p,s);np.testing.assert_allclose(r['fill'],80*.9995)
    def test_target_cent_rounding(self):
        b,p,s=fixture();b['ohlc'][0,2,:,1]=110;r=check(b,p,s);np.testing.assert_allclose(r['fill'],105.06);self.assertTrue((r['reason']==1).all())
    def test_first_day_dividend_excluded(self):
        b,p,s=fixture(1);b['dividend_ratio'][0]=.1;r=check(b,p,s);np.testing.assert_allclose(r['cash'],0)
    def test_dividend_before_exit_included_once(self):
        b,p,s=fixture();b['dividend_ratio'][1]=.02;b['ohlc'][1,3,:,1]=110;r=check(b,p,s);np.testing.assert_allclose(r['cash'],2.)
    def test_dividend_after_exit_excluded(self):
        b,p,s=fixture();b['dividend_ratio'][1]=.02;b['ohlc'][0,3,:,1]=110;r=check(b,p,s);np.testing.assert_allclose(r['cash'],0)
    def test_split_economic_tick(self):
        b,p,s=fixture();b['split_ratio'][1]=10.;b['ohlc'][1,3,:,1]=106;r=check(b,p,s);np.testing.assert_allclose(r['fill'],105.1)
    def test_source_prefix_before_entry_ignored(self):
        b,p,s=fixture();p['first_offset']=12;b['ohlc'][0,5,:,1]=120;b['ohlc'][0,5,:,2]=80;r=check(b,p,s);self.assertTrue((r['reason']==3).all())
    def test_halfday_target_truncation(self):
        b,p,s=fixture(1);p['bar_counts']=[42];b['ohlc'][0,45,:,1]=120;r=check(b,p,s);self.assertTrue((r['reason']==3).all());self.assertTrue((r['exit_bar']==41).all())
    def test_scalar_original_risk_engine(self):
        b,p,s=fixture(2);b['ohlc'][0,12,0,1]=106;b['ohlc'][1,3,1,2]=90;r=check(b,p,s)
        for j in range(2):
            vals=b['ohlc'][:2,:,j,:].reshape(-1,4);frame=pd.DataFrame(vals,columns=['econ_open','econ_high','econ_low','econ_close']);frame['share_factor']=1.;frame['start_at']=pd.date_range('2026-01-05',periods=156,freq='5min',tz='UTC');frame['available_at']=frame.start_at+pd.Timedelta(minutes=5,seconds=1)
            v=first_exit(frame,r['entry'][0,j],np.ceil(r['entry'][0,j]*1.05*100-1e-10)/100,.06,None,.0005,.0005);self.assertAlmostEqual(v['fill'],r['fill'][0,j]);self.assertEqual(v['offset'],r['exit_day'][0,j]*78+r['exit_bar'][0,j])
    def test_paired_paths_not_inverse_substitution(self):
        b,p,s=fixture();b['ohlc'][0,3,0,1]=106;b['ohlc'][0,3,1,2]=90;r=check(b,p,s);self.assertTrue((r['reason'][:,0]==1).all());self.assertTrue((r['reason'][:,1]==2).all())
    def test_mean_log_and_direction_identity(self):
        b,p,s=fixture();b['ohlc'][0,3,0,1]=106;r=check(b,p,s);z=summarize(r);self.assertAlmostEqual(z['M']*(2*z['q']-1),z['D']);self.assertAlmostEqual(z['C']+z['D'],r['log_returns'][:,0].mean())
    def test_kernel_independent_and_constant(self):
        x=np.random.default_rng(1).normal(size=(160,4))
        for k in VARIANTS:
            a=kernel(x,np.ones(4),'unconditional' if k=='unconditional' else 'conditional');b=reference_kernel(x,np.ones(4),k!='unconditional')
            for f in ['scale','probability','effective_dates']:np.testing.assert_allclose(a[f],b[f],atol=1e-12)
        z=kernel(np.ones((160,4)),np.ones(4),'conditional');self.assertEqual(z['bandwidth'],1.);np.testing.assert_allclose(z['recency'],z['probability'])
    def test_draw_reproducible_and_common_uniform(self):
        at=pd.Timestamp('2026-01-05 14:25:30Z');m=np.arange(130);w=np.ones(130)/130;a,seed=draw_blocks(m,w,w,at,CLOCKS[0],10,2048);b,_=draw_blocks(m,w,w,at,CLOCKS[0],10,2048);np.testing.assert_array_equal(a,b);self.assertEqual(seed,1062+at.date().toordinal()*16)
    def test_bank_future_and_release_excluded(self):
        at=pd.Timestamp('2026-01-05 14:25:30Z');b={'blocks':np.array([[0,1],[1,2],[2,3]]),'release_ns':np.array([pd.Timestamp('2026-01-01 22:00Z').value,pd.Timestamp('2026-01-02 22:00Z').value,pd.Timestamp('2026-01-05 05:00Z').value,pd.Timestamp('2026-01-06 22:00Z').value]),'date_ns':pd.to_datetime(['2026-01-01','2026-01-02','2026-01-03','2026-01-06']).asi8,'states':np.ones((4,8,4)),'quotes':np.ones((4,8,2))};b['successor_states']=np.ones((3,3));b['successor_at_ns']=np.array([pd.Timestamp('2026-01-02 15:00Z').value]*3);np.testing.assert_array_equal(source_pool(b,at,CLOCKS[0]),[0]);b['states'][0,0,0]=np.nan;self.assertEqual(len(source_pool(b,at,CLOCKS[0])),0)
    def test_calendar_entry_latency_and_deadline(self):
        ds=pd.bdate_range('2026-01-05','2026-01-30');cal={d:{'open_at':pd.Timestamp(str(d.date())+' 09:30',tz=core.ET).tz_convert('UTC'),'close_at':pd.Timestamp(str(d.date())+' 16:00',tz=core.ET).tz_convert('UTC')} for d in ds};a=horizon(cal,pd.Timestamp('2026-01-05 09:25:30',tz=core.ET));self.assertEqual(a['first_offset'],0);self.assertEqual(a['dates'][-1],pd.Timestamp('2026-01-16'));b=horizon(cal,pd.Timestamp('2026-01-05 10:05:30',tz=core.ET));self.assertEqual(b['first_offset'],8)
    def test_serialized_calendar_in_independent_audit(self):
        from audit_paths import check_forecast
        at=pd.Timestamp('2026-01-05 14:25:30Z');days=pd.bdate_range('2026-01-05','2026-01-30');cal={d:{'open_at':str(pd.Timestamp(str(d.date())+' 09:30',tz=core.ET).tz_convert('UTC')),'close_at':str(pd.Timestamp(str(d.date())+' 16:00',tz=core.ET).tz_convert('UTC'))} for d in days};end=at-pd.Timedelta(seconds=30)
        frame=pd.DataFrame([{'start_at':end-pd.Timedelta(minutes=5),'available_at':end+pd.Timedelta(seconds=1),'close':100.,'econ_close':100.}],index=[end]);market=SimpleNamespace(calendar=cal,bars={s:frame for s in core.ETFS});bank={'blocks':np.empty((0,2),int),'release_ns':np.empty(0,dtype='int64'),'date_ns':np.empty(0,dtype='int64'),'quotes':np.empty((0,8,2)),'states':np.empty((0,8,4)),'successor_states':np.empty((0,3)),'successor_at_ns':np.empty(0,dtype='int64')};panel=pd.DataFrame([{'clock':CLOCKS[0],'feature_ready':True,**{f:1. for f in FIELDS}}],index=[at]);model,info,raw=forecast(bank,market,panel,at,'conditional');self.assertEqual(check_forecast(bank,market,panel,at,'conditional',info,model),(None,0))
    def test_future_quote_rejected(self):
        at=pd.Timestamp('2026-01-05 14:25:30Z');end=at-pd.Timedelta(seconds=30);f=pd.DataFrame([{'available_at':at+pd.Timedelta(seconds=1),'start_at':end-pd.Timedelta(minutes=5),'close':100,'econ_close':100}],index=[end]);m=SimpleNamespace(bars={s:f for s in core.ETFS});self.assertEqual(quote(m,at),(None,None))
    def test_no_month_specific_gate_change(self):
        p=read(SOURCE/'protocol.json');old=read(core.ROOT/'research/soxl_soxs_14d5pct_v59/protocol.json')
        for f in ['acceptance','development','cases','capital','contract']:self.assertEqual(p[f],old[f])
    def test_expired_calibration_abstains(self):
        at=pd.Timestamp('2026-01-05 14:25:30Z');info={k:{'ready':True,'point':[.03,-.01],'lower':[.01,-.03]} for k in VARIANTS};daily={k:{'feedback':{'expired_unresolved_prefix':True}} for k in VARIANTS};cash={'ready':True,'action':'CASH','allocation':0.};out=events(at,pd.Timestamp('2026-01-05'),CLOCKS[0],info,daily,cash,cash)
        for n in ['conditional_band','conditional_point','unconditional_band']:self.assertEqual(out[n]['action'],'CASH')

if __name__=='__main__':unittest.main()
