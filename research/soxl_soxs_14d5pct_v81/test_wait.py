import unittest
from types import SimpleNamespace
from wait_learning import *
import reference_wait as ref


def sample():
    days=pd.to_datetime(['2025-01-02','2025-01-03','2025-01-06','2025-01-07']);calendar={};rows=[];index=[];records={}
    for i,d in enumerate(days):
        calendar[d]={'close_at':pd.Timestamp(str(d.date())+' 16:00',tz=core.ET).tz_convert('UTC')}
        for clock in CLOCKS:
            t=pd.Timestamp(str(d.date())+' '+clock,tz=core.ET).tz_convert('UTC');index.append(t)
            rows.append({'date':d,'clock':clock,'feature_ready':True,'terminal_release':pd.Timestamp(str((d+pd.Timedelta(days=13)).date())+' 16:00:01',tz=core.ET).tz_convert('UTC'),
                'L_log':.04,'S_log':-.07,**{f:float(j+1) for j,f in enumerate(FIELDS)}})
            e={'ready':True,'action':['SOXL','SOXS','CASH','SOXL'][i],'top_symbol':'SOXL','values':[.02,.01],'allocation':1.}
            records[t]={'events':{k:dict(e) for k in ['recent_ratio','lagged_ratio','uniform_top','source_blend']}}
    return pd.DataFrame(rows,index=index),SimpleNamespace(sessions=days,calendar=calendar),records


class WaitTests(unittest.TestCase):
    def test_official_next_day_losses_and_cash_no_oracle(self):
        panel,market,rec=sample();bank=target_registry(panel,market,rec)
        self.assertEqual(bank[1]['next_date'],pd.Timestamp('2025-01-06'))
        X,Y,c,w,meta=arrays(panel,bank[:2],rec)
        np.testing.assert_allclose(Y[:8,1],-.07);np.testing.assert_allclose(Y[8:,1],0.)
        np.testing.assert_allclose(Y[:,0],.04);self.assertEqual(len(meta['dates']),2)
        self.assertEqual(bank,ref.bank(panel,market,rec,CLOCKS))

    def test_missing_future_not_zero_and_cash_still_waits_full_window(self):
        panel,market,rec=sample();bank=target_registry(panel,market,rec)
        self.assertEqual(bank[1]['release'],panel.loc[bank[1]['next_at'],'terminal_release'])
        rec[bank[0]['next_at']]['events']['recent_ratio']['ready']=False
        new=target_registry(panel,market,rec);self.assertIsNone(new[0]['release'])
        at=pd.Timestamp('2025-02-01 09:25:30',tz=core.ET).tz_convert('UTC');chosen,fb=admitted(new,at)
        self.assertEqual(chosen,[]);self.assertTrue(fb['expired_unresolved_prefix'])

    def test_exact_cutoff_and_unknown_prefix_cannot_skip(self):
        panel,market,rec=sample();bank=target_registry(panel,market,rec)
        cutoff=pd.Timestamp('2025-01-20 00:00',tz=core.ET).tz_convert('UTC');bank[1]['release']=cutoff
        chosen,fb=admitted(bank,cutoff);self.assertEqual(len(chosen),1)
        bank[1]['release']=None;bank[2]['release']=cutoff-pd.Timedelta(days=1)
        self.assertEqual(len(admitted(bank,cutoff)[0]),1)

    def test_future_outcomes_features_and_actions_do_not_change_past_model(self):
        panel,market,rec=sample();bank=target_registry(panel,market,rec);at=pd.Timestamp('2025-01-17 09:25:30',tz=core.ET).tz_convert('UTC')
        parents,_=admitted(bank,at);self.assertEqual(len(parents),1);a=arrays(panel,parents,rec)
        for t,row in panel.iterrows():
            if row.date>=pd.Timestamp('2025-01-06'):
                panel.loc[t,FIELDS]=1000;panel.loc[t,['L_log','S_log']]=[-.9,.9];rec[t]['events']['recent_ratio']['action']='SOXS'
        fresh=target_registry(panel,market,rec);new,_=admitted(fresh,at);b=arrays(panel,new,rec)
        for x,y in zip(a[:4],b[:4]):np.testing.assert_array_equal(x,y)

    def test_svd_matches_normal_equations_and_paired_difference(self):
        rng=np.random.default_rng(81);X=rng.normal(size=(80,16));Y=rng.normal(size=(80,2));c=np.arange(80)%8;w=np.linspace(.3,1,80)
        for mode in VARIANTS:
            m=fit(X,Y,c,w,mode);other,A=ref.solve(X,Y,c,w,mode)
            for field in ['center','scale','design_center','intercept','coefficients','penalty']:np.testing.assert_allclose(m[field],other[field],atol=1e-12)
            direct=fit(X,np.c_[Y[:,0]-Y[:,1],Y[:,0]-Y[:,1]],c,w,mode)
            np.testing.assert_allclose(m['coefficients'][:,0]-m['coefficients'][:,1],direct['coefficients'][:,0],atol=1e-12)
            for i in [0,7,31]:np.testing.assert_allclose(predict(m,X[i],c[i]),ref.forecast(other,X[i],c[i]),atol=1e-12)

    def test_repeated_clocks_split_weight_and_unit_change_invariance(self):
        rng=np.random.default_rng(88);X=rng.normal(size=(40,16));Y=rng.normal(size=(40,2));c=np.arange(40)%8;w=np.ones(40)
        m=fit(X,Y,c,w,'state');d=fit(np.repeat(X,2,axis=0),np.repeat(Y,2,axis=0),np.repeat(c,2),np.repeat(w/2,2),'state')
        np.testing.assert_allclose(m['coefficients'],d['coefficients'],atol=1e-12)
        X[:,2]=1;m=fit(X,Y,c,w,'state');self.assertEqual(m['scale'][2],1.)
        scales=np.arange(1,17)*10;offset=np.arange(16)*20;n=fit(X*scales+offset,Y,c,w,'state')
        np.testing.assert_allclose(predict(m,X[0],0),predict(n,X[0]*scales+offset,0),atol=1e-12)

    def test_actions_keep_direction_source_cash_and_absolute_controls(self):
        panel,market,records=sample();at=panel.index[0];row=panel.loc[at];src=records[at];X=np.ones((60,16));clocks=np.arange(60)%8;w=np.ones(60)
        m=fit(X,np.tile([-.01,-.03],(60,1)),clocks,w,'mean');daily={'ready':True,'models':{k:m for k in VARIANTS}}
        e=issue(at,row,daily,src);self.assertEqual(e['paired_state']['action'],'SOXL');self.assertEqual(e['absolute_pair']['action'],'CASH')
        src['events']['recent_ratio']['action']='CASH';self.assertEqual(issue(at,row,daily,src)['paired_state']['action'],'CASH')
        m=fit(X,np.tile([.01,.02],(60,1)),clocks,w,'mean');daily['models']={k:m for k in VARIANTS};src['events']['recent_ratio']['action']='SOXL'
        e=issue(at,row,daily,src);self.assertEqual(e['paired_state']['action'],'CASH');self.assertEqual(e['absolute_now']['action'],'SOXL')

    def test_invalid_rows_are_rejected(self):
        with self.assertRaises(ValueError):fit(np.ones((2,16)),np.array([[0,1],[np.nan,1]]),[0,1],[1,1],'state')
        with self.assertRaises(ValueError):fit(np.ones((2,16)),np.ones((2,2)),[0,1],[1,0],'state')


if __name__=='__main__':unittest.main()
