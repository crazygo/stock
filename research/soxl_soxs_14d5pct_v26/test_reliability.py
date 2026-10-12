import unittest
import numpy as np
import pandas as pd
from reliability import *
from planning import COLUMNS


class ReliabilityTests(unittest.TestCase):
    def fixture(self,n=145):
        rng=np.random.default_rng(26);days=pd.bdate_range('2025-01-01',periods=n)
        at=(days.tz_localize(core.ET)+pd.Timedelta(hours=9,minutes=25,seconds=30)).tz_convert('UTC')
        panel=pd.DataFrame(rng.normal(size=(n,len(COLUMNS))),index=at,columns=COLUMNS)
        panel['date']=days;panel['feature_ready']=True
        panel['C']=rng.normal(-.01,.03,n);panel['D']=rng.normal(0,.04,n)
        panel['terminal_release']=at+pd.Timedelta(days=3);panel['release']=at+pd.Timedelta(days=14)
        history={t:rng.normal(0,.02,2) for t in at}
        return panel,history

    def test_future_truth_and_forecasts_do_not_change_estimate(self):
        p,h=self.fixture();at=p.index[130];past={t:v for t,v in h.items() if t<at};ids,_=rows(p,past,at)
        a=estimate(p,past,at,h[at],ids,'context');q=p.copy();q.loc[q.index>=at,['C','D']]=999
        shifted={t:(np.array([999.,999.]) if t>at else v) for t,v in h.items()}
        ids2,_=rows(q,shifted,at);self.assertEqual(ids,ids2)
        b=estimate(q,shifted,at,h[at],ids2,'context');self.assertEqual(a,b)

    def test_pending_earlier_parent_is_not_skipped(self):
        p,h=self.fixture();p.loc[p.index[30],'terminal_release']=pd.NaT
        ids,feedback=rows(p,h,p.index[-1]);self.assertEqual(ids,list(p.index[:30]))
        self.assertTrue(feedback['expired_unresolved_prefix'])

    def test_exact_cutoff_is_excluded(self):
        p,h=self.fixture();at=p.index[100];p.loc[p.index[70],'terminal_release']=day_cutoff(at)
        ids,_=rows(p,h,at);self.assertEqual(ids,list(p.index[:70]))

    def test_bootstrap_preserves_paired_date_and_block_structure(self):
        at=pd.Timestamp('2026-02-02 14:25:30Z');counts,starts,seed=block_counts(63,at,10)
        self.assertEqual(counts.shape,(199,63));np.testing.assert_array_equal(counts.sum(axis=1),np.full(199,63))
        for i in range(199):
            expanded=[(int(s)+j)%63 for s in starts[i] for j in range(10)][:63]
            np.testing.assert_array_equal(counts[i],np.bincount(expanded,minlength=63))
        np.testing.assert_array_equal(counts,block_counts(63,at,10)[0])

    def test_zero_count_rows_cannot_change_fit(self):
        rng=np.random.default_rng(1);X=rng.normal(size=(20,3));Y=rng.normal(size=(20,2));w=np.r_[np.ones(15),np.zeros(5)]
        a=fit_error(X,Y,w);X[15:]=9e9;Y[15:]=9e9;b=fit_error(X,Y,w)
        for k in a:np.testing.assert_allclose(a[k],b[k],atol=1e-12)

    def test_joint_band_never_exceeds_point(self):
        p,h=self.fixture();at=p.index[-1];ids,_=rows(p,h,at);info=estimate(p,h,at,h[at],ids,'context')
        self.assertGreaterEqual(info['radius'],0);self.assertTrue(np.all(np.array(info['lower'])<=info['point']))
        sample=np.array(info['bootstrap_predictions']);point=np.array(info['point'])
        margin=max(0.,sorted(np.max(sample-point,axis=1))[int(np.ceil(.9*198))])
        self.assertAlmostEqual(margin,info['radius'])

    def test_cash_ties_and_expired_feedback(self):
        m={'ready':True,'lower':[0.,-.01],'point':[.03,.02]};fb={'expired_unresolved_prefix':False}
        self.assertEqual(decide(m,fb)['action'],'CASH');self.assertEqual(decide(m,fb,False)['action'],'SOXL')
        m['lower']=[.01,.01];self.assertEqual(decide(m,fb)['action'],'SOXL')
        fb['expired_unresolved_prefix']=True;self.assertEqual(decide(m,fb)['action'],'CASH')

    def test_minimum_is_dates_not_bootstrap_replicates(self):
        p,h=self.fixture();at=p.index[-1];info=estimate(p,h,at,h[at],list(p.index[:59]),'context')
        self.assertFalse(info['ready'])

    def test_intercept_only_preserves_weighted_paired_mean(self):
        Y=np.array([[.1,-.2],[.3,.4],[-.3,.2]]);w=np.array([1.,2.,3.])
        model=fit_error(np.empty((3,0)),Y,w)
        np.testing.assert_allclose(predict_error(model,[]),np.average(Y,axis=0,weights=w))


if __name__=='__main__':unittest.main()
