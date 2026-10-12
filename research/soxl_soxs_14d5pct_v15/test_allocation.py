import unittest
import numpy as np
import pandas as pd
from allocation import core, COLUMNS, FRACTIONS, utility, choose, weights, prior_pool, weighted_tail
from allocation_ledger import replay
from risk import risk_label
from test_contract import market_fixture


class AllocationTests(unittest.TestCase):
    def test_tail_risk_changes_fraction_without_changing_direction(self):
        r=np.array([.05]*9+[-.30]);w=np.ones(10)/10
        log=utility(r,FRACTIONS,1)@w;power=utility(r,FRACTIONS,5)@w
        self.assertEqual(FRACTIONS[np.argmax(log)],1.)
        self.assertEqual(FRACTIONS[np.argmax(power)],.25)
        self.assertEqual(FRACTIONS[np.argmax(utility(-np.ones(10)*.05,FRACTIONS,5)@w)],0.)
        self.assertTrue(np.all(utility(r,[0.],5)==0))
        with self.assertRaises(ValueError):utility([-1.],[1.],5)

    def test_tail_fractional_mass_keeps_worst_observation(self):
        self.assertAlmostEqual(weighted_tail(np.array([.05,-.5,-.1]),np.array([.9,.02,.08])),(-.5*.02-.1*.03)/.05)

    def test_future_and_exact_release_are_excluded(self):
        cutoff=pd.Timestamp('2026-01-02',tz=core.ET).tz_convert('UTC')
        ats=pd.date_range('2025-12-01',periods=4,tz='UTC')
        frame=pd.DataFrame({'at':ats,'symbol':'SOXL','date':ats.tz_localize(None).normalize(),
            'release':[cutoff-pd.Timedelta(seconds=1),cutoff,cutoff+pd.Timedelta(seconds=1),pd.NaT],
            'net_return':[.05,99.,99.,99.]}).set_index(['at','symbol'])
        self.assertEqual(list(prior_pool(frame,cutoff,'SOXL').index),[ats[0]])

    def test_distance_weights_do_not_use_outcomes_or_query_in_scaler(self):
        idx=pd.date_range('2025-01-01',periods=60,tz='UTC');pool=pd.DataFrame(index=idx)
        for j,c in enumerate(COLUMNS):pool[c]=np.arange(60)+j
        pool['net_return']=.02;pool['date']=idx.tz_localize(None);current=pool.iloc[0].copy()
        w,info=weights(pool,current,'local');pool.net_return=-.9
        z,other=weights(pool,current,'local');np.testing.assert_array_equal(w,z)
        current['score']=1e6;_,changed=weights(pool,current,'local')
        self.assertEqual(info['center'],changed['center']);self.assertAlmostEqual(w.sum(),1.)
        self.assertLess(info['effective_dates'],60)
        self.assertEqual(choose(pool.iloc[:59],current,'local',5,FRACTIONS)['reason'],'insufficient_mature_dates')

    def test_dynamic_cash_reserve_settlement_and_single_position(self):
        m=market_fixture();dates=m.sessions[(m.sessions>='2026-05-04')&(m.sessions<='2026-05-06')]
        ats=pd.DatetimeIndex([d.tz_localize(core.ET)+pd.Timedelta(hours=9,minutes=25,seconds=30) for d in dates]).tz_convert('UTC')
        m.bars['SOXL'].loc[pd.Timestamp('2026-05-04 09:35',tz=core.ET).tz_convert('UTC'),['low','econ_low']]=80.
        table=pd.DataFrame({'date':dates,'clock':'09:25:30'},index=ats)
        labels=[risk_label(m,'SOXL',at,{'id':'stop6','stop_fraction':.06,'breakeven_trigger':None}) for at in ats]
        eq,trades,decisions,_=replay(m,table,labels,dict.fromkeys(ats,'SOXL'),'2026-05-04','2026-05-06',allocations=dict(zip(ats,[.25,.5,1.])))
        self.assertAlmostEqual(eq.cash_settled.iloc[0],.75);self.assertGreater(eq.cash_unsettled.iloc[0],0.)
        self.assertAlmostEqual(eq.equity.iloc[0],1+.25*trades.net_return.iloc[0])
        self.assertAlmostEqual(trades.capital.iloc[1],eq.equity.iloc[0]*.5)
        self.assertEqual(decisions.decision.tolist(),['buy_SOXL','buy_SOXL','position_held'])
        self.assertGreater(eq.cash_settled.iloc[-1],0.)
        self.assertEqual(trades.reason_at_evaluation_end.iloc[-1],'pending')
        with self.assertRaises(ValueError):replay(m,table,labels,dict.fromkeys(ats,'SOXL'),'2026-05-04','2026-05-06',allocations=dict.fromkeys(ats,0.))


if __name__=='__main__':unittest.main()
