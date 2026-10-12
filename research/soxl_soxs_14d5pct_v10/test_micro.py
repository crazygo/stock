"""Early-observation, active-order latency and cost-path regression tests."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from micro_manager import (core, context, state_at, delayed_sale, target, apply_exit,
    simple_exit, MANAGER_COLUMNS, ENTRY_COLUMNS, entry_state, prepare_reviews)
from continuation import OWN, SIGNED, UNSIGNED


def fixture():
    day = pd.Timestamp('2026-05-01'); sessions = pd.bdate_range('2026-03-01','2026-05-20')
    opening = pd.Timestamp('2026-05-01 09:30',tz=core.ET).tz_convert('UTC')
    rows = []
    for i,(minute,price) in enumerate([(-10,100.),(0,101.),(5,99.),(10,98.),(15,500.)]):
        start = opening+pd.Timedelta(minutes=minute); end = start+pd.Timedelta(minutes=5)
        rows.append({'date':day,'start_at':start,'end_at':end,'available_at':end+pd.Timedelta(seconds=1),
            'open':price,'high':price+1,'low':price-1,'close':price-.5,
            'econ_open':price,'econ_high':price+1,'econ_low':price-1,'econ_close':price-.5,
            'volume':100.,'econ_volume':100.,'share_factor':1.,
            'session_type':'pre_market' if minute<0 else 'regular'})
    f = pd.DataFrame(rows).set_index('end_at',drop=False)
    daily = pd.DataFrame({'complete':True,'volume':1000.,'high':9000.,
        'available_at':[pd.Timestamp(str(d.date())+' 16:00',tz=core.ET).tz_convert('UTC')+pd.Timedelta(seconds=1) for d in sessions]},index=sessions)
    market = SimpleNamespace(bars={s:f.copy() for s in core.SYMBOLS},daily={s:daily.copy() for s in core.SYMBOLS},
        sessions=sessions,calendar={day:{'open_at':opening,'close_at':opening+pd.Timedelta(minutes=390)}})
    row = pd.Series({'date':day,**{s+'_'+k:.2 for s in core.ETFS for k in OWN},**{k:.01 for k in SIGNED+UNSIGNED}})
    label = {'symbol':'SOXL','entry_date':day,'entry_at':opening,'decision_at':opening-pd.Timedelta(minutes=4,seconds=30),
        'entry_fill_economic':100.,'limit_economic':105.,'expiry_at':opening+pd.Timedelta(days=13,hours=6,minutes=30),
        'expiry_date':day+pd.Timedelta(days=13),'exit_at':opening+pd.Timedelta(minutes=30),'exit_bar_start':opening+pd.Timedelta(minutes=25),
        'exit_fill_economic':105.,'fee_each_side':.0005,'net_return':.049,'dividends':[],
        'label_available_at':opening+pd.Timedelta(days=13,hours=6,minutes=30,seconds=1),'complete_window':True}
    return market,row,label,opening+pd.Timedelta(minutes=5,seconds=30)


class MicroTests(unittest.TestCase):
    def test_first_bar_state_and_future_mutation(self):
        market,row,label,at = fixture();cx,days = context(market)
        before,audit = state_at(market,cx,days,row,label,at)
        self.assertEqual(audit['observed_peak'],102.)
        self.assertAlmostEqual(before['micro_own_bars'],1/78)
        self.assertAlmostEqual(before['micro_own_recent15'],100.5/101-1)
        self.assertAlmostEqual(before['own_relative_volume'],.2)
        self.assertAlmostEqual(before['micro_own_relative_volume'],.1)
        self.assertEqual(len(set(MANAGER_COLUMNS)),len(MANAGER_COLUMNS))
        for f in market.bars.values():
            for c in ['open','high','low','close','econ_open','econ_high','econ_low','econ_close']:
                f.loc[f.available_at>at,c] *= 100
        cx2,days2 = context(market)
        self.assertEqual(before,state_at(market,cx2,days2,row,label,at)[0])

    def test_earlier_barrier_cancels_pending_market_exit(self):
        market,_,label,at = fixture();_,days = context(market)
        label.update(exit_at=at.floor('5min')+pd.Timedelta(minutes=5),exit_bar_start=at.floor('5min'),
                     exit_fill_economic=93.9,reason='risk_stop',net_return=-.062)
        sale = delayed_sale(days,label,at,1)
        self.assertTrue(sale['barrier_during_latency'])
        self.assertEqual(sale['exit_at'],sale['scheduled_open'])
        self.assertEqual(target(label,sale,label['entry_date']),0.)
        self.assertEqual(apply_exit(label,sale,at),label)

    def test_market_open_precedes_its_bar_high_low(self):
        market,_,label,at = fixture();_,days = context(market)
        label.update(exit_at=at.floor('5min')+pd.Timedelta(minutes=10),exit_bar_start=at.floor('5min')+pd.Timedelta(minutes=5))
        sale = delayed_sale(days,label,at,1)
        self.assertFalse(sale['barrier_during_latency'])
        self.assertGreater(sale['scheduled_open'],at+pd.Timedelta(seconds=30))
        self.assertAlmostEqual(sale['exit_fill_economic'],98*.9995)
        exited = apply_exit(label,sale,at)
        self.assertEqual(exited['exit_at'],sale['scheduled_open'])
        self.assertEqual(exited['label_available_at'],label['label_available_at'])

    def test_missing_execution_does_not_release_capital(self):
        market,_,label,at = fixture();_,days = context(market)
        days['SOXL'][label['entry_date']] = days['SOXL'][label['entry_date']].iloc[:2]
        self.assertIsNone(delayed_sale(days,label,at,1))
        self.assertEqual(apply_exit(label,None,at),label)

    def test_delayed_price_not_prediction_input(self):
        market,row,label,at = fixture();cx,days = context(market)
        state,_ = state_at(market,cx,days,row,label,at);sale = delayed_sale(days,label,at,1)
        f = days['SOXL'][label['entry_date']]
        f.loc[f.start_at==sale['scheduled_open'],'open'] = 104.
        self.assertEqual(state,state_at(market,cx,days,row,label,at)[0])
        self.assertNotEqual(sale['sale_net'],delayed_sale(days,label,at,1)['sale_net'])

    def test_inverse_sector_fields_and_frozen_premarket_features(self):
        market,row,label,at = fixture();cx,days = context(market)
        a,_ = state_at(market,cx,days,row,label,at)
        b,_ = state_at(market,cx,days,row,{**label,'symbol':'SOXS'},at)
        self.assertAlmostEqual(a['micro_sector_session_return'],-b['micro_sector_session_return'])
        self.assertAlmostEqual(a['micro_sector_range_position']+b['micro_sector_range_position'],1.)
        self.assertEqual(a['micro_own_session_return'],b['micro_own_session_return'])
        self.assertEqual({k:a[k] for k in ENTRY_COLUMNS},entry_state(row,'SOXL'))

    def test_missing_completed_sector_bar_rejects_state(self):
        market,row,label,at = fixture();cx,days = context(market)
        cx['SOXX'].loc[at.floor('5min'),'available_at'] = at+pd.Timedelta(seconds=1)
        self.assertIsNone(state_at(market,cx,days,row,label,at)[0])

    def test_current_day_review_and_structural_control(self):
        market,row,label,at = fixture();prepared = context(market)
        features = pd.DataFrame([row],index=[label['decision_at']])
        rows = prepare_reviews(market,features,{'x':label},prepared,{'review_clocks_et':['09:25:30','09:35:30']})
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['at'],at)
        state = rows[0]['state'].copy()
        for c in ['micro_sector_session_return','micro_sector_vwap_distance','micro_sector_recent15','micro_qqq_session_return','micro_qqq_vwap_distance']:state[c]=-.01
        self.assertTrue(simple_exit(state));state['regular_ready']=0.;self.assertFalse(simple_exit(state))


if __name__=='__main__': unittest.main()
