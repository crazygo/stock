"""Counterexamples for conservative risk exits and a sealed regression boundary."""
import unittest
from unittest.mock import patch
import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd
from risk import core, first_exit, risk_label
from risk_ledger import replay
from run_study import choose, score
from test_contract import market_fixture
import analyze as legacy_analyze
from analyze_risk import analyze as guarded_analyze


def path(rows):
    starts=pd.date_range('2026-01-02T14:30:00Z',periods=len(rows),freq='5min')
    return pd.DataFrame([{'start_at':start,'available_at':start+pd.Timedelta(minutes=5,seconds=1),'econ_open':o,'econ_high':h,'econ_low':l,'share_factor':1.0} for start,(o,h,l) in zip(starts,rows)])


class RiskTests(unittest.TestCase):
    def test_stop_gap_is_not_filled_at_unavailable_stop_price(self):
        x=first_exit(path([(90,96,89)]),100,105,.03,None,.0005,.0005)
        self.assertAlmostEqual(x['fill'],90*.9995)
        self.assertTrue(x['gap_below_stop'])

    def test_both_barriers_uses_adverse_ordering(self):
        x=first_exit(path([(100,106,96)]),100,105,.03,None,.0005,.0005)
        self.assertEqual(x['reason'],'risk_stop')
        self.assertTrue(x['same_bar_both_barriers'])

    def test_protection_cannot_use_same_bar_high_before_low(self):
        x=first_exit(path([(100,103,99)]),100,105,.06,.02,.0005,.0005)
        self.assertIsNone(x)
        self.assertIsNone(first_exit(path([(100,103,99),(101,102,100)]),100,105,.06,.02,.0005,.0005))
        x=first_exit(path([(100,103,99),(101,102,100),(101,102,100)]),100,105,.06,.02,.0005,.0005)
        self.assertEqual(x['offset'],2)
        self.assertEqual(x['reason'],'profit_protection')
        self.assertGreaterEqual(x['fill']*.9995/(100*1.0005)-1,-1e-12)

    def test_protected_trade_can_still_lose_on_a_gap(self):
        x=first_exit(path([(100,103,99),(101,102,100),(98,100,97)]),100,105,.06,.02,.0005,.0005)
        self.assertLess(x['fill']*.9995/(100*1.0005)-1,0)

    def test_post_exit_future_mutation_cannot_change_first_stop(self):
        a=path([(100,101,96),(97,98,90)])
        x=first_exit(a,100,105,.03,None,.0005,.0005)
        a.loc[1,['econ_open','econ_high','econ_low']]=[1000,2000,500]
        self.assertEqual(x,first_exit(a,100,105,.03,None,.0005,.0005))

    def test_size_reserve_and_loss_cooldown_keep_the_cash_ledger(self):
        m=market_fixture()
        dates=m.sessions[(m.sessions>=pd.Timestamp('2026-05-04')) & (m.sessions<=pd.Timestamp('2026-05-08'))]
        ats=pd.DatetimeIndex([pd.Timestamp(str(d.date())+' 09:25:30',tz=core.ET).tz_convert('UTC') for d in dates])
        frame=m.bars['SOXL']
        stop_bar=pd.Timestamp('2026-05-04 09:35',tz=core.ET).tz_convert('UTC')
        frame.loc[stop_bar,['low','econ_low']]=90
        table=pd.DataFrame({'date':dates,'clock':'09:25:30'},index=ats)
        rule={'id':'stop3','stop_fraction':.03,'breakeven_trigger':None}
        labels=[risk_label(m,'SOXL',at,rule) for at in ats]
        result=replay(m,table,labels,{at:'SOXL' for at in ats},'2026-05-04','2026-05-08',allocation=.5,cooldown_sessions=3)
        eq,trades,decisions,_=result
        self.assertAlmostEqual(eq.cash_settled.iloc[0],.5)
        self.assertEqual(decisions.decision.tolist(),['buy_SOXL','loss_cooldown','loss_cooldown','loss_cooldown','buy_SOXL'])
        self.assertAlmostEqual(eq.equity.iloc[0],1+.5*trades.net_return.iloc[0])
        self.assertEqual(trades.capital.iloc[0],.5)

    def test_monthly_loss_fails_even_when_mean_passes(self):
        dates=pd.to_datetime(['2026-05-29','2026-06-30','2026-07-31','2026-08-31','2026-09-30'])
        returns=np.array([.5,0,.5,0,-.1])
        eq=pd.DataFrame({'equity':np.cumprod(1+returns),'mark_complete':True,'symbol':'CASH'},index=dates)
        s=score((eq,pd.DataFrame(),None,None))
        self.assertTrue(s['arithmetic_gate_from_v2'])
        self.assertFalse(s['primary_pass'])
        self.assertEqual([m['month'] for m in s['negative_months']],['2026-09'])

    def test_cash_nonnegative_is_not_return_success(self):
        dates=pd.date_range('2026-05-01','2026-09-30',freq='B')
        eq=pd.DataFrame({'equity':1.0,'mark_complete':True,'symbol':'CASH'},index=dates)
        s=score((eq,pd.DataFrame(),None,None))
        self.assertTrue(s['all_calendar_months_nonnegative'])
        self.assertFalse(s['primary_pass'])

    def test_legacy_arithmetic_pass_cannot_bypass_negative_month_gate(self):
        m=market_fixture()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'results.json').write_text(json.dumps({'trend20_premarket':{'primary_pass':True,'monthly':[{'month':'2026-09','return':-.2}]}}))
            with patch.object(core,'load_market',return_value=m):
                answer=legacy_analyze.analyze(p,p,'2026-05-04T09:25:30-04:00')
            self.assertEqual(answer['action'],'NO_ACTION')
            self.assertEqual(answer['reason'],'negative_months_fail_current_requirement')

    def test_failed_current_policy_abstains_without_quote_or_order(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'results.json').write_text(json.dumps({'acceptance':{'new_buy_eligible':False},'selected_policy':'failed','frozen_candidate':{'negative_months':[{'month':'2026-09','return':-.04}]}}))
            with patch.object(core,'load_market',side_effect=AssertionError('No quote needed for a rejected policy')):
                answer=guarded_analyze(p,'2026-05-04T08:57:20-04:00')
            self.assertEqual(answer['action'],'NO_ACTION')
            self.assertFalse(answer['broker_order_submitted'])

    def test_existing_position_expiry_survives_new_buy_rejection(self):
        m=market_fixture()
        position={'symbol':'SOXL','entry_at':'2026-05-04T09:30:00-04:00','entry_fill_raw':100.05,'quantity':10}
        with patch.object(core,'load_market',return_value=m):
            answer=guarded_analyze(Path('/nonexistent'), '2026-05-15T16:00:00-04:00',position=position)
        self.assertEqual(answer['exit_action'],'FORCE_SELL_AT_EXPIRY')
        self.assertTrue(answer['force_sell_intent']['only_if_position_still_open'])

    def test_selector_rejects_regression_period_input(self):
        with self.assertRaises(ValueError):
            choose([{'selection_end':'2026-05-31'}])
        a={'id':'a','selection_end':'2026-04-30','score':{'minimum_month_return':-.01,'all_months_geometric_return':.08,'close_max_drawdown':-.2}}
        b={'id':'b','selection_end':'2026-04-30','score':{'minimum_month_return':0,'all_months_geometric_return':.02,'close_max_drawdown':-.05}}
        self.assertEqual(choose([a,b])['id'],'b')


if __name__=='__main__':
    unittest.main(verbosity=2)
