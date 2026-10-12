"""Accounting, causality, deadline and hourly-bar verification. No model tuning."""
from pathlib import Path
import json
import unittest
import numpy as np
import pandas as pd
import backtest as bt

def synthetic():
    days=pd.bdate_range('2026-01-02','2026-02-06')
    f=pd.DataFrame({'open':100.,'high':101.,'low':90.,'close':98.,'volume':1000.,'factor':1.},index=days)
    return {s:f.copy() for s in bt.SYMBOLS},{s:[] for s in bt.SYMBOLS},days

class ContractTests(unittest.TestCase):
    def test_missing_future_calendar_never_truncates_deadline(self):
        data,ev,days=synthetic()
        label=bt.label_one('SOXL',days[-2],data,ev,days)
        self.assertFalse(label['complete_window']);self.assertTrue(pd.isna(label['deadline']))
        self.assertEqual(label['reason'],'pending_calendar')
    def test_fourteen_calendar_days_inclusive(self):
        data,ev,days=synthetic()
        self.assertEqual(bt.deadline(pd.Timestamp('2026-01-05'),days),pd.Timestamp('2026-01-16'))
    def test_profit_target_fee_and_no_future_target(self):
        data,ev,days=synthetic();entry=days[1]
        data['SOXL'].loc[days[0],'high']=110
        label=bt.label_one('SOXL',entry,data,ev,days)
        self.assertEqual(label['reason'],'expiry')
        data['SOXL'].loc[entry,'high']=110
        label=bt.label_one('SOXL',entry,data,ev,days)
        self.assertEqual(label['exit'],entry)
        self.assertAlmostEqual(label['net_return'],label['limit']*(1-bt.FEE)/(label['entry_fill']*(1+bt.FEE))-1)
    def test_target_after_deadline_does_not_win(self):
        data,ev,days=synthetic();entry=days[0];end=bt.deadline(entry,days)
        data['SOXL'].loc[days[days>end][0],'high']=130
        self.assertEqual(bt.label_one('SOXL',entry,data,ev,days)['reason'],'expiry')
    def test_censored_not_forced_loss(self):
        data,ev,days=synthetic();label=bt.label_one('SOXL',days[-3],data,ev,pd.bdate_range(days[0],days[-1]+pd.Timedelta(days=20)))
        self.assertEqual(label['reason'],'pending');self.assertTrue(np.isnan(label['net_return']))
    def test_no_same_day_reentry_and_one_capital(self):
        data,ev,days=synthetic();data['SOXL']['high']=110
        eq,ts,_=bt.replay(data,ev,days,dict.fromkeys(days[:3],'SOXL'),days[0],days[2])
        self.assertEqual(len(ts),3)
        self.assertAlmostEqual(eq.equity.iloc[-1],np.prod(1+ts.net_return))
    def test_dividend_overnight_entitlement_and_receivable(self):
        data,ev,days=synthetic();ev['SOXL']=[{'ex':days[1],'pay':days[4],'amount':1.}]
        eq,ts,_=bt.replay(data,ev,days,{days[0]:'SOXL'},days[0],days[3])
        self.assertGreater(eq.loc[days[1],'receivables'],0)
        eq,ts,_=bt.replay(data,ev,days,{days[1]:'SOXL'},days[0],days[3])
        self.assertEqual(eq.receivables.max(),0)
    def test_future_feature_invariance(self):
        data,ev,days=synthetic();base=bt.features(data)
        for f in data.values():f.loc[days[-2]:,['open','close','high','low']]*=9
        changed=bt.features(data)
        pd.testing.assert_frame_equal(base.loc[:days[-2]],changed.loc[:days[-2]])
    def test_realized_label_equals_ledger(self):
        data,events,sessions,_=bt.load_inputs()
        trades=pd.read_csv(bt.ROOT/'results/model_trades_followup.csv',parse_dates=['entry','exit'])
        for _,t in trades.iterrows():
            if t.reason=='pending':continue
            lab=bt.label_one(t.symbol,t.entry,data,events,sessions)
            self.assertEqual(lab['exit'],t.exit)
            self.assertAlmostEqual(lab['net_return'],t.net_return,places=11)
    def test_month_boundaries_are_purged(self):
        regs=json.loads((bt.ROOT/'results/model_registration.json').read_text())
        for r in regs:
            self.assertLess(pd.Timestamp(r['train_last_deadline']),pd.Timestamp(r['selection_start']))
            self.assertLess(pd.Timestamp(r['selection_last_deadline']),pd.Period(r['month']).start_time)

def audit_hourly():
    data,events,sessions,_=bt.load_inputs();intraday={};coverage=[];checks=[]
    for s in bt.ETFS:
        d=json.loads((bt.ROOT/f'raw/{s}_ibkr_hourly_count.json').read_text())
        h=pd.DataFrame({k:d[k] for k in ['time','open','high','low','close','volume']})
        h.index=pd.DatetimeIndex(pd.to_datetime(h.pop('time'),utc=True).dt.tz_convert('America/New_York'))
        h['date']=h.index.tz_localize(None).normalize();intraday[s]=h
        for date,g in h.groupby('date'):
            if date<pd.Timestamp('2026-05-01'):continue
            daily=data[s].loc[date]
            coverage.append({'symbol':s,'date':date,'bars':len(g),'first':str(g.index[0]),'last':str(g.index[-1]),'high_ratio_difference':g.high.max()/daily.high-1,'low_ratio_difference':g.low.min()/daily.low-1,'close_ratio_difference':g.close.iloc[-1]/daily.close-1})
    for name in ['model','always_SOXL','always_SOXS','trend20']:
        trades=pd.read_csv(bt.ROOT/f'results/{name}_trades.csv',parse_dates=['entry','exit'])
        for _,t in trades.iterrows():
            if t.reason=='pending':continue
            h=intraday[t.symbol]
            path=h[(h.date>=t.entry)&(h.date<=t.exit)]
            eligible=path[path.high>=t.limit]
            first=eligible.iloc[0] if len(eligible) else None
            first_date=first['date'] if first is not None else None
            agrees=first_date==t.exit if t.reason=='target' else first_date is None
            checks.append({'policy':name,'symbol':t.symbol,'entry':t.entry,'exit':t.exit,'reason':t.reason,'limit_split_adjusted':t.limit,'hourly_first_hit_bar_start':str(eligible.index[0]) if len(eligible) else None,'agrees_with_daily':bool(agrees),'all_path_sessions_have_7_bars':bool(path.groupby('date').size().eq(7).all())})
    pd.DataFrame(coverage).to_csv(bt.ROOT/'results/hourly_coverage.csv',index=False)
    pd.DataFrame(checks).to_csv(bt.ROOT/'results/hourly_trade_audit.csv',index=False)
    return {'completed_trade_checks':len(checks),'target_or_expiry_agreements':sum(x['agrees_with_daily'] for x in checks),'model_trade_checks':sum(x['policy']=='model' for x in checks),'model_agreements':sum(x['policy']=='model' and x['agrees_with_daily'] for x in checks),'all_checked_paths_have_full_hourly_sessions':all(x['all_path_sessions_have_7_bars'] for x in checks),'disagreements':[x for x in checks if not x['agrees_with_daily']],'limitation':'OHLC target trade-through is not broker fill proof. First bar is 09:30-10:00, later bars hourly. Requested 5-minute bars are absent.'}

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    audit=audit_hourly();audit['unit_tests']={'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}
    bt.dump(bt.ROOT/'results/verification.json',audit)
    print(json.dumps(audit,indent=2,default=str))
    if not result.wasSuccessful():raise SystemExit(1)
