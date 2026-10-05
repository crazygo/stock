import unittest,json,sys
from pathlib import Path
import pandas as pd,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from data import CACHE,FEATURES,sessions,peak_and_stage
from model import wilson,signals
from quality import quality_from_facts

class CausalEvidenceTests(unittest.TestCase):
    def test_all_training_and_calibration_label_maturities_precede_next_block(self):
        p=Path(__file__).parent/'backtest_v1/split_audit.json'
        audits=json.loads(p.read_text())
        self.assertEqual(len(audits),8)
        for a in audits:
            if a['status']!='development_calibrated':continue
            self.assertLess(pd.Timestamp(a['train_latest_label_maturity']),pd.Timestamp(a['calibration_start']))
            self.assertLess(pd.Timestamp(a['cal_latest_label_maturity']),pd.Timestamp(a['evaluation_start']))
            self.assertGreater(a['calibration_positives'],19)

    def test_stored_forward_labels_recompute_raw_actual_future_path(self):
        panel=pd.read_parquet(CACHE/'panel.parquet');pred=pd.read_parquet(CACHE/'forward_predictions.parquet')
        # Every emitted >80% example plus random mature positives/negatives; include failures and unknowns.
        tested=pd.concat([pred[pred.probability>.8],pred[pred.label.notna()].sample(100,random_state=20261005)]).drop_duplicates(['ticker','date','horizon'])
        lookup={t:q.set_index('date') for t,q in panel.groupby('ticker')}
        for r in tested.itertuples():
            expiry=r.date+pd.Timedelta(days=r.horizon)
            if r.label_status=='immature':self.assertGreater(expiry,panel.date.max());continue
            if pd.isna(r.label):continue
            idx=sessions((r.date+pd.Timedelta(days=1)).date().isoformat(),expiry.date().isoformat())
            actual=lookup[r.ticker].reindex(idx)
            self.assertFalse(actual.high.isna().any())
            y=int((actual.high>=r.reference_close*1.002*2).any())
            self.assertEqual(y,int(r.label))

    def test_no_overlapping_high_confidence_signals(self):
        p=Path(__file__).parent/'backtest_v1/backtest_signals.csv';q=pd.read_csv(p,parse_dates=['date'])
        for _,g in q.groupby(['ticker','horizon','algorithm']):
            self.assertTrue((g.sort_values('date').date.diff().dropna().dt.days>=60).all())
        self.assertTrue((q.probability>.8).all())

    def test_peak_confirmation_does_not_read_future(self):
        idx=sessions('2026-01-02','2026-02-02')[:15]
        q=pd.DataFrame({'ticker':'X','date':idx,'high':[10]*5+[20]+[10]*9,'low':[9]*15,'close':[9.5]*15})
        at_unconfirmed=peak_and_stage(q,'X',idx[8].date().isoformat())
        self.assertIsNone(at_unconfirmed['previous_peak_date'])
        at_confirmed=peak_and_stage(q,'X',idx[12].date().isoformat())
        self.assertEqual(at_confirmed['previous_peak_date'],idx[5].date().isoformat())

    def test_financials_exclude_later_filings(self):
        j={'entityName':'X','facts':{'us-gaap':{'Revenues':{'units':{'USD':[
            {'start':'2026-01-01','end':'2026-03-31','filed':'2026-05-01','val':100,'form':'10-Q','accn':'old'},
            {'start':'2026-01-01','end':'2026-03-31','filed':'2026-08-01','val':900,'form':'10-Q','accn':'future'}]}}}}}
        x=quality_from_facts(j,'2026-06-01')
        self.assertEqual(x['source_facts']['revenue']['val'],100)
        self.assertEqual(x['source_facts']['revenue']['accn'],'old')

    def test_report_usable_probabilities_obey_nested_horizons(self):
        ptr=json.loads((Path(__file__).parent/'latest_run.json').read_text())
        q=pd.read_csv(Path(ptr['path'])/'all_stocks.csv')
        usable=q[q.p30.notna()&q.p60.notna()]
        self.assertTrue((usable.p30<=usable.p60).all())
        rejected=q[q.score_status=='incoherent_horizon_predictions_require_joint_calibration']
        self.assertTrue(rejected.p30.isna().all() and rejected.p60.isna().all())
        self.assertTrue((rejected.raw_model_p30>rejected.raw_model_p60).all())

    def test_small_perfect_sample_is_not_80pct_evidence(self):
        self.assertLess(wilson(4,4)[0],.8)
        self.assertEqual(wilson(0,0),[None,None])

if __name__=='__main__':unittest.main()
