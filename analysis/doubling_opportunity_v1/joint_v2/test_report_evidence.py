import importlib.util,json,sys,tempfile,unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import pandas as pd
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import prepare_joint
import financials
spec=importlib.util.spec_from_file_location('doubling_report_test',HERE/'run.py');report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

class ReportAndCohortTests(unittest.TestCase):
    def test_financial_training_uses_joint_model_despite_parent_name_collision(self):
        parent_spec=importlib.util.spec_from_file_location('model',HERE.parent/'model.py')
        parent_model=importlib.util.module_from_spec(parent_spec);parent_spec.loader.exec_module(parent_model)
        train_spec=importlib.util.spec_from_file_location('doubling_training_test',HERE/'train_financial.py')
        training=importlib.util.module_from_spec(train_spec)
        with patch.dict(sys.modules,{'model':parent_model}):train_spec.loader.exec_module(training)
        self.assertEqual(Path(training.forward_backtest.__code__.co_filename).resolve(),(HERE/'model.py').resolve())
        self.assertIn('signal_gate',training.forward_backtest.__code__.co_varnames)

    def test_invalid_cached_identity_is_retained_missing_and_cannot_crash_cohort(self):
        with tempfile.TemporaryDirectory() as td:
            cache=Path(td);(cache/'companyfacts').mkdir()
            (cache/'companyfacts/CIK0000000001.json').write_text(json.dumps({'entityName':'Unverified','facts':{'us-gaap':{}}}))
            with patch.object(financials,'CACHE',cache):m=financials.prepare_events('2026-01-01','2026-09-29')
            self.assertEqual(m['available_issuer_count'],0);self.assertEqual(m['invalid_cached_issuer_count'],1)
            self.assertIsNone(m['issuers'][0]['path']);self.assertFalse(m['issuers'][0]['source_valid'])
    def test_latest_closed_session_obeys_actual_close_and_half_day(self):
        self.assertEqual(report.last_completed_session(datetime(2026,10,5,19,59,tzinfo=timezone.utc)),'2026-10-02')
        self.assertEqual(report.last_completed_session(datetime(2026,10,5,20,1,tzinfo=timezone.utc)),'2026-10-05')
        self.assertEqual(report.last_completed_session(datetime(2026,11,27,17,59,tzinfo=timezone.utc)),'2026-11-25')
        self.assertEqual(report.last_completed_session(datetime(2026,11,27,18,1,tzinfo=timezone.utc)),'2026-11-27')

    def test_no_download_order_subset_can_start_joint_evaluation(self):
        with patch.object(prepare_joint,'coverage_state',return_value={'fundamentals':{'terminal':False},'submissions':{'terminal':True}}):
            with self.assertRaisesRegex(RuntimeError,'terminate'):prepare_joint.prepare()

    def test_industry_features_ignore_future_days_and_ineligible_rows(self):
        with tempfile.TemporaryDirectory() as td:
            cache=Path(td);(cache/'submissions').mkdir()
            u={};rows=[]
            for i in range(7):
                cik=i+1;u[f'X{i}']={'cik':cik}
                (cache/'submissions'/f'CIK{cik:010d}.json').write_text(json.dumps({'cik':cik,'sic':'3674'}))
                for day in ['2026-09-28','2026-09-29']:
                    rows.append({'ticker':f'X{i}','date':pd.Timestamp(day),'eligible':i<6,'ret20':.2 if i<6 else 1e9,
                         'rs20':.1,'volume_ratio':1.2,'quality_status_at_event':'growth_and_operating_health_checks_pass'})
            q=pd.DataFrame(rows)
            with patch.object(prepare_joint,'CACHE',cache):
                before=prepare_joint.industry_features(q,u)
                q.loc[q.date==pd.Timestamp('2026-09-29'),'ret20']=999
                after=prepare_joint.industry_features(q,u)
            cols=['ticker','industry_ret20','industry_rs20','industry_members','joint_signal_eligible']
            a=before[before.date==pd.Timestamp('2026-09-28')][cols].reset_index(drop=True)
            b=after[after.date==pd.Timestamp('2026-09-28')][cols].reset_index(drop=True)
            pd.testing.assert_frame_equal(a,b)
            self.assertTrue((a.industry_ret20==.2).all());self.assertTrue((a.industry_members==6).all())
            self.assertFalse(bool(a[a.ticker=='X6'].joint_signal_eligible.iloc[0]))

    def test_current_quote_cannot_change_probability_anchor(self):
        pointer=json.loads((HERE/'latest_run.json').read_text());j=json.loads((Path(pointer['path'])/'report.json').read_text())
        mx=j['case_stocks'][0]
        self.assertEqual(mx['latest_quote_price'],105.93)
        self.assertEqual(mx['probability_anchor_date'],'2026-09-29')
        self.assertEqual(mx['current_close'],92.72)
        self.assertFalse(j['metadata']['quotes']['historical_probability_refresh'])
        self.assertFalse(j['metadata']['regular_hours_label_verified'])
        self.assertEqual(j['qualified_stocks'],[])

if __name__=='__main__':unittest.main()
