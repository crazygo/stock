import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).parent;PARENT=HERE.parent;sys.path.insert(0,str(HERE))
from financials import extract_facts,quarters,trailing_four,snapshot,issuer_events,join_price_features
spec=importlib.util.spec_from_file_location('doubling_joint_test_model',HERE/'model.py');joint=importlib.util.module_from_spec(spec);spec.loader.exec_module(joint)
sys.path.insert(0,str(PARENT));from data import CACHE


def raw(start,end,val,filed='2026-08-01',accn='known'):
    return {'start':start,'end':end,'val':val,'filed':filed,'accn':accn,'form':'10-Q'}

def facts(tags):
    return {'cik':1,'entityName':'Fixture','facts':{'us-gaap':{tag:{'units':{'USD':rows}} for tag,rows in tags.items()}}}

class JointAndFinancialEvidenceTests(unittest.TestCase):
    def test_real_forward_probabilities_have_consistent_events(self):
        p=pd.read_parquet(CACHE/'joint_v2_forward_predictions.parquet',columns=['p30','p60'])
        self.assertGreater(len(p),100_000)
        self.assertTrue(((p.p30>=0)&(p.p30<=p.p60)&(p.p60<=1)).all())

    def test_joint_training_maturity_precedes_calibration_and_test(self):
        ptr=json.loads((HERE/'latest_backtest.json').read_text());audits=json.loads((Path(ptr['path'])/'split_audit.json').read_text())
        self.assertEqual(len(audits),4)
        for a in audits:
            if a['status']!='development_joint_calibrated':continue
            self.assertLess(pd.Timestamp(a['train_latest_label_maturity']),pd.Timestamp(a['calibration_start']))
            self.assertLess(pd.Timestamp(a['cal_latest_label_maturity']),pd.Timestamp(a['evaluation_start']))
            self.assertGreaterEqual(min(a['class_counts']['calibrate']),20)

    def test_unknown_or_impossible_labels_are_not_failure(self):
        q=pd.DataFrame({'y30':[0,0,1,np.nan],'y60':[0,1,1,np.nan]})
        self.assertEqual(joint.event_class(q).tolist(),[0,1,2,-1])
        with self.assertRaises(ValueError):joint.event_class(pd.DataFrame({'y30':[1],'y60':[0]}))

    def test_half_year_cashflow_is_not_a_quarter(self):
        j=facts({'NetCashProvidedByUsedInOperatingActivities':[
            raw('2026-01-01','2026-03-31',10,'2026-05-01','q1'),raw('2026-01-01','2026-06-30',30,'2026-08-01','ytd')]})
        q=quarters(extract_facts(j)['operating_cashflow'],'2026-09-01')
        self.assertEqual(q[-1]['val'],20);self.assertEqual(q[-1]['start'],'2026-04-01')
        self.assertEqual(q[-1]['source_kind'],'cumulative_difference')
        self.assertEqual({r['accn'] for r in q[-1]['source_records']},{'q1','ytd'})

    def test_direct_quarter_precedes_inferred_difference(self):
        j=facts({'Revenues':[raw('2026-01-01','2026-03-31',10),raw('2026-01-01','2026-06-30',30),raw('2026-04-01','2026-06-30',19)]})
        q=quarters(extract_facts(j)['revenue'],'2026-09-01')
        self.assertEqual(q[-1]['val'],19);self.assertEqual(q[-1]['source_kind'],'direct')

    def test_future_revisions_cannot_change_previous_snapshot(self):
        j=facts({'Revenues':[raw('2026-04-01','2026-06-30',100,'2026-08-01','initial'),raw('2026-04-01','2026-06-30',999,'2026-10-01','future')]})
        x=snapshot(extract_facts(j),'2026-09-29');self.assertEqual(x['metrics']['revenue_quarter'],100)
        self.assertEqual(x['sources']['revenue_quarter']['accn'],'initial')

    def test_mismatched_flow_and_balance_periods_stay_missing(self):
        j=facts({'Revenues':[raw('2026-04-01','2026-06-30',100)],'NetIncomeLoss':[raw('2026-01-01','2026-03-31',50)],
                 'AssetsCurrent':[raw(None,'2026-06-30',100)],'LiabilitiesCurrent':[raw(None,'2026-03-31',10)]})
        x=snapshot(extract_facts(j),'2026-09-01')
        self.assertNotIn('net_margin',x['metrics']);self.assertNotIn('current_ratio',x['metrics'])
        self.assertFalse(x['quality_checks']['current_ratio_at_least_one'])

    def test_ttm_requires_four_contiguous_quarters(self):
        j=facts({'Revenues':[raw('2025-07-01','2025-09-30',1),raw('2025-10-01','2025-12-31',2),raw('2026-01-01','2026-03-31',3),raw('2026-04-01','2026-06-30',4)]})
        q=quarters(extract_facts(j)['revenue'],'2026-09-01')
        self.assertEqual(trailing_four(q,'2026-06-30')['val'],10)
        self.assertIsNone(trailing_four(q[1:],'2026-06-30'))
        q[2]={**q[2],'start':'2026-02-01'};self.assertIsNone(trailing_four(q,'2026-06-30'))

    def test_release_next_day_and_period_staleness_are_enforced(self):
        j=facts({'Revenues':[raw('2026-04-01','2026-06-30',100,'2026-08-03')]})
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'facts.json';p.write_text(json.dumps(j));e=issuer_events(p,'2026-07-01','2026-12-31')
            self.assertEqual(e.available_at.min(),pd.Timestamp('2026-08-04'))
            ep=Path(td)/'e.parquet';e.to_parquet(ep)
            price=pd.DataFrame({'ticker':['X']*3,'date':pd.to_datetime(['2026-08-03','2026-08-04','2027-02-01'])})
            q=join_price_features(price,{'X':{'cik':1}},{'issuers':[{'cik':1,'path':str(ep)}]})
            self.assertEqual(q.iloc[0].fin_statement_missing,1)
            self.assertEqual(q.iloc[1].fin_statement_missing,0)
            self.assertEqual(q.iloc[2].fin_fresh,0)
            self.assertEqual(q.iloc[2].quality_status_at_event,'stale_or_missing_financial_statement')

    def test_mxl_real_quarter_and_ttm_values_are_recomputable(self):
        j=json.loads((CACHE/'companyfacts/CIK0001288469.json').read_text());x=snapshot(extract_facts(j),'2026-09-29')
        self.assertEqual(x['metrics']['revenue_quarter'],168_847_000)
        self.assertEqual(x['metrics']['operating_cashflow_quarter'],4_809_000)
        self.assertEqual(x['metrics']['operating_cashflow_ttm'],16_467_000)
        self.assertEqual(sum(r['val'] for r in [x['sources']['operating_cashflow_quarter']]),4_809_000)
        self.assertEqual(x['sources']['operating_cashflow_quarter']['source_kind'],'cumulative_difference')
        self.assertTrue(all(r['filed']<'2026-09-29' for s in x['sources'].values() for r in s.get('source_records',[])))

if __name__=='__main__':unittest.main()
