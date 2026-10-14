import unittest,copy,math
import pandas as pd
from features91 import coordinates,read,DATA,snapshot
from reference_features91 import reference,FIELDS


class FundamentalTests(unittest.TestCase):
    def setUp(self):self.data=read(DATA)

    def check(self,at,data=None):
        data=data or self.data
        values,meta=coordinates(snapshot(data,at));expected,refmeta=reference(data,pd.Timestamp(at))
        for k in FIELDS:self.assertAlmostEqual(values[k],expected[k],places=12)
        self.assertEqual(meta,refmeta)
        return values,meta

    def test_all_publication_availability_boundaries(self):
        for r in self.data['records']:
            at=pd.Timestamp(r['available_at'])
            if '2023-01-01'<=str(at.date())<='2026-04-30':
                self.check((at-pd.Timedelta(seconds=1)).isoformat());self.check(at.isoformat())

    def test_future_mutations_and_input_order_do_not_change_old_state(self):
        at='2025-01-02T09:25:30-05:00';a=self.check(at);changed=copy.deepcopy(self.data)
        for r in changed['records']:
            if pd.Timestamp(r['available_at'])>pd.Timestamp(at):
                r['actual_revenue_million']=1e9;r['guidance_midpoint_million']=1.;r['scope_status']='unresolved'
        changed['records'].reverse();self.assertEqual(a,self.check(at,changed))

    def test_annual_guidance_never_enters_quarter_outlook(self):
        _,meta=self.check('2024-07-01T09:25:30-04:00')
        annual={r['id'] for r in self.data['records'] if r['guidance_scope']=='annual'}
        self.assertFalse(set(meta['forward_ids'])&annual)

    def test_unknown_current_scope_does_not_reuse_old_pair(self):
        at='2025-08-08T09:25:30-04:00';_,meta=self.check(at)
        state=snapshot(self.data,at);amd=next(r for r in state['issuers'] if r['symbol']=='AMD')
        self.assertEqual(amd['status'],'business_scope_unresolved')
        self.assertNotIn(amd['latest_result_id'],meta['paired_ids']);self.assertNotIn(amd['latest_result_id'],meta['forward_ids'])

    def test_empty_means_have_zero_counts_and_all_missing_kept(self):
        s=dict(known=True,issuers=[dict(symbol=str(i),status='no_result_yet') for i in range(8)])
        values,meta=coordinates(s);self.assertEqual(list(values.values()),[0.]*6);self.assertEqual(len(meta['unknown']),8)

    def test_outside_coverage_rejected(self):
        for at in ['2022-12-31T15:00:00-05:00','2026-05-01T09:25:30-04:00']:
            with self.assertRaises(ValueError):coordinates(snapshot(self.data,at))

    def test_above_upper_is_strict_and_mean_is_log_not_log_mean(self):
        s=dict(known=True,issuers=[dict(symbol=str(i),status='no_result_yet') for i in range(8)])
        for i,(actual,ref) in enumerate([(120.,100.),(80.,100.)]):
            s['issuers'][i].update(status='matched_regular_outlook',realization=actual/ref-1,actual_revenue_million=actual,reference_upper_million=actual,
                age_calendar_days=10+i*2,latest_result_id=str(i),reference_id='r'+str(i),scope_status='reported_issuer_revenue_proxy',next_guidance_scope='quarter',next_guidance_midpoint_million=ref)
        values,meta=coordinates(s)
        self.assertEqual(values[FIELDS[0]],2);self.assertEqual(values[FIELDS[2]],-1);self.assertEqual(values[FIELDS[5]],11)
        self.assertAlmostEqual(values[FIELDS[1]],math.log(.96)/2);self.assertAlmostEqual(values[FIELDS[4]],-math.log(.96)/2)


if __name__=='__main__':unittest.main()
