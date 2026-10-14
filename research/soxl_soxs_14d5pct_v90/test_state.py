import copy,json,unittest
from datetime import datetime,timedelta
from prepare_review import SOURCE,read,CACHE
from issuer_state import DATA,snapshot
from review_register import selected_table


class StateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=read(DATA)
    def get(self,when,symbol,data=None):
        return next(r for r in snapshot(data or self.data,when)['issuers'] if r['symbol']==symbol)
    def test_all_reviewed_table_cells_and_original_numbers(self):
        for r in self.data['records']:
            with self.subTest(id=r['id']):
                p=read(CACHE/'review_packets'/(r['id']+'.json'));t,i,n=selected_table(p)
                self.assertEqual(r['actual_revenue_million'],n)
                self.assertEqual((r['actual_table'],r['actual_table_row'],r['actual_value_cell']),(t['table'],t['row'],i))
    def test_date_plus_two_boundary(self):
        before=self.get('2024-02-22T23:59:59-05:00','NVDA')
        after=self.get('2024-02-23T00:00:00-05:00','NVDA')
        self.assertEqual('2024Q3',before['actual_period']);self.assertEqual('2024Q4',after['actual_period'])
        self.assertAlmostEqual(22103/20000-1,after['realization'])
    def test_equivalent_timezones(self):
        a=self.get('2024-02-23T00:00:00-05:00','NVDA')
        b=self.get('2024-02-23T05:00:00+00:00','NVDA');self.assertEqual(a,b)
    def test_missing_timezone_rejected(self):
        with self.assertRaises(ValueError):snapshot(self.data,'2024-02-23T00:00:00')
    def test_annual_cannot_supply_quarter_reference_or_old_fallback(self):
        r=self.get('2024-06-15T09:25:30-04:00','AVGO')
        self.assertEqual('2024Q2',r['actual_period']);self.assertIsNone(r['realization'])
        self.assertIsNone(r['reference_id']);self.assertEqual('annual',r['next_guidance_scope'])
    def test_scope_uncertainty_does_not_fall_back_to_prior_good_pair(self):
        r=self.get('2025-10-25T09:25:30-04:00','INTC')
        self.assertEqual('2025Q3',r['actual_period'])
        self.assertEqual('business_scope_unresolved',r['status']);self.assertIsNone(r['realization'])
        later=self.get('2026-01-24T09:25:30-05:00','INTC')
        self.assertEqual('matched_regular_outlook',later['status'])
        self.assertAlmostEqual(13674/13300-1,later['realization'])
    def test_interim_guidance_does_not_inject_final_actual(self):
        # Micron's Nov28 revision is visible, but the Dec20 result is not.
        r=self.get('2023-11-30T09:25:30-05:00','MU')
        self.assertEqual('2023Q4',r['actual_period'])
        self.assertEqual(4010,r['actual_revenue_million'])
        self.assertEqual(4400,r['next_guidance_midpoint_million'])
    def test_original_guidance_not_replaced_by_later_revision(self):
        r=self.get('2023-12-23T09:25:30-05:00','MU')
        self.assertEqual('2024Q1',r['actual_period']);self.assertEqual(4400,r['reference_midpoint_million'])
        self.assertAlmostEqual(4726/4400-1,r['realization']);self.assertEqual(1,len(r['known_intervening_event_ids']))
    def test_future_values_scope_and_events_cannot_rewrite_old_snapshot(self):
        cutoff='2024-04-01T12:00:00-04:00';changed=copy.deepcopy(self.data)
        for r in changed['records']:
            if datetime.fromisoformat(r['available_at'])>datetime.fromisoformat(cutoff):
                r['actual_revenue_million']=9999999;r['scope_status']='unresolved';r['guidance_midpoint_million']=.01
        for e in changed['special_events']:
            if datetime.fromisoformat(e['available_at'])>datetime.fromisoformat(cutoff):e['new_revenue_million']=999999
        self.assertEqual(snapshot(self.data,cutoff),snapshot(changed,cutoff))
    def test_removing_future_records_keeps_prefix(self):
        cutoff='2025-01-01T00:00:00-05:00';changed=copy.deepcopy(self.data);at=datetime.fromisoformat(cutoff)
        changed['records']=[r for r in changed['records'] if datetime.fromisoformat(r['available_at'])<=at]
        changed['special_events']=[r for r in changed['special_events'] if datetime.fromisoformat(r['available_at'])<=at]
        self.assertEqual(snapshot(self.data,cutoff),snapshot(changed,cutoff))
    def test_outside_coverage_is_unknown(self):
        for t in ['2022-12-31T23:59:59-05:00','2026-05-01T00:00:00-04:00']:
            self.assertFalse(snapshot(self.data,t)['known'])
    def test_no_strategy_qualification_from_source_review(self):
        self.assertFalse(snapshot(self.data,'2026-04-30T12:00:00-04:00')['market_model_qualified'])


if __name__=='__main__':unittest.main()
