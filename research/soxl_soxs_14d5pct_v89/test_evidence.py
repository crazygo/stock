"""Failure cases for revenue evidence; no market outcomes or network calls."""
import copy
import json
import unittest
from datetime import datetime
from catalogue import SOURCE, CACHE, sha
from parse_revenue import actual, guidance, range_value, quarter, lam_quarter
from pair_candidates import pairings, require_training_ready


class RevenueEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads((SOURCE/'evidence/revenue_candidates.json').read_text())
        cls.rows = cls.dataset['records']
        cls.review = json.loads((SOURCE/'review_cases.json').read_text())
        cls.updates = cls.review['updates']

    def preview(self, rows=None, updates=None, cutoff='2026-05-02T00:00:00-04:00'):
        return pairings(self.rows if rows is None else rows,
                        self.updates if updates is None else updates,
                        datetime.fromisoformat(cutoff))

    def test_eight_independently_read_table_and_guidance_anchors(self):
        by_id = {r['id']: r for r in self.rows}
        self.assertEqual(8, len(self.review['numeric_anchors']))
        for anchor in self.review['numeric_anchors']:
            with self.subTest(symbol=anchor['symbol']):
                r = by_id[anchor['id']]
                self.assertEqual(anchor['source_sha256'], r['source_sha256'])
                a, g = r['actual_candidate'], r['guidance_candidate']
                self.assertEqual(anchor['actual_period'], a['fiscal_period'])
                self.assertEqual(anchor['guidance_period'], g['fiscal_period'])
                self.assertAlmostEqual(anchor['actual_million'], a['revenue_million'])
                self.assertAlmostEqual(anchor['guidance_lower_million'], g['range_million']['lower'])
                self.assertAlmostEqual(anchor['guidance_upper_million'], g['range_million']['upper'])

    def test_annual_guidance_is_not_a_quarter_denominator(self):
        for row in self.preview():
            if row['symbol']=='AVGO' and row['fiscal_period'] in ['2024Q1','2024Q2','2024Q3']:
                self.assertIsNone(row['ratio_to_regular_release_midpoint'])
                self.assertEqual('no_prior_quarter_guidance_in_catalogue', row['status'])

    def test_missing_beginning_of_catalogue_stays_missing(self):
        for s in {r['symbol'] for r in self.rows}:
            first = next(r for r in self.preview() if r['symbol']==s)
            self.assertIsNone(first['regular_release_reference_id'])
            self.assertIsNone(first['ratio_to_regular_release_midpoint'])

    def test_preliminary_revenue_is_never_final_actual(self):
        for u in self.updates:
            d = json.loads((CACHE/'documents'/(u['id']+'.json')).read_text())
            self.assertIsNone(actual(d))
            self.assertIsNone(guidance(d))
        amd = next(r for r in self.preview() if r['symbol']=='AMD' and r['fiscal_period']=='2022Q3')
        self.assertTrue(amd['preliminary_results_seen'])
        self.assertAlmostEqual(5565/6700-1, amd['ratio_to_last_known_numeric_reference'])

    def test_approaching_amount_does_not_become_exact_midpoint(self):
        r = next(r for r in self.preview() if r['symbol']=='MU' and r['fiscal_period']=='2024Q1')
        self.assertEqual('latest_known_revision_has_no_exact_midpoint', r['status'])
        self.assertIsNone(r['ratio_to_last_known_numeric_reference'])
        self.assertAlmostEqual(4726/4400-1, r['ratio_to_regular_release_midpoint'])

    def test_revisions_keep_original_and_updated_comparisons(self):
        expectations = [('AMAT','2022Q4',6749,6650,6400),('MU','2025Q4',11315,10700,11200)]
        for s,p,a,original,revised in expectations:
            r = next(r for r in self.preview() if r['symbol']==s and r['fiscal_period']==p)
            self.assertAlmostEqual(a/original-1, r['ratio_to_regular_release_midpoint'])
            self.assertAlmostEqual(a/revised-1, r['ratio_to_last_known_numeric_reference'])
            self.assertNotEqual(r['regular_release_reference_id'], r['last_known_reference_id'])

    def test_reaffirmation_does_not_extract_buyback_as_revenue(self):
        r = next(r for r in self.preview() if r['symbol']=='KLAC' and r['fiscal_period']=='2026Q3')
        self.assertEqual(r['regular_release_reference_id'], r['last_known_reference_id'])
        self.assertEqual(1, len(r['known_intervening_events']))
        self.assertAlmostEqual(3415.078/3350-1, r['ratio_to_last_known_numeric_reference'])

    def test_future_values_and_events_cannot_change_past_prefix(self):
        cutoff='2024-01-01T00:00:00-05:00'
        changed=copy.deepcopy(self.rows); updates=copy.deepcopy(self.updates)
        for row in changed:
            if row['published_date']>='2024-01-01':
                row['actual_candidate']={'fiscal_period':'2023Q3','revenue_million':999999999}
                if row['guidance_candidate']:
                    row['guidance_candidate']['range_million']['midpoint']=0.01
        for row in updates:
            if row['published_date']>='2024-01-01':
                row['fiscal_period']='2023Q3';row['new_revenue_million']={'midpoint':0.01}
        self.assertEqual(self.preview(cutoff=cutoff),self.preview(changed,updates,cutoff))

    def test_result_unavailable_before_assumed_release_cutoff(self):
        before=self.preview(cutoff='2024-02-22T23:59:59-05:00')
        after=self.preview(cutoff='2024-02-23T00:00:00-05:00')
        target=lambda r:r['symbol']=='NVDA' and r['fiscal_period']=='2024Q4'
        self.assertFalse(any(map(target,before)));self.assertTrue(any(map(target,after)))

    def test_timezone_is_mandatory(self):
        with self.assertRaises(ValueError):self.preview(cutoff='2024-01-01T00:00:00')

    def test_pending_candidates_and_preview_are_rejected_for_training(self):
        with self.assertRaises(ValueError):require_training_ready(self.dataset)
        with self.assertRaises(ValueError):require_training_ready(json.loads((SOURCE/'evidence/pair_candidates.json').read_text()))
        forged=dict(stage='fully_reviewed_point_in_time_features',training_use_allowed=True,records=self.preview())
        with self.assertRaises(ValueError):require_training_ready(forged)

    def test_micron_inventory_charge_is_not_revenue_guidance(self):
        r=next(r for r in self.rows if r['symbol']=='MU' and r['published_date']=='2023-03-28')
        self.assertEqual(3700,r['guidance_candidate']['range_million']['midpoint'])

    def test_amat_millions_table_requires_explicit_unit(self):
        r=next(r for r in self.rows if r['symbol']=='AMAT' and r['published_date']=='2026-02-12')
        d=json.loads((CACHE/'documents'/(r['id']+'.json')).read_text())
        self.assertEqual(7150,guidance(d)['range_million']['lower'])
        d['text']=d['text'].replace('In millions','In unspecified units')
        self.assertIsNone(guidance(d))

    def test_million_billion_range_and_percent_units(self):
        self.assertEqual({'lower':980,'upper':1020,'form':'symmetric_band'},range_value('$1 billion, plus or minus 2%'))
        self.assertEqual({'lower':2200,'upper':2500,'form':'range'},range_value('$2,200 million to $2,500 million'))
        self.assertIsNone(range_value('$1 billion plus or minus unknown'))

    def test_fiscal_not_calendar_year_and_lam_june_boundary(self):
        self.assertEqual('2026Q2',quarter('Fiscal 2026 Second Quarter'))
        self.assertEqual('2024Q4',lam_quarter('quarter ended June 30, 2024'))
        self.assertEqual('2025Q1',lam_quarter('quarter ending September 29, 2024'))


if __name__=='__main__':
    unittest.main()
