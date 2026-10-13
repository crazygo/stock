import copy
import json
from pathlib import Path
import unittest
from earnings_calendar import load_records, query, historical_availability
from prepare_review import candidate, parse_publication
from audit_calendar import audit, results_title

SOURCE=Path(__file__).resolve().parent


class CalendarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=load_records()
        cls.coverage=json.loads((SOURCE/'coverage.json').read_text())['issuers']

    def q(self, cutoff, records=None, coverage=None):
        return query(cutoff,self.records if records is None else records,self.coverage if coverage is None else coverage)

    def event(self, result, symbol, kind, day):
        return next(r for r in result['events'] if r['symbol']==symbol and r['kind']==kind and r['event_date']==day)

    def test_real_revision_only_visible_after_publication_delay(self):
        before=self.event(self.q('2024-02-17T09:25:30-05:00'),'AVGO','earnings_call','2024-03-07')
        after=self.event(self.q('2024-02-18T09:25:30-05:00'),'AVGO','earnings_call','2024-03-07')
        self.assertEqual(before['planned_time_et'],'17:00');self.assertEqual(after['planned_time_et'],'17:15')
        self.assertEqual(before['event_key'],after['event_key'])

    def test_future_append_does_not_change_earlier_snapshot(self):
        prefix=[r for r in self.records if r['published_date']<'2024-01-01']
        self.assertEqual(self.q('2023-12-01T09:25:30-05:00',prefix),self.q('2023-12-01T09:25:30-05:00'))

    def test_amd_embedded_announcements_are_retained(self):
        for day,cut in [('2022-11-01','2022-10-10T09:25:30-04:00'),('2023-01-31','2023-01-12T09:25:30-05:00'),('2025-11-04','2025-10-07T09:25:30-04:00')]:
            self.assertIsNotNone(self.event(self.q(cut),'AMD','earnings_call',day))

    def test_amd_partnership_call_does_not_replace_earnings_call(self):
        facts=json.loads((SOURCE/'reviewed_announcements.json').read_text())['announcements']
        a=next(a for a in facts if a['symbol']=='AMD' and a['published_date']=='2025-10-06')
        self.assertEqual({e['event_date'] for e in a['events']},{'2025-11-04'})
        self.assertEqual(next(e for e in a['events'] if e['kind']=='earnings_call')['time_precision'],'local_time_unresolved')

    def test_results_date_not_inferred_from_call_only_body(self):
        for symbol in ['MU','AMAT','LRCX']:
            self.assertFalse(any(r['symbol']==symbol and r['kind']=='earnings_release' for r in self.records))
        self.assertFalse(any(r['symbol']=='AMD' and r['published_date'] in ['2022-10-06','2023-01-11'] and r['kind']=='earnings_release' for r in self.records))

    def test_market_summary_not_counted_as_earnings_day(self):
        q=self.q('2024-05-01T09:25:30-04:00')
        self.event(q,'AMAT','market_summary','2024-05-02')
        self.assertNotIn(('AMAT','2024-05-02'),q['observed_earnings_days'])

    def test_day14_inclusive_day15_excluded(self):
        record=copy.deepcopy(self.records[0]);record.update(published_date='2024-01-01',published_at='2024-01-01T16:00:00-05:00',event_date='2024-02-14')
        a=self.q('2024-02-01T09:25:30-05:00',[record]);self.assertEqual(a['features']['planned_activity_days_next14'],1)
        record['event_date']='2024-02-15';a=self.q('2024-02-01T09:25:30-05:00',[record]);self.assertEqual(a['features']['planned_activity_days_next14'],0)

    def test_release_and_call_count_once(self):
        q=self.q('2024-02-14T09:25:30-05:00')
        hits=[r for r in q['events'] if r['symbol']=='NVDA' and r['in_window']]
        self.assertEqual(len(hits),2)
        self.assertEqual(q['observed_earnings_days'].count(('NVDA','2024-02-21')),1)

    def test_unknown_coverage_stays_unknown(self):
        cov=copy.deepcopy(self.coverage);cov['AMD']['body_catalogue_verified']=False
        q=self.q('2024-02-14T09:25:30-05:00',coverage=cov)
        self.assertTrue(q['observed_earnings_days']);self.assertIsNone(q['count_in_window'])
        self.assertTrue(all(v is None for v in q['features'].values()))

    def test_outside_review_range_stays_unknown(self):
        for cutoff in ['2022-12-30T09:25:30-05:00','2026-05-01T09:25:30-04:00']:
            self.assertEqual(len(self.q(cutoff)['missing_coverage']),8)

    def test_planned_elapsed_does_not_confirm_actual_event(self):
        q=self.q('2024-02-22T09:25:30-05:00')
        r=self.event(q,'NVDA','earnings_call','2024-02-21')
        self.assertEqual(r['temporal_state'],'planned_date_elapsed')
        self.assertFalse(q['actual_event_occurrence_verified']);self.assertFalse(q['model_or_trade_eligibility'])

    def test_date_only_and_precise_publication_delay(self):
        date_only=next(r for r in self.records if r['symbol']=='NVDA' and r['published_date']=='2024-05-01')
        precise=next(r for r in self.records if r['symbol']=='AMD' and r['published_date']=='2024-04-16')
        self.assertEqual(historical_availability(date_only).isoformat(),'2024-05-03T00:00:00-04:00')
        self.assertEqual(historical_availability(precise).isoformat(),'2024-04-17T00:00:00-04:00')

    def test_body_date_extractors_do_not_use_unrelated_first_call(self):
        text='AMD will hold a conference call today at 8 a.m. ET. AMD will report fiscal results on Nov. 4, 2025 after close.'
        r=candidate(dict(symbol='AMD',url='https://ir.amd.com/example',published_date='2025-10-06'),dict(text=text))
        self.assertEqual(r['candidate_event_date'],'2025-11-04')
        self.assertTrue(r['candidate_statement'].startswith('AMD will report'))

    def test_corrupt_lam_archive_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'No supported'):
            candidate(dict(symbol='LRCX',url='https://newsroom.lamresearch.com/example',published_date='2022-09-28'),dict(text='September 28, 2022: CFO will participate in an investor event on September 7, 2022.'))

    def test_publisher_publication_formats(self):
        self.assertEqual(parse_publication('Released Apr 3, 2024 • 4:30 PM EDT','2024-04-03'),'2024-04-03T16:30:00-04:00')
        self.assertEqual(parse_publication('Jan 22, 2026 at 7:30 AM EST','2026-01-22'),'2026-01-22T07:30:00-05:00')
        self.assertIsNone(parse_publication('May 1, 2024','2024-05-01'))

    def test_directory_results_filter_handles_word_order_and_mixed_titles(self):
        self.assertTrue(results_title('Micron Reports Results for the First Quarter of Fiscal 2024'))
        self.assertTrue(results_title('Intel Reports Second-Quarter Financial Results; Announces Plan to Improve Efficiency'))
        self.assertFalse(results_title('AMD Announces Preliminary Third Quarter Financial Results'))

    def test_frozen_source_audit_reproduces(self):
        expected=json.loads((SOURCE/'evidence/source_audit.json').read_text())
        self.assertEqual(audit(),expected)
        self.assertEqual(expected['announcements'],119);self.assertEqual(expected['result_directory_census'],115)

    def test_default_query_checks_frozen_input_contract(self):
        q=query('2026-04-15T09:25:30-04:00')
        self.assertEqual(q['coverage_status'],'reviewed_catalogue')
        self.assertFalse(q['model_or_trade_eligibility'])


if __name__=='__main__':unittest.main()
