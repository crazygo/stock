import copy
import hashlib
import json
import unittest
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from event_calendar import CATALOG, ROOT, features, from_records, inputs
from verify_calendar import reference_raw, reference


class CalendarTests(unittest.TestCase):
    def setUp(self):
        self.catalog=json.loads(CATALOG.read_text())
        self.records,self.coverage=inputs(json.dumps(self.catalog,sort_keys=True))

    def sample(self,activity='one',url='https://example.test/first',published='2024-03-07T20:00:00-05:00',
               pubdate='2024-03-07',day='2024-03-11',kinds=('earnings_call','earnings_release')):
        a=dict(symbol='NVDA',source_url=url,published_at=published,published_date=pubdate,
               activity_key=activity,events=[dict(kind=k,event_date=day) for k in kinds])
        rows=[dict(record_id=hashlib.sha256((url+'#'+k).encode()).hexdigest(),event_key=activity+'_'+k,
            symbol='NVDA',kind=k,published_at=published,published_date=pubdate,event_date=day,
            planned_time_et=None,time_precision='date_only',source_url=url,body_verified=True) for k in kinds]
        return a,rows

    def check(self,ann,rows,at,end,coverage=None):
        coverage=self.coverage if coverage is None else coverage
        actual=from_records(rows,coverage,at,end)
        self.assertEqual(actual,reference_raw(ann,coverage,at,end))
        return actual

    def test_exact_publication_is_unavailable_until_next_et_midnight(self):
        a,r=self.sample()
        before=self.check([a],r,'2024-03-07T23:59:59-05:00','2024-03-20T16:00:00-04:00')
        after=self.check([a],r,'2024-03-08T00:00:00-05:00','2024-03-20T16:00:00-04:00')
        self.assertEqual(before['values'],[0,0,0,0]);self.assertEqual(after['values'][0],1)
        self.assertEqual(len(after['visible_record_ids']),2)

    def test_date_only_is_delayed_two_calendar_dates(self):
        a,r=self.sample(published=None)
        for at,expected in [('2024-03-08T23:59:59-05:00',0),('2024-03-09T00:00:00-05:00',1)]:
            self.assertEqual(self.check([a],r,at,'2024-03-20T16:00:00-04:00')['values'][0],expected)

    def test_revision_is_visible_only_after_its_own_cutoff(self):
        a,r=self.sample();b,s=self.sample(url='https://example.test/revision',pubdate='2024-03-08',published='2024-03-08T12:00:00-05:00',day='2024-03-12')
        old=self.check([a,b],r+s,'2024-03-08T15:00:00-05:00','2024-03-20T16:00:00-04:00')
        new=self.check([a,b],r+s,'2024-03-09T09:00:00-05:00','2024-03-20T16:00:00-04:00')
        self.assertEqual(old['future_days'],[['NVDA','2024-03-11']]);self.assertEqual(new['future_days'],[['NVDA','2024-03-12']])
        self.assertEqual(len(new['visible_record_ids']),2)

    def test_event_day_deduplication_and_no_intraday_occurrence_claim(self):
        a,r=self.sample()
        morning=self.check([a],r,'2024-03-11T09:25:30-04:00','2024-03-22T16:00:00-04:00')
        late=self.check([a],r,'2024-03-11T15:05:30-04:00','2024-03-22T16:00:00-04:00')
        self.assertEqual(morning,late);self.assertEqual(late['values'],[1,1,0,0])
        self.assertFalse(late['actual_event_occurrence_verified']);self.assertFalse(late['exact_intraday_event_window'])

    def test_official_expiry_boundary_and_market_summary_exclusion(self):
        a,r=self.sample(day='2024-03-22');b,s=self.sample(activity='two',url='https://example.test/late',day='2024-03-23')
        c,t=self.sample(activity='three',url='https://example.test/summary',day='2024-03-12',kinds=('market_summary',))
        e=self.check([a,b,c],r+s+t,'2024-03-11T09:25:30-04:00','2024-03-22T16:00:00-04:00')
        self.assertEqual(e['future_days'],[['NVDA','2024-03-22']]);self.assertEqual(e['values'][0],1)

    def test_recent_day_boundaries_are_calendar_days_across_dst(self):
        a,r=self.sample(day='2024-03-08');b,s=self.sample(activity='two',url='https://example.test/two',day='2024-03-09')
        e=self.check([a,b],r+s,'2024-03-22T09:25:30-04:00','2024-04-04T16:00:00-04:00')
        self.assertEqual(e['values'],[0,0,2,1-13/15])
        f=self.check([a,b],r+s,'2024-03-23T09:25:30-04:00','2024-04-04T16:00:00-04:00')
        self.assertEqual(f['previous_days'],[['NVDA','2024-03-09']])

    def test_missing_issuer_and_expired_catalogue_stay_unknown(self):
        bad=copy.deepcopy(self.coverage);del bad['AMD']
        a,r=self.sample()
        e=self.check([a],r,'2024-03-11T09:25:30-04:00','2024-03-22T16:00:00-04:00',bad)
        self.assertFalse(e['known']);self.assertIsNone(e['values']);self.assertEqual(e['missing_coverage'],['AMD'])
        f=features(self.catalog,'2026-05-01T09:25:30-04:00','2026-05-14T16:00:00-04:00')
        self.assertFalse(f['known']);self.assertIsNone(f['values']);self.assertEqual(len(f['missing_coverage']),8)

    def test_future_announcement_append_leaves_existing_query_identical(self):
        a,r=self.sample();b,s=self.sample(activity='future',url='https://example.test/future',pubdate='2024-03-12',published='2024-03-12T12:00:00-04:00',day='2024-03-15')
        at='2024-03-11T09:25:30-04:00';end='2024-03-22T16:00:00-04:00'
        self.assertEqual(self.check([a],r,at,end),self.check([a,b],r+s,at,end))

    def test_original_calendar_reference_on_release_and_revision_dates(self):
        for day in ['2023-04-03','2024-02-08','2024-02-17','2025-10-07','2026-04-30']:
            zone=ZoneInfo('America/New_York')
            at=datetime.fromisoformat(day+'T09:25:30').replace(tzinfo=zone).isoformat()
            end=datetime.fromisoformat((date.fromisoformat(day)+timedelta(days=13)).isoformat()+'T16:00:00').replace(tzinfo=zone).isoformat()
            self.assertEqual(features(self.catalog,at,end),reference(self.catalog,at,end))


if __name__=='__main__':unittest.main()
