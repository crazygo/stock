import copy
import unittest
from datetime import timedelta
from event_calendar import *


class CalendarTests(unittest.TestCase):
    def fixture(self):
        return json.loads(CATALOG.read_text())

    def test_original_notice_unknown_until_next_midnight(self):
        c=self.fixture()
        self.assertFalse(features(c,'2022-06-24T16:00:00-04:00','2023-02-01T16:00:00-05:00')['known'])
        self.assertNotIn('2023-01',asof(c,'2022-06-24T23:59:59-04:00'))
        self.assertIn('2023-01',asof(c,'2022-06-25T00:00:00-04:00'))

    def test_future_revisions_do_not_change_past(self):
        c=self.fixture();at='2026-01-27T09:25:30-05:00';end='2026-02-06T16:00:00-05:00'
        a=features(c,at,end);v=copy.deepcopy(c['versions'][-1]);v.update(event_id='2026-01',available_at='2026-01-29T00:00:00-05:00',statement_at='2026-01-30T14:00:00-05:00')
        c['versions'].append(v);self.assertEqual(a,features(c,at,end))
        self.assertEqual(asof(c,'2026-01-29T00:00:00-05:00')['2026-01'],v)

    def test_latest_known_revision_updates_only_one_event(self):
        c=self.fixture();v=copy.deepcopy(asof(c,'2026-01-27T00:00:00-05:00')['2026-01'])
        v.update(available_at='2026-01-27T00:00:00-05:00',statement_at='2026-01-30T14:00:00-05:00');c['versions'].insert(0,v)
        known=asof(c,'2026-01-27T09:25:30-05:00');self.assertEqual(known['2026-01'],v)
        self.assertEqual(known['2026-02']['end_date'],'2026-03-18')

    def test_unknown_year_not_zero(self):
        c=self.fixture();c['coverage']=[x for x in c['coverage'] if x['year']!=2026]
        e=features(c,'2026-01-27T09:25:30-05:00','2026-02-06T16:00:00-05:00')
        self.assertFalse(e['known']);self.assertIsNone(e['values']);self.assertEqual(e['missing_years'],[2026])

    def test_expiry_excludes_later_event(self):
        e=features(self.fixture(),'2026-01-20T09:25:30-05:00','2026-01-27T16:00:00-05:00')
        self.assertEqual(e['values'][0],0);self.assertIsNone(e['next_event'])

    def test_statement_boundary_is_not_future(self):
        e=features(self.fixture(),'2026-01-28T14:00:00-05:00','2026-02-06T16:00:00-05:00')
        self.assertEqual(e['values'],[0.,1.]);self.assertEqual(e['previous_event']['event_id'],'2026-01')

    def test_dst_uses_real_elapsed_seconds(self):
        at='2026-03-05T15:05:30-05:00';end='2026-03-18T16:00:00-04:00';e=features(self.fixture(),at,end)
        a=dt(at).astimezone(ZoneInfo('UTC'));b=dt('2026-03-18T14:00:00-04:00').astimezone(ZoneInfo('UTC'))
        self.assertAlmostEqual(e['values'][0],1-(b-a).total_seconds()/(14*86400))

    def test_no_recent_event_known_zero(self):
        e=features(self.fixture(),'2026-02-20T09:25:30-05:00','2026-03-05T16:00:00-05:00')
        self.assertTrue(e['known']);self.assertEqual(e['values'],[0.,0.])

    def test_cross_year_requires_both_annual_coverages(self):
        c=self.fixture();c['coverage']=[x for x in c['coverage'] if x['year']!=2025]
        self.assertFalse(features(c,'2026-01-02T09:25:30-05:00','2026-01-15T16:00:00-05:00')['known'])

    def test_cross_month_and_thursday(self):
        self.assertEqual(dates('April 30-May 1',2024),('2024-04-30','2024-05-01'))
        self.assertEqual(dates('Wednesday-Thursday, November 6-7, 2024',2024),('2024-11-06','2024-11-07'))

    def test_timing_rule_must_be_known(self):
        c=self.fixture();c['timing_source']['available_at']='2027-01-01T00:00:00-05:00'
        self.assertFalse(features(c,'2026-01-20T09:25:30-05:00','2026-02-02T16:00:00-05:00')['known'])


if __name__=='__main__':unittest.main()
