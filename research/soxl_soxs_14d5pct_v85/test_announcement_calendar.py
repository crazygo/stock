import copy
import json
from pathlib import Path
import unittest

from announcement_calendar import PANEL, historical_availability, snapshot, validate
from acquire_earnings import validate_html
from crawl_static import article_rows

DATA = json.loads((Path(__file__).parent / 'verified_examples.json').read_text())
ROWS = DATA['records']


def complete():
    # Synthetic coverage for testing query semantics; never used for the actual fixture audit.
    return {s: dict(body_catalogue_verified=True, publication_from='2022-10-01',
                   publication_through='2026-04-30', pending_announcements_at_start_resolved=True)
            for s in PANEL}


class PublicationCalendarTests(unittest.TestCase):
    def test_verified_examples_cover_eight_issuers_but_are_not_training_data(self):
        validate(ROWS)
        self.assertEqual(set(PANEL), {r['symbol'] for r in ROWS})
        self.assertFalse(DATA['training_eligible'])

    def test_day_only_publication_keeps_unknown_time_and_delayed_assumption(self):
        r = next(r for r in ROWS if r['symbol'] == 'NVDA')
        self.assertIsNone(r['published_at'])
        self.assertEqual('2024-05-03T00:00:00-04:00', historical_availability(r).isoformat())

    def test_exact_publication_used_only_next_et_day(self):
        r = next(r for r in ROWS if r['symbol'] == 'AMD')
        self.assertEqual('2024-04-17T00:00:00-04:00', historical_availability(r).isoformat())
        q = snapshot(ROWS, '2024-04-16T23:59:59-04:00', '2024-04-30', complete())
        self.assertFalse(any(r['symbol'] == 'AMD' for r in q['events']))

    def test_future_revision_cannot_overwrite_earlier_call_time(self):
        q = snapshot(ROWS, '2024-02-15T09:25:30-05:00', '2024-03-07', complete())
        calls = [r for r in q['events'] if r['symbol'] == 'AVGO' and r['kind'] == 'earnings_call']
        self.assertEqual(['17:00'], [r['planned_time_et'] for r in calls])

    def test_revision_replaces_same_event_instead_of_double_counting(self):
        q = snapshot(ROWS, '2024-02-18T00:00:00-05:00', '2024-03-07', complete())
        calls = [r for r in q['events'] if r['symbol'] == 'AVGO' and r['kind'] == 'earnings_call']
        self.assertEqual(['17:15'], [r['planned_time_et'] for r in calls])
        self.assertEqual([('AVGO', '2024-03-07')], q['observed_earnings_days'])
        self.assertEqual(1, q['count_in_window'])

    def test_future_append_preserves_prefix_snapshot(self):
        old = [r for r in ROWS if r['published_date'] < '2024-02-16']
        self.assertEqual(snapshot(old, '2024-02-15T12:00:00-05:00', '2024-03-07', complete()),
                         snapshot(ROWS, '2024-02-15T12:00:00-05:00', '2024-03-07', complete()))

    def test_asof_before_initial_announcement_has_no_known_avgo_event(self):
        q = snapshot(ROWS, '2024-02-08T23:59:59-05:00', '2024-03-07', complete())
        self.assertFalse(any(r['symbol'] == 'AVGO' for r in q['events']))

    def test_market_summary_is_not_earnings_and_fourteen_days_are_inclusive(self):
        q = snapshot(ROWS, '2024-05-01T09:25:30-04:00', '2024-05-14', complete())
        self.assertTrue(any(r['kind'] == 'market_summary' and r['in_window'] for r in q['events']))
        self.assertEqual(0, q['count_in_window'])
        q2 = snapshot(ROWS, '2024-05-03T09:25:30-04:00', '2024-05-16', complete())
        self.assertIn(('AMAT', '2024-05-16'), q2['observed_earnings_days'])

    def test_missing_catalogue_is_unknown_not_zero(self):
        q = snapshot(ROWS, '2024-05-01T09:25:30-04:00', '2024-05-14', DATA['coverage'])
        self.assertIsNone(q['count_in_window'])
        self.assertEqual(list(PANEL), q['missing_coverage'])
        self.assertFalse(q['model_or_trade_eligibility'])

    def test_one_missing_issuer_prevents_full_panel_count(self):
        cov = complete(); del cov['MU']
        q = snapshot(ROWS, '2024-02-18T09:00:00-05:00', '2024-03-07', cov)
        self.assertEqual(['MU'], q['missing_coverage'])
        self.assertIsNone(q['count_in_window'])
        self.assertEqual(1, len(q['observed_earnings_days']))

    def test_expired_catalogue_and_unresolved_start_inventory_are_unknown(self):
        cov = complete(); cov['MU']['publication_through'] = '2024-02-17'
        cov['AMD']['pending_announcements_at_start_resolved'] = False
        q = snapshot(ROWS, '2024-02-18T09:00:00-05:00', '2024-03-07', cov)
        self.assertEqual(['AMD', 'MU'], q['missing_coverage'])

    def test_elapsed_planned_date_does_not_prove_actual_disclosure(self):
        q = snapshot(ROWS, '2024-03-08T09:00:00-05:00', '2024-03-21', complete())
        avgo = [r for r in q['events'] if r['symbol'] == 'AVGO']
        self.assertTrue(all(r['temporal_state'] == 'planned_date_elapsed' for r in avgo))
        self.assertFalse(q['actual_event_occurrence_verified'])

    def test_no_exact_release_time_from_conference_only_sources(self):
        self.assertFalse(any(r['kind'] == 'earnings_release' and r['symbol'] in ['MU', 'LRCX', 'AMAT'] for r in ROWS))
        self.assertTrue(all(r['planned_time_et'] is None for r in ROWS if r['time_precision'] == 'after_regular_close'))

    def test_unknown_mountain_zone_not_silently_converted(self):
        r = next(r for r in ROWS if r['symbol'] == 'MU')
        self.assertIsNone(r['planned_time_et'])
        self.assertEqual('local_time_unresolved', r['time_precision'])

    def test_naive_cutoff_and_backwards_window_rejected(self):
        with self.assertRaises(ValueError):snapshot(ROWS, '2024-03-08T09:00:00', '2024-03-21', complete())
        with self.assertRaises(ValueError):snapshot(ROWS, '2024-03-08T09:00:00-05:00', '2024-03-07', complete())

    def test_duplicate_or_ambiguous_versions_rejected(self):
        with self.assertRaises(ValueError):validate(ROWS + [ROWS[0]])
        extra = dict(ROWS[0], record_id='different_id')
        with self.assertRaises(ValueError):validate(ROWS + [extra])

    def test_title_only_or_misclassified_evidence_rejected(self):
        for changed in [dict(body_verified=False), dict(time_precision='after_regular_close'), dict(kind='rumor')]:
            rows = copy.deepcopy(ROWS); rows[0].update(changed)
            with self.assertRaises(ValueError):validate(rows)

    def test_issuer_or_kind_cannot_change_in_revision(self):
        rows = copy.deepcopy(ROWS); rows[6]['symbol'] = 'AMD'
        with self.assertRaises(ValueError):validate(rows)


class AcquisitionTests(unittest.TestCase):
    def test_truncated_http_200_document_is_rejected(self):
        with self.assertRaises(ValueError):validate_html(b'<html><body>' + b'x' * 200)
        self.assertIn('</html>', validate_html(b'<html><body>' + b'x' * 200 + b'</body></html>'))

    def test_undated_lam_url_does_not_break_directory_traversal(self):
        html = '<li class="wd_item"><div class="wd_date">May 20, 2026</div><a href="/some-official-release">An example release</a></li>'
        r = article_rows(html, 'https://newsroom.lamresearch.com', 'LRCX')
        self.assertEqual('2026-05-20', r[0]['published_date'])
        self.assertEqual('https://newsroom.lamresearch.com/some-official-release', r[0]['url'])


if __name__ == '__main__':unittest.main()
