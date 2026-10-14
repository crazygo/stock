"""Preview same-quarter pairings. This module deliberately exposes no model loader.

Publication date + two calendar days is a conservative *candidate* availability
assumption. Source dates/versions and business scope remain under review.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from catalogue import *

ET = ZoneInfo('America/New_York')


def available_at(row):
    day = date.fromisoformat(row['published_date']) + timedelta(days=2)
    return datetime.combine(day, datetime.min.time(), ET)


def pairings(rows, updates, cutoff):
    if cutoff.tzinfo is None:
        raise ValueError('An explicit timezone is required')
    visible = [r for r in rows if available_at(r) <= cutoff]
    changes = [r for r in updates if available_at(r) <= cutoff]
    output = []
    for current in visible:
        actual = current['actual_candidate']
        if actual is None:
            continue
        period = actual['fiscal_period']
        previous = [r for r in visible if r['symbol'] == current['symbol']
                    and r['published_date'] < current['published_date']
                    and (g := r['guidance_candidate'])
                    and g['scope'] == 'quarter' and g['fiscal_period'] == period
                    and g['range_million'] is not None]
        # A regular quarterly release is a reproducible named reference. It is
        # not asserted to be the issuer's earliest or latest public guidance.
        initial = min(previous, key=lambda r: r['published_date']) if previous else None
        known = sorted([u for u in changes if u['symbol'] == current['symbol']
                        and u['fiscal_period'] == period
                        and u['published_date'] < current['published_date']],
                       key=lambda r: r['published_date'])
        reference = initial
        denominator = initial['guidance_candidate']['range_million']['midpoint'] if initial else None
        initial_ratio = actual['revenue_million'] / denominator - 1 if denominator else None
        status = 'numeric_regular_release_reference' if initial else 'no_prior_quarter_guidance_in_catalogue'
        for change in known:
            if change['kind'] != 'guidance_revision':
                continue
            reference = change
            value = change['new_revenue_million']
            denominator = value['midpoint'] if isinstance(value, dict) else None
            status = 'numeric_known_revision' if denominator else 'latest_known_revision_has_no_exact_midpoint'
        updated_ratio = actual['revenue_million'] / denominator - 1 if denominator else None
        output.append(dict(actual_id=current['id'], symbol=current['symbol'],
            fiscal_period=period, actual_published_date=current['published_date'],
            assumed_available_at=available_at(current).isoformat(),
            regular_release_reference_id=initial['id'] if initial else None,
            last_known_reference_id=reference['id'] if reference else None,
            known_intervening_events=[u['id'] for u in known],
            preliminary_results_seen=any(u['kind']=='preliminary_results' for u in known),
            status=status, ratio_to_regular_release_midpoint=initial_ratio,
            ratio_to_last_known_numeric_reference=updated_ratio,
            training_use_allowed=False,
            unresolved=['full_numeric_review', 'publication_and_revision_history',
                        'business_scope_comparability']))
    return sorted(output, key=lambda r: (r['symbol'], r['fiscal_period']))


def require_training_ready(dataset):
    """Fail closed even if a caller accidentally passes this preview to a model."""
    if dataset.get('stage') != 'fully_reviewed_point_in_time_features':
        raise ValueError('v89 contains unapproved candidate evidence, not training features')
    if not dataset.get('training_use_allowed') or not dataset.get('records'):
        raise ValueError('No approved records')
    for row in dataset['records']:
        if not row.get('training_use_allowed') or row.get('unresolved'):
            raise ValueError('An unresolved or unapproved record is present')
    return dataset['records']


def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('--as-of', default='2026-05-02T00:00:00-04:00')
    args = parser.parse_args()
    candidates = json.loads((SOURCE/'evidence/revenue_candidates.json').read_text())
    review = json.loads((SOURCE/'review_cases.json').read_text())
    rows = pairings(candidates['records'], review['updates'], datetime.fromisoformat(args.as_of))
    result = dict(stage='mechanical_pairing_preview_not_features', as_of=args.as_of,
        availability_assumption='publication_date_plus_two_calendar_days_at_ET_midnight_not_verified_historical_receipt',
        summary=dict(Counter(r['status'] for r in rows)), records=rows,
        known_revision_history_complete=False, training_use_allowed=False)
    dump(SOURCE/'evidence/pair_candidates.json', result)
    print(result['summary'])


if __name__ == '__main__':
    build()
