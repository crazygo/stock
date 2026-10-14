"""As-known snapshots of reviewed issuer archive proxies, with explicit unknowns."""
from datetime import datetime,date
from zoneinfo import ZoneInfo
from prepare_review import SOURCE,read

SYMBOLS=('NVDA','AMD','AVGO','MU','AMAT','LRCX','KLAC','INTC')
ET=ZoneInfo('America/New_York')
DATA=SOURCE/'evidence/reviewed_records.json'


def timestamp(text):
    d=datetime.fromisoformat(text)
    if d.tzinfo is None:raise ValueError('Explicit timezone required')
    return d


def visible(dataset,at):
    return [r for r in dataset['records'] if timestamp(r['available_at'])<=at]


def snapshot(dataset,asof):
    at=timestamp(asof);day=at.astimezone(ET).date()
    if not date(2023,1,1)<=day<=date(2026,4,30):
        return dict(known=False,reason='outside_reviewed_query_coverage',as_of=asof,issuers=[])
    if dataset.get('stage')!='reviewed_historical_archive_proxy_records':
        raise ValueError('Reviewed source records required')
    rows=visible(dataset,at)
    events=[r for r in dataset['special_events'] if timestamp(r['available_at'])<=at]
    states=[]
    for symbol in SYMBOLS:
        history=sorted([r for r in rows if r['symbol']==symbol],key=lambda r:(r['available_at'],r['id']))
        if not history:
            states.append(dict(symbol=symbol,status='no_result_yet',realization=None,latest_result_id=None))
            continue
        latest=history[-1];period=latest['actual_period']
        references=[r for r in history if r['guidance_scope']=='quarter'
                    and r['guidance_period']==period
                    and timestamp(r['available_at'])<timestamp(latest['available_at'])]
        reference=references[0] if references else None
        related=[r for r in events if r['symbol']==symbol and r['fiscal_period']==period
                 and timestamp(r['available_at'])<timestamp(latest['available_at'])]
        status='matched_regular_outlook'
        if reference is None:status='no_same_quarter_regular_outlook'
        elif latest['scope_status']=='unresolved':status='business_scope_unresolved'
        value=None if status!='matched_regular_outlook' else latest['actual_revenue_million']/reference['guidance_midpoint_million']-1
        # Missing current data must never silently fall back to a nicer old pair.
        state=dict(symbol=symbol,status=status,latest_result_id=latest['id'],
            result_available_at=latest['available_at'],actual_period=period,
            actual_revenue_million=latest['actual_revenue_million'],
            reference_id=reference['id'] if reference else None,
            reference_midpoint_million=reference['guidance_midpoint_million'] if reference else None,
            reference_lower_million=reference['guidance_lower_million'] if reference else None,
            reference_upper_million=reference['guidance_upper_million'] if reference else None,
            realization=value,scope_status=latest['scope_status'],scope_note=latest['scope_note'],
            age_calendar_days=(day-timestamp(latest['available_at']).astimezone(ET).date()).days,
            known_intervening_event_ids=[r['id'] for r in related],
            preliminary_already_known=any(r['kind']=='preliminary_results' for r in related),
            next_guidance_scope=latest['guidance_scope'],next_guidance_period=latest['guidance_period'],
            next_guidance_midpoint_million=latest['guidance_midpoint_million'])
        states.append(state)
    return dict(known=True,as_of=asof,issuers=states,
        visible_special_event_ids=[r['id'] for r in events],
        matched_issuers=sum(s['status']=='matched_regular_outlook' for s in states),
        market_model_qualified=False,source_meaning='reported_issuer_revenue_vs_scheduled_regular_outlook',
        historical_real_receipt=False,complete_revision_history=False)


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--as-of',required=True);a=p.parse_args()
    print(json.dumps(snapshot(read(DATA),a.as_of),ensure_ascii=False,indent=2))
