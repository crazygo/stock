"""Verify all historical query prefixes with a separate date-based reference."""
import gzip
from collections import Counter
from datetime import date,datetime,timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from prepare_review import *
from issuer_state import snapshot,DATA,SYMBOLS


def verify():
    data=read(DATA);panel=ROOT/'.cache/soxl_soxs_14d5pct_v80/runs/development/semiconductor_panel.parquet'
    # Read timestamps only; future financial or market labels are not inputs.
    times=pd.read_parquet(panel,columns=[]).index
    coverage=Counter();statuses=Counter();source_counts=Counter();checks=0;daily={};pairs=[]
    for r in data['records']+data['special_events']:
        expected=datetime.combine(date.fromisoformat(r['published_date'])+timedelta(days=2),datetime.min.time(),ZoneInfo('America/New_York'))
        assert datetime.fromisoformat(r['available_at'])==expected
    path=CACHE/'snapshots.jsonl.gz'
    with gzip.open(path,'wt',encoding='utf-8') as out:
        for at in times:
            result=snapshot(data,at.isoformat());assert result['known']
            d=at.to_pydatetime().astimezone(ZoneInfo('America/New_York')).date()
            cutoff=(d-timedelta(days=2)).isoformat()
            history=[r for r in data['records'] if r['published_date']<=cutoff]
            events=[r for r in data['special_events'] if r['published_date']<=cutoff]
            for s,state in zip(SYMBOLS,result['issuers']):
                assert state['symbol']==s
                issued=sorted([r for r in history if r['symbol']==s],key=lambda r:r['published_date'])
                last=issued[-1]
                assert state['latest_result_id']==last['id']
                assert state['actual_period']==last['actual_period']
                assert state['actual_revenue_million']==last['actual_revenue_million']
                refs=[r for r in issued if r['published_date']<last['published_date']
                      and r['guidance_scope']=='quarter' and r['guidance_period']==last['actual_period']]
                ref=refs[0] if refs else None
                status='no_same_quarter_regular_outlook' if ref is None else ('business_scope_unresolved' if last['scope_status']=='unresolved' else 'matched_regular_outlook')
                assert state['status']==status
                assert state['reference_id']==(ref['id'] if ref else None)
                expected=last['actual_revenue_million']/ref['guidance_midpoint_million']-1 if status=='matched_regular_outlook' else None
                assert state['realization']==expected
                assert state['age_calendar_days']==(d-date.fromisoformat(last['published_date'])).days-2
                assert state['next_guidance_period']==last['guidance_period']
                assert state['next_guidance_midpoint_million']==last['guidance_midpoint_million']
                known=[e['id'] for e in events if e['symbol']==s and e['fiscal_period']==last['actual_period'] and e['published_date']<last['published_date']]
                assert state['known_intervening_event_ids']==known
                statuses[status]+=1;source_counts[last['id']]+=1;checks+=10
            assert result['visible_special_event_ids']==[e['id'] for e in events]
            coverage[result['matched_issuers']]+=1
            if str(d) in daily:assert result['issuers']==daily[str(d)]
            else:daily[str(d)]=result['issuers']
            out.write(json.dumps(result,separators=(',',':'))+'\n')
    for last in data['records']:
        refs=sorted([r for r in data['records'] if r['symbol']==last['symbol']
                    and r['guidance_scope']=='quarter' and r['guidance_period']==last['actual_period']
                    and r['published_date']<last['published_date']],key=lambda r:r['published_date'])
        ref=refs[0] if refs else None
        status='no_same_quarter_regular_outlook' if not ref else ('business_scope_unresolved' if last['scope_status']=='unresolved' else 'matched_regular_outlook')
        pairs.append(dict(id=last['id'],symbol=last['symbol'],period=last['actual_period'],status=status,
            reference_id=ref['id'] if ref else None,
            realization=last['actual_revenue_million']/ref['guidance_midpoint_million']-1 if status=='matched_regular_outlook' else None))
    result=dict(passed=True,query_count=len(times),date_count=len(daily),issuer_states=len(times)*8,
        scalar_field_checks=checks,matched_issuer_distribution=dict(coverage),status_counts=dict(statuses),
        distinct_result_records_referenced=len(source_counts),pair_status_counts=dict(Counter(r['status'] for r in pairs)),
        timestamps_only_input=dict(path=str(panel.relative_to(ROOT)),sha256=sha(panel)),
        snapshot_cache=dict(path=str(path.relative_to(ROOT)),sha256=sha(path)),
        reviewed_records_sha256=sha(DATA),independent_method='Date cutoff d-2, issuer publication ordering, direct reference/ratio arithmetic. No production snapshot helpers used to build expected values.',
        new_market_returns_read=False,new_model_trained=False,goal_complete=False)
    dump(SOURCE/'evidence/pairs.json',dict(records=pairs,reference='prior_regular_quarterly_release'))
    dump(SOURCE/'evidence/state_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify()
