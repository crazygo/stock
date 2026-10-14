"""Independent publication-date cutoff and scalar fundamental coordinates."""
from datetime import date,timedelta
from zoneinfo import ZoneInfo
import math,json
import pandas as pd
ET=ZoneInfo('America/New_York')
SYMBOLS=('NVDA','AMD','AVGO','MU','AMAT','LRCX','KLAC','INTC')
FIELDS=['fund_matched_count','fund_realization_log','fund_above_upper_centered',
        'fund_outlook_count','fund_outlook_log_growth','fund_result_mean_age']


def reference(dataset,at):
    """Do not use production available_at, state snapshot or aggregation helpers."""
    day=pd.Timestamp(at).tz_convert(ET).date()
    if not date(2023,1,1)<=day<=date(2026,4,30):raise ValueError('Outside reviewed range')
    cutoff=day-timedelta(days=2)
    logs=[];above=[];ages=[];outlooks=[];ids=[];refs=[];forward=[];unknown=[]
    for symbol in SYMBOLS:
        rows=sorted([r for r in dataset['records'] if r['symbol']==symbol and date.fromisoformat(r['published_date'])<=cutoff],key=lambda r:(r['published_date'],r['id']))
        if not rows:
            unknown.append(dict(symbol=symbol,status='no_result_yet'));continue
        current=rows[-1]
        earlier=[r for r in rows[:-1] if r['published_date']<current['published_date'] and r['guidance_scope']=='quarter' and r['guidance_period']==current['actual_period']]
        prior=earlier[0] if earlier else None
        status='no_same_quarter_regular_outlook' if prior is None else 'business_scope_unresolved' if current['scope_status']=='unresolved' else 'matched_regular_outlook'
        if status=='matched_regular_outlook':
            logs.append(math.log(current['actual_revenue_million']/prior['guidance_midpoint_million']))
            above.append(int(current['actual_revenue_million']>prior['guidance_upper_million']))
            ages.append((cutoff-date.fromisoformat(current['published_date'])).days)
            ids.append(current['id']);refs.append(prior['id'])
        else:unknown.append(dict(symbol=symbol,status=status))
        if current['scope_status']=='reported_issuer_revenue_proxy' and current['guidance_scope']=='quarter':
            outlooks.append(math.log(current['guidance_midpoint_million']/current['actual_revenue_million']));forward.append(current['id'])
    n=len(logs);k=len(outlooks)
    values=[float(n),math.fsum(logs)/n if n else 0.,2*math.fsum(above)/n-1 if n else 0.,float(k),math.fsum(outlooks)/k if k else 0.,math.fsum(ages)/n if n else 0.]
    return dict(zip(FIELDS,values)),dict(paired_ids=ids,paired_reference_ids=refs,forward_ids=forward,unknown=unknown)


def verify_table(table,panel,dataset):
    assert len(table)==len(panel) and not table['at'].duplicated().any()
    count=0;dates=set();matched={};forward={}
    for (_,saved),(at,row) in zip(table.iterrows(),panel.iterrows()):
        assert pd.Timestamp(saved['at'])==at and row.fundamental_known
        expected,meta=reference(dataset,at)
        for field,value in expected.items():
            assert math.isclose(float(saved[field]),value,rel_tol=1e-11,abs_tol=1e-13),(at,field)
            assert math.isclose(float(row[field]),value,rel_tol=1e-11,abs_tol=1e-13),(at,field)
            count+=2
        for field,values in meta.items():assert json.loads(saved[field])==values;count+=1
        dates.add(at.tz_convert(ET).date());n=int(expected[FIELDS[0]]);k=int(expected[FIELDS[3]])
        matched[n]=matched.get(n,0)+1;forward[k]=forward.get(k,0)+1
    return dict(passed=True,queries=len(table),dates=len(dates),field_checks=count,matched_issuer_distribution=matched,forward_issuer_distribution=forward)


if __name__=='__main__':
    from calibration91 import SOURCE,CACHE,PARENT,prepare_panel,extended_market,dump,sha,read,verify_parent
    from features91 import DATA
    verify_parent();market=extended_market();p=prepare_panel(pd.read_parquet(PARENT/'semiconductor_panel.parquet'),market,CACHE/'event_features.csv')
    result=verify_table(pd.read_csv(CACHE/'fundamental_features.csv'),p,read(DATA))
    result.update(source_records_sha256=sha(DATA),reference_source_sha256=sha(__import__('pathlib').Path(__file__)),new_capital_outcomes_seen=False)
    dump(SOURCE/'evidence/fundamental_verification.json',result);print(result,flush=True)
