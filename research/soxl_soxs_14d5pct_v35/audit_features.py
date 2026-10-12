"""Independent financial reconstruction via the repository's older quarter parser."""
from financial_features import core,np,pd,PANEL,FINANCIAL,TAGS
from pathlib import Path
from datetime import date,timedelta
import sys,math
sys.path.insert(0,str(core.ROOT/'analysis/doubling_opportunity_v1/joint_v2'))
from financials import extract_facts,quarters as legacy_quarters
from verify_continuation import near


def reference_registry(payload):
    recent=payload['filings']['recent'];rows=list(zip(*(recent[k] for k in recent)))
    result={}
    for values in rows:
        row=dict(zip(recent,values));d=max(date.fromisoformat(row['filingDate']),date.fromisoformat(row['acceptanceDateTime'][:10]))
        row['available_date']=(d+timedelta(days=1)).isoformat();result[row['accessionNumber']]=row
    return result


def reference_state(payload,catalog,day):
    parsed=extract_facts(payload);selected={}
    for field,tags in [('revenue',TAGS['revenue']),('net_income',TAGS['net_income'])]:
        rows=[]
        for x in parsed[field]:
            f=catalog.get(x['accn'])
            if not f or f['available_date']>day or f['form']!=x['form'] or f['filingDate']!=x['filed'] or x['tag'] not in tags:continue
            rows.append(x)
        selected[field]=legacy_quarters(rows,day)
    revenue=selected['revenue'];net=selected['net_income']
    if not revenue:return None
    q=revenue[-1]
    if (date.fromisoformat(day)-date.fromisoformat(q['end'])).days>183:return None
    previous=[x for x in revenue if 0<(date.fromisoformat(q['start'])-date.fromisoformat(x['end'])).days<=7]
    if not previous:return None
    previous=previous[-1]
    def year(r):
        xs=[x for x in revenue if 350<=(date.fromisoformat(r['end'])-date.fromisoformat(x['end'])).days<=380 and abs(r['duration_days']-x['duration_days'])<=10]
        return xs[-1] if xs else None
    y=year(q);py=year(previous)
    if y is None or py is None or min(x['val'] for x in [q,y,previous,py])<=0:return None
    matched=[next((x for x in net if x['start']==r['start'] and x['end']==r['end']),None) for r in [q,y]]
    if any(x is None for x in matched):return None
    growth=np.log(q['val'])-np.log(y['val']);acc=growth-(np.log(previous['val'])-np.log(py['val']))
    margin=matched[0]['val']/q['val']-matched[1]['val']/y['val']
    refs=[q,y,previous,py,*matched];release=max(catalog[x['accn']]['available_date'] for r in refs for x in r['source_records'])
    return [max(-2.,min(2.,growth)),max(-2.,min(2.,acc)),max(-1.,min(1.,margin)),release],refs


def audit_features(features,new,receipts,sources,sessions):
    catalogs={s:reference_registry(sources[s]['submissions']) for s in PANEL};checks=rawchecks=known=0;seen=set();derived=set()
    byday={r['date']:r for r in receipts['days']};assert len(byday)==features.date.nunique()
    for day,group in features.groupby('date'):
        d=str(day.date());saved=byday[d];values=[];events={};available_dates=list(sessions[sessions<=day][-5:]);five={str(x.date()) for x in available_dates}
        for s in PANEL:
            result=reference_state(sources[s]['companyfacts'],catalogs[s],d)
            assert (result is not None)==(saved['states'][s] is not None)
            if result:
                state,refs=result;values.append(state);actual=saved['states'][s]
                near(actual['growth'],state[0]);near(actual['acceleration'],state[1]);near(actual['margin_change'],state[2]);assert actual['release_date']==state[3];checks+=4
                for name,q in actual['sources'].items():
                    if q['derived']:derived.add((s,q['start'],q['end'],q['filed'],name))
                    raw=[]
                    for x in q['sources']:
                        rows=sources[s]['companyfacts']['facts']['us-gaap'][x['tag']]['units']['USD'];r=rows[x['raw_index']];f=catalogs[s][x['accession']]
                        assert r['accn']==x['accession'] and r['start']==x['start'] and r['end']==x['end'] and r['filed']==x['filed'] and r['form']==x['form']
                        assert f['available_date']==x['available_date']<=d and r['end']<d and r['filed']==f['filingDate'];near(r['val'],x['value']);raw.append(r['val']);rawchecks+=1
                        seen.add((s,x['tag'],x['raw_index']))
                    assert len(raw)==(2 if q['derived'] else 1);near(q['value'],raw[0]-raw[1] if q['derived'] else raw[0])
            events[s]=[]
            for f in catalogs[s].values():
                if f['form'] not in ['8-K','8-K/A'] or '2.02' not in {x.strip() for x in f['items'].split(',')} or f['available_date']>d:continue
                next_session=next((x for x in sessions if str(x.date())>=f['available_date']),None)
                if next_session is not None and str(next_session.date()) in five:events[s].append(f['accessionNumber'])
            assert sorted(events[s])==sorted(saved['recent_event_accessions'][s])
        ready=len(values)==8;assert saved['ready']==ready
        assert new.loc[group.index,'financial_ready'].eq(ready).all()
        if not ready:assert new.loc[group.index,FINANCIAL].isna().all().all();continue
        known+=len(group);a=np.array([x[:3] for x in values]);ages=np.array([(date.fromisoformat(d)-date.fromisoformat(x[3])).days for x in values])
        expected=[sum(a[:,0])/8,sum(a[:,0]>0)/8,sum(a[:,1])/8,sum(a[:,1]>0)/8,sum(a[:,2])/8,sum(a[:,2]>0)/8,
            sum(bool(events[s]) for s in PANEL)/8,sum(math.exp(-n/63) for n in ages)/8]
        for at in group.index:
            for col,v in zip(FINANCIAL,expected):near(new.loc[at,col],v);checks+=1
    return {'passed':True,'ticks':len(features),'ready_ticks':known,'days':len(byday),'independent_feature_equalities':checks,
        'raw_fact_reference_checks':rawchecks,'unique_raw_facts_used':len(seen),'derived_quarter_receipts':len(derived),
        'all_facts_accession_linked_and_available_before_query':True,'independent_legacy_quarter_parser':True,
        'assumed_daily_availability_is_not_first_public_timestamp':True}
