"""Accession-linked financial history; next-calendar-day availability is an assumption."""
from pathlib import Path
from datetime import date,timedelta
import sys,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v32'))
from hierarchical_error import core,np,pd,read,dump,sha

PANEL=['NVDA','AMD','AVGO','MU','AMAT','LRCX','KLAC','INTC']
STATE=['revenue_growth','growth_breadth','growth_acceleration','acceleration_breadth','margin_improvement','margin_breadth']
EVENT=['recent_earnings_share','disclosure_freshness']
FINANCIAL=STATE+EVENT
TAGS={'revenue':['RevenueFromContractWithCustomerExcludingAssessedTax','Revenues','SalesRevenueNet','RevenueFromContractWithCustomerIncludingAssessedTax'],
      'net_income':['NetIncomeLoss','ProfitLoss']}
SOURCE=Path(__file__).resolve().parent


def ordinal(d):return date.fromisoformat(d).toordinal()


def filings(payload):
    p=payload['filings']['recent'];out={}
    assert len(set(map(len,p.values())))==1
    for i,a in enumerate(p['accessionNumber']):
        r={k:v[i] for k,v in p.items()};d=max(r['filingDate'],r['acceptanceDateTime'][:10])
        r['available_date']=(date.fromisoformat(d)+timedelta(days=1)).isoformat()
        assert a not in out;out[a]=r
    return out


def load_sources():
    manifest=read(SOURCE/'evidence/financial_input_manifest.json');out={}
    assert list(manifest)==PANEL
    for s,v in manifest.items():
        inputs={}
        for kind in ['companyfacts','submissions']:
            p=core.ROOT/v[kind]['path'];assert sha(p)==v[kind]['sha256'];j=read(p);assert int(j['cik'])==v['cik'];inputs[kind]=j
        assert s in inputs['submissions']['tickers']
        out[s]=inputs
    return out,manifest


def extract(payload,registry):
    out={f:[] for f in TAGS};discarded={}
    for field,tags in TAGS.items():
        for priority,tag in enumerate(tags):
            for i,r in enumerate(payload['facts'].get('us-gaap',{}).get(tag,{}).get('units',{}).get('USD',[])):
                if r.get('form') not in ['10-K','10-Q','10-K/A','10-Q/A'] or not r.get('start'):continue
                if not isinstance(r.get('val'),(int,float)) or not math.isfinite(r['val']):continue
                if r.get('accn') not in registry:
                    discarded['accession_not_in_cached_catalog']=discarded.get('accession_not_in_cached_catalog',0)+1;continue
                f=registry[r['accn']]
                if f['form']!=r['form'] or f['filingDate']!=r['filed']:
                    discarded['filing_metadata_mismatch']=discarded.get('filing_metadata_mismatch',0)+1;continue
                n=ordinal(r['end'])-ordinal(r['start'])+1
                if not 60<=n<=400:continue
                out[field].append({'start':r['start'],'end':r['end'],'value':float(r['val']),'filed':r['filed'],
                    'available_date':f['available_date'],'tag':tag,'priority':priority,'days':n,
                    'accession':r['accn'],'form':r['form'],'raw_index':i})
    return out,discarded


def quarters(rows,day):
    selected={}
    for r in rows:
        if r['available_date']>day or r['end']>=day:continue
        key=(r['start'],r['end']);rank=(r['filed'],-r['priority'])
        prior=selected.get(key)
        if prior is None or rank>(prior['filed'],-prior['priority']):selected[key]=r
        elif rank==(prior['filed'],-prior['priority']) and r['value']!=prior['value']:
            raise ValueError('Conflicting same-period same-priority visible facts')
    visible=list(selected.values());out={}
    for r in visible:
        if r['days']<=120:
            if r['end'] not in out or r['filed']>out[r['end']]['filed']:
                out[r['end']]={**r,'derived':False,'sources':[r]}
    for r in visible:
        if r['days']<=120:continue
        candidates=[x for x in visible if x['start']==r['start'] and x['tag']==r['tag'] and 60<=ordinal(r['end'])-ordinal(x['end'])<=120]
        if not candidates:continue
        prev=max(candidates,key=lambda x:(x['end'],x['filed']))
        start=(date.fromisoformat(prev['end'])+timedelta(days=1)).isoformat()
        if r['end'] not in out:
            out[r['end']]={**r,'start':start,'days':ordinal(r['end'])-ordinal(prev['end']),
                'value':r['value']-prev['value'],'filed':max(r['filed'],prev['filed']),
                'available_date':max(r['available_date'],prev['available_date']),'derived':True,'sources':[r,prev]}
    return sorted(out.values(),key=lambda x:x['end'])


def prior_year(rows,q):
    xs=[r for r in rows if 350<=ordinal(q['end'])-ordinal(r['end'])<=380 and abs(q['days']-r['days'])<=10]
    return max(xs,key=lambda r:r['end']) if xs else None


def state(parsed,day):
    revenue=quarters(parsed['revenue'],day);net=quarters(parsed['net_income'],day)
    if not revenue:return None,'no_revenue'
    now=revenue[-1];year=prior_year(revenue,now)
    previous=[r for r in revenue if 1<=ordinal(now['start'])-ordinal(r['end'])<=7]
    prev=max(previous,key=lambda r:r['end']) if previous else None;prev_year=prior_year(revenue,prev) if prev else None
    if year is None or prev is None or prev_year is None:return None,'quarter_comparisons_missing'
    if min(x['value'] for x in [now,year,prev,prev_year])<=0:return None,'nonpositive_revenue'
    if ordinal(day)-ordinal(now['end'])>183:return None,'stale_period'
    op=next((r for r in net if (r['start'],r['end'])==(now['start'],now['end'])),None)
    old_op=next((r for r in net if (r['start'],r['end'])==(year['start'],year['end'])),None)
    if op is None or old_op is None:return None,'net_margin_missing'
    g=math.log(now['value']/year['value']);a=g-math.log(prev['value']/prev_year['value'])
    margin=op['value']/now['value']-old_op['value']/year['value']
    refs={'revenue':now,'year_revenue':year,'previous_revenue':prev,'previous_year_revenue':prev_year,'net_income':op,'year_net_income':old_op}
    return {'growth':float(np.clip(g,-2,2)),'acceleration':float(np.clip(a,-2,2)),'margin_change':float(np.clip(margin,-1,1)),
        'latest_period':now['end'],'release_date':max(x['available_date'] for q in refs.values() for x in q['sources']),
        'sources':refs},None


def build(features,sources,sessions):
    registries={s:filings(sources[s]['submissions']) for s in PANEL};parsed={};excluded={}
    for s in PANEL:parsed[s],excluded[s]=extract(sources[s]['companyfacts'],registries[s])
    days={};receipts=[]
    for d in sorted(features.date.unique()):
        day=str(pd.Timestamp(d).date());prior=list(sessions[sessions<=d][-5:]);eligible_dates={str(x.date()) for x in prior}
        states={};reason={};event_counts={}
        for s in PANEL:
            v,why=state(parsed[s],day);states[s]=v
            if why:reason[s]=why
            # A disclosure during a non-session becomes observable at the next session.
            ev=[]
            for f in registries[s].values():
                if f['form'] not in ['8-K','8-K/A'] or '2.02' not in [x.strip() for x in f['items'].split(',')]:continue
                if f['available_date']>day:continue
                dates=sessions[sessions>=f['available_date']]
                if len(dates) and str(dates[0].date()) in eligible_dates:ev.append(f['accessionNumber'])
            event_counts[s]=ev
        ready=not reason;values={c:np.nan for c in FINANCIAL}
        if ready:
            growth=np.array([states[s]['growth'] for s in PANEL]);acc=np.array([states[s]['acceleration'] for s in PANEL]);margin=np.array([states[s]['margin_change'] for s in PANEL])
            ages=np.array([ordinal(day)-ordinal(states[s]['release_date']) for s in PANEL]);assert (ages>=0).all()
            values=dict(zip(FINANCIAL,[growth.mean(),np.mean(growth>0),acc.mean(),np.mean(acc>0),margin.mean(),np.mean(margin>0),
                np.mean([bool(event_counts[s]) for s in PANEL]),np.exp(-ages/63.).mean()]))
        days[day]={'financial_ready':ready,**values}
        receipts.append({'date':day,'ready':ready,'reasons':reason,'states':states,'recent_event_accessions':event_counts})
    rows=[{'at':at,**days[str(row.date.date())]} for at,row in features.iterrows()]
    return pd.DataFrame(rows).set_index('at'),{'days':receipts,'excluded_raw_rows':excluded,
        'availability':'assumed next ET calendar day after max(filingDate, acceptance timestamp date); not observed first-public or historical receipt time'}
