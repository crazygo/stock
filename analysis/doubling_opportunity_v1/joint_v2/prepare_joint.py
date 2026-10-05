"""Freeze complete registered SEC coverage and join only released financial states."""
from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(HERE))
from financials import prepare_events,join_price_features,FEATURES as FIN_FEATURES
sys.path.insert(0,str(PARENT))
from data import CACHE,FEATURES as PRICE_FEATURES

VERSION='joint_v2_financial_market_heat'
INDUSTRY_FEATURES=['industry_ret20','industry_rs20','industry_volume_ratio','industry_members','industry_missing']
FEATURES=PRICE_FEATURES+FIN_FEATURES+INDUSTRY_FEATURES


def coverage_state():
    states={}
    for name in ['fundamentals','submissions']:
        p=CACHE/(name+'_acquisition.json')
        j=json.loads(p.read_text()) if p.exists() else {}
        states[name]={k:v for k,v in j.items() if k!='results'}
    return states


def industry_features(d,universe):
    mapping={}
    for p in (CACHE/'submissions').glob('*.json'):
        j=json.loads(p.read_text());sic=str(j.get('sic') or '')
        if len(sic)==4 and sic.isdigit():mapping[int(j['cik'])]=sic[:2]
    q=d.copy();q['industry_code']=q.ticker.map(lambda t:mapping.get(universe.get(t,{}).get('cik')))
    valid=q.eligible&q.industry_code.notna()
    grouped=q[valid].groupby(['date','industry_code'])
    table=grouped.agg(industry_ret20=('ret20','median'),industry_rs20=('rs20','median'),
                      industry_volume_ratio=('volume_ratio','median'),industry_members=('ticker','count')).reset_index()
    q=q.merge(table,on=['date','industry_code'],how='left',validate='many_to_one')
    q['industry_missing']=(q.industry_members.fillna(0)<5).astype(float)
    q.loc[q.industry_missing==1,['industry_ret20','industry_rs20','industry_volume_ratio']]=np.nan
    q['industry_members']=q.industry_members.fillna(0)
    q['joint_signal_eligible']=(q.eligible&(q.quality_status_at_event=='growth_and_operating_health_checks_pass')&
        (q.ret20>.1)&(q.rs20>.05)&(q.volume_ratio>=1.1)&(q.industry_missing==0)&
        (q.industry_rs20>.03)&(q.industry_volume_ratio>=1.05))
    return q


def prepare():
    states=coverage_state()
    if not all(s.get('terminal') for s in states.values()):
        raise RuntimeError('Both registered SEC acquisition jobs must terminate before freezing joint evaluation coverage')
    u=json.loads((CACHE/'universe.json').read_text())['stocks']
    verified_u={};unconfirmed=[]
    for ticker,item in u.items():
        cik=item.get('cik');p=CACHE/'submissions'/f'CIK{cik:010d}.json' if cik else None
        tickers=json.loads(p.read_text()).get('tickers',[]) if p and p.exists() else []
        match=ticker in [t.replace('-','.') for t in tickers]
        verified_u[ticker]=dict(item) if match else {k:v for k,v in item.items() if k!='cik'}
        if cik and not match:unconfirmed.append({'ticker':ticker,'cik':cik,'reason':'current_SEC_ticker_not_confirmed'})
    d=pd.read_parquet(CACHE/'dataset.parquet');dataset_key=json.loads((CACHE/'dataset_key.json').read_text())['key']
    start=d.date.min().date().isoformat();end=d.date.max().date().isoformat()
    manifest=prepare_events(start,end)
    manifest['scope']='frozen registered acquisition terminal snapshot; missing and unmapped issuers retained'
    manifest['acquisition']=states;manifest['frozen_at']=datetime.now(timezone.utc).isoformat()
    manifest['unmapped_tickers']=[t for t,v in u.items() if not v.get('cik')]
    manifest['unconfirmed_current_ticker_mappings']=unconfirmed
    registered={v['cik'] for v in u.values() if v.get('cik')}
    found={v['cik'] for v in manifest['issuers'] if v['source_valid']}
    manifest['registered_issuers_without_facts']=sorted(registered-found)
    sources=[HERE/'financials.py',Path(__file__),HERE/'model.py',HERE/'PROTOCOL.md',HERE/'train_financial.py',PARENT/'model.py']
    hashes={str(p.relative_to(HERE)) if p.parent==HERE else '../'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    submissions={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((CACHE/'submissions').glob('*.json'))}
    manifest['submissions_hashes']=submissions;manifest['source_hashes']=hashes;manifest['price_dataset_key']=dataset_key
    key=hashlib.sha256(json.dumps([dataset_key,manifest['issuers'],submissions,hashes],sort_keys=True).encode()).hexdigest()
    target=CACHE/f'{VERSION}_{key[:16]}.parquet'
    if target.exists():joint=pd.read_parquet(target)
    else:
        joint=industry_features(join_price_features(d,verified_u,manifest),verified_u)
        joint.to_parquet(target,compression='zstd',compression_level=7,index=False)
    manifest.update(key=key,path=str(target),version=VERSION,features=FEATURES,
       price_eligible_rows=int(joint.eligible.sum()),joint_policy_eligible_rows=int(joint.joint_signal_eligible.sum()),
       financial_missing_ticker_count=int(joint[joint.date==joint.date.max()].fin_statement_missing.sum()),
       sic_mode='current_SEC_SIC_retrospective_not_PIT',regular_hours_labels_verified=False)
    p=CACHE/f'{VERSION}_{key[:16]}_manifest.json';p.write_text(json.dumps(manifest,indent=2))
    return joint,manifest

if __name__=='__main__':
    d,m=prepare();print(json.dumps({k:m[k] for k in ['version','key','path','price_eligible_rows','joint_policy_eligible_rows','financial_missing_ticker_count']},indent=2))
