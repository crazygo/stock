#!/usr/bin/env python3
"""Freeze full acquisition, native split math and released SEC features for v4."""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=PARENT.parents[1]
sys.path.insert(0,str(HERE));from market import normalized,price_features,labels,PRICE_FEATURES
from prepare_actions import load_audit
from actions import audited_events
sys.path.insert(0,str(PARENT));from context import CACHE
sys.path.insert(0,str(PARENT/'joint_v2'))
from financials import prepare_events,join_price_features,FEATURES as FIN_FEATURES
from prepare_joint import industry_features,INDUSTRY_FEATURES
import financials,prepare_joint
financials.CACHE=CACHE;prepare_joint.CACHE=CACHE
from scripts.model_history_calendar import calendar
from financial_source_profile import build_profiles
from reuse_financial_events import reuse as reuse_financial_events
VERSION='coverage_financial_split_heat_v4';FEATURES=PRICE_FEATURES+FIN_FEATURES+INDUSTRY_FEATURES

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def current_acquisition():
    pointer=json.loads((CACHE/'futu_daily_v4_latest.json').read_text());base=Path(pointer['path'])
    registration=json.loads((base/'registration.json').read_text());state=json.loads((base/'acquisition.json').read_text())
    if not state.get('terminal') or not state.get('full_registered_pool') or state.get('completed')!=len(registration['members']):
        raise RuntimeError('Full registered source acquisition must terminate; a download-order subset cannot be evaluated')
    for name in ['fundamentals','submissions']:
        s=json.loads((CACHE/(name+'_acquisition.json')).read_text())
        sec_registration=CACHE/'sec_registration.json';sr=json.loads(sec_registration.read_text())
        registered=set(sr['issuers']);observed=set(s['results'])
        if not s.get('terminal') or not s.get('full_registered_pool') or observed!=registered or s['completed']!=len(registered):
            raise RuntimeError('Full registered SEC source acquisition is not terminal or omits issuers')
        if s['registration_sha256']!=sha(sec_registration) or sr['catalog_sha256']!=sha(CACHE/'universe.json'):
            raise ValueError('SEC registration belongs to another catalog')
        code_snapshot=CACHE/'registered_code/acquire_sec.py'
        if sha(code_snapshot)!=sr['code_sha256']:raise ValueError('Actual registered SEC acquisition code snapshot mismatch')
        for r in s['results'].values():
            if r['status']=='available' and sha(Path(r['path']))!=r['sha256']:raise ValueError('Registered SEC raw source changed')
            if r['status']=='available' and r.get('original_sha256') and r['sha256']!=r['original_sha256']:
                raise ValueError('Local source changed during copy')
            if r.get('actual_acquisition_code_path') and sha(Path(r['actual_acquisition_code_path']))!=r['actual_acquisition_code_sha256']:
                raise ValueError('Actual resumed acquisition implementation snapshot changed')
    action_audit=HERE/'ACTION_AUDIT_READY.json'
    if not action_audit.exists() or not json.loads(action_audit.read_text()).get('ready_for_first_training'):
        raise RuntimeError('Confirmed native corporate-action omissions require a frozen source audit before first training')
    load_audit(base,validate_sources=True)
    if sha(base/'protocol_snapshot.md')!=registration['protocol_sha256']:raise ValueError('Source protocol snapshot SHA mismatch')
    return base,registration,state

def build():
    base,registration,state=current_acquisition();asof=registration['asof'];members=registration['members'];u=registration['universe']['stocks']
    cfg=json.loads((HERE/'config.json').read_text());audit,audit_ready=load_audit(base);available={}
    for t,r in audit['selected_sources'].items():
        if sha(Path(r['daily_path']))!=r['daily_sha256'] or sha(Path(r['rehab_path']))!=r['rehab_sha256']:raise ValueError('Native source SHA mismatch: '+t)
        available[t]=r
    if 'SPY' not in available:raise RuntimeError('Same-source SPY benchmark unavailable')
    cal=calendar('2023-01-01',min('2026-12-31',(pd.Timestamp(asof)+pd.Timedelta(days=65)).date().isoformat()))
    index=pd.DatetimeIndex([r['session_date'] for r in cal['sessions']]);expected=pd.Series(1.,index=index).rolling('183D').count()
    spy_raw=pd.read_parquet(available['SPY']['daily_path']);spy_events=pd.read_parquet(available['SPY']['rehab_path'])
    spy_events,_=audited_events(spy_events,audit['registry']['SPY'],asof)
    _,spy,_,_,_,_,_=normalized(spy_raw,index,spy_events,asof);spyret={n:spy.close/spy.close.shift(n)-1 for n in (20,60)}
    reuse_financial_events('2023-01-01',asof)
    financial=prepare_events('2023-01-01',asof)
    financial['scope']='complete registered SEC terminal source; missing and unsupported retained'
    financial_inputs={str((CACHE/'companyfacts'/f"CIK{r['cik']:010d}.json").resolve()):r['raw_sha256'] for r in financial['issuers']}
    financial_inputs.update({str(Path(r['path']).resolve()):sha(Path(r['path'])) for r in financial['issuers'] if r['path']})
    verified={};identity_missing=[]
    submissions={p.name:sha(p) for p in sorted((CACHE/'submissions').glob('*.json'))}
    financial_inputs.update({str((CACHE/'submissions'/name).resolve()):digest for name,digest in submissions.items()})
    for t,item in u.items():
        cik=item.get('cik');p=CACHE/'submissions'/f'CIK{cik:010d}.json' if cik else None
        live=json.loads(p.read_text()).get('tickers',[]) if p and p.exists() else []
        if t in [x.replace('-','.') for x in live]:verified[t]=item
        else:
            verified[t]={k:v for k,v in item.items() if k!='cik'}
            if cik:identity_missing.append({'ticker':t,'cik':cik})
    sources=[HERE/'reuse_financial_events.py',HERE/'financial_source_profile.py',HERE/'catalog.py',HERE/'PROTOCOL.md',HERE/'context.py',HERE/'acquire_sec.py',HERE/'acquire_daily.py',HERE/'recover_sources.py',HERE/'market.py',HERE/'actions.py',HERE/'prepare_actions.py',HERE/'ACTION_RULES_ADDENDUM.md',HERE/'primary_split_supplements.json',Path(__file__),HERE/'config.json',HERE/'TRAINING_PROTOCOL.md',CACHE/'sec_registration.json',CACHE/'registered_code/acquire_sec.py',CACHE/'fundamentals_acquisition.json',CACHE/'submissions_acquisition.json',
             PARENT/'joint_v2/financials.py',PARENT/'joint_v2/prepare_joint.py',PARENT/'joint_v2/model.py',PARENT/'model.py',PARENT/'data.py',
             ROOT/'scripts/model_history_calendar.py']
    hashes={str(p.resolve()):sha(p) for p in sources}
    for name in ['fundamentals','submissions']:
        s=json.loads((CACHE/(name+'_acquisition.json')).read_text())
        for r in s['results'].values():
            if r.get('actual_acquisition_code_path'):hashes[r['actual_acquisition_code_path']]=r['actual_acquisition_code_sha256']
    fingerprint=hashlib.sha256(json.dumps([registration,available,audit_ready,financial['issuers'],financial_inputs,hashes],sort_keys=True).encode()).hexdigest()
    out=base/'derived'/fingerprint[:16];out.mkdir(parents=True,exist_ok=True)
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        cached=json.loads(manifest_path.read_text())
        for path,digest in cached['derived_hashes'].items():
            if sha(Path(path))!=digest:raise ValueError('Frozen derived dataset SHA mismatch: '+path)
        (HERE/'latest_dataset.json').write_text(json.dumps({'path':str(out.resolve()),'key':fingerprint,'asof':asof},indent=2))
        return pd.read_parquet(out/'dataset.parquet'),cached
    rows=[];panels=[];coverage=[];actions=[]
    for n,t in enumerate(members):
        meta=available.get(t)
        if meta is None:
            coverage.append({'ticker':t,'source_status':audit['source_outcomes'][t]['selected_status'],'price_eligible':False});continue
        raw=pd.read_parquet(meta['daily_path']);events=pd.read_parquet(meta['rehab_path'])
        events,_=audited_events(events,audit['registry'][t],asof)
        f,adjusted,unknown,issues=price_features(raw,index,events,asof,spyret,expected)
        rawq,_,factor,_,_,_,original=normalized(raw,index,events,asof)
        observed=(index<=pd.Timestamp(asof))&original.close.notna();real=index[observed]
        panel=original.loc[real].copy()
        for col in ['open','high','low','close']:panel['native_'+col]=panel[col];panel[col]=adjusted.loc[real,col]
        panel['native_volume']=panel.volume;panel.volume=adjusted.loc[real,'volume']
        panel['split_basis_multiplier']=factor[observed];panel['ticker']=t;panel['date']=real;panels.append(panel.reset_index(drop=True))
        coverage.append({'ticker':t,'source_status':'available','native_rows_2023_onward':int(observed.sum()),
            'first_native_date':real.min().date().isoformat() if len(real) else None,'last_native_date':real.max().date().isoformat() if len(real) else None,
            'latest_official_session_covered':bool(len(real) and real.max()==pd.Timestamp(asof)),
            'invalid_native_bars':int(f.invalid_native_bar.sum()),'unsupported_actions':len(issues),'price_eligible':bool(f.loc[pd.Timestamp(asof),'eligible'])})
        coverage[-1].update(audit['source_outcomes'][t]);coverage[-1].update(primary_split_records=len(audit['registry'][t]['verified_splits']),source_unknown_events=len(audit['registry'][t]['unknown_events']),whole_history_unknown=audit['registry'][t]['whole_history_unknown'])
        actions.extend({'ticker':t,**x} for x in issues)
        if t=='SPY':continue
        for h in cfg['horizons']:f=f.join(labels(adjusted,index,unknown,asof,h,cfg['reference_cost']))
        f['ticker']=t;f['date']=index
        rows.append(f.loc[observed].reset_index(drop=True))
        if n%250==0:print('v4 native features',n,'/',len(members),flush=True)
    price=pd.concat(rows,ignore_index=True);panel=pd.concat(panels,ignore_index=True)
    print('v4 released financial join',len(price),'real stock days',flush=True)
    dataset=industry_features(join_price_features(price,verified,financial),verified)
    dataset['security_description_confirmed']=dataset.ticker.map(lambda t:bool(u[t].get('security_description_confirmed',False)))
    dataset['joint_signal_eligible'] &= dataset.security_description_confirmed
    dataset.to_parquet(out/'dataset.parquet',compression='zstd',compression_level=7,index=False)
    panel.to_parquet(out/'panel.parquet',compression='zstd',compression_level=7,index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    manifest={'version':VERSION,'key':fingerprint,'asof':asof,'path':str(out.resolve()),'dataset_path':str((out/'dataset.parquet').resolve()),
        'panel_path':str((out/'panel.parquet').resolve()),'registered_count':len(members),'equity_candidates':len(u),
        'derived_hashes':{str(p.resolve()):sha(p) for p in [out/'dataset.parquet',out/'panel.parquet',out/'coverage.csv']},
        'source_available_count':len(available),'source_unavailable_count':len(members)-len(available),
        'native_stock_day_rows':len(price),'price_eligible_rows':int(price.eligible.sum()),'joint_policy_eligible_rows':int(dataset.joint_signal_eligible.sum()),
        'unconfirmed_current_ticker_mappings':identity_missing,'unsupported_action_cases':actions,'features':FEATURES,
        'source_hashes':hashes,'registration':registration,'acquisition_state':{k:v for k,v in state.items() if k!='results'},
        'native_sources':available,'action_source_audit':audit_ready,'action_source_summary':{k:audit[k] for k in ['article_registered_count','article_status_counts','reconciliation_counts','unknown_gap_count','whole_history_unknown_tickers']},
        'SEC_financial_event_manifest':financial,'SEC_submissions_hashes':submissions,'financial_input_hashes':financial_inputs,'calendar':cal,
        'financial_source_profiles':build_profiles(CACHE,asof),
        'regular_hours_label_verified':False,'membership_mode':'current_retrospective_not_PIT','new_independent_effect_verification':False}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(HERE/'latest_dataset.json').write_text(json.dumps({'path':str(out.resolve()),'key':fingerprint,'asof':asof},indent=2))
    return dataset,manifest

if __name__=='__main__':
    d,m=build();print(json.dumps({k:m[k] for k in ['key','asof','native_stock_day_rows','price_eligible_rows','joint_policy_eligible_rows']},indent=2))
