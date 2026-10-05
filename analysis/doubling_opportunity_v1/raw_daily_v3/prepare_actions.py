#!/usr/bin/env python3
"""Freeze primary action reconciliation and recovery only after all registered sources terminate."""
from __future__ import annotations
import argparse,collections,fcntl,hashlib,json,os,re,sys,time
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(PARENT));from data import CACHE
sys.path.insert(0,str(HERE));from actions import parse_html,combine_events,unresolved_overnight_gaps
from scripts.model_history_calendar import calendar

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,j):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def company_name(name):
    name=re.split(r'\s+(?:-\s+)?(?:Class\s+[A-Z]\s+)?(?:Common Stock|Ordinary Shares|Common Shares|American Depositary|American Depository)',name,flags=re.I)[0]
    words=re.findall(r'[a-z0-9]+',name.lower().replace('&',' and ').replace("'",'').replace('’',''))
    while words and words[-1] in {'inc','incorporated','corp','corporation','ltd','limited','plc','co','company'}:words.pop()
    return ' '.join(words)
def title_company(title):
    parts=re.split(r'\bfor\s+',title,flags=re.I)
    return company_name(parts[-1].split('(')[0]) if len(parts)>1 else ''

def source_state():
    pointer=json.loads((CACHE/'futu_daily_v3_latest.json').read_text());base=Path(pointer['path'])
    state=json.loads((base/'acquisition.json').read_text());registration=json.loads((base/'registration.json').read_text())
    recovery_file=base/'recovery/acquisition.json';recovery=json.loads(recovery_file.read_text()) if recovery_file.exists() else {}
    action_base=CACHE/'corporate_actions_v3';action_file=action_base/'acquisition.json'
    action=json.loads(action_file.read_text()) if action_file.exists() else {}
    correction_file=action_base/'corrections/acquisition.json';correction=json.loads(correction_file.read_text()) if correction_file.exists() else {}
    complete=state.get('terminal') and state.get('full_registered_pool') and state.get('completed')==len(registration['members'])
    return base,state,registration,recovery,action_base,action,bool(complete and recovery.get('terminal') and action.get('terminal') and correction.get('terminal'))

def load_audit(base=None,validate_sources=False):
    ready=json.loads((HERE/'ACTION_AUDIT_READY.json').read_text())
    if not ready.get('ready_for_first_training') or (base is not None and ready['source_snapshot_path']!=str(Path(base).resolve())):
        raise ValueError('Action audit not ready for the selected source snapshot')
    if sha(ready['path'])!=ready['sha256']:raise ValueError('Frozen action audit SHA mismatch')
    audit=json.loads(Path(ready['path']).read_text())
    for p,digest in audit['rule_hashes'].items():
        if sha(p)!=digest:raise ValueError('Action rules changed after audit: '+p)
    for p,digest in audit['detail_hashes'].items():
        if sha(p)!=digest:raise ValueError('Action detail artifact changed after audit: '+p)
    if validate_sources:
        for p,digest in audit['all_source_hashes'].items():
            if sha(p)!=digest:raise ValueError('Source action/recovery SHA mismatch: '+p)
    return audit,ready

def prepare():
    base,state,reg,recovery,action_base,action,complete=source_state()
    if not complete:raise RuntimeError('Registered full daily source, recovery, and primary articles must all terminate before audit')
    if (HERE/'ACTION_AUDIT_READY.json').exists():
        try:
            audit,ready=load_audit(base,validate_sources=True)
            print('reusing unchanged complete source action audit',ready['key'],flush=True);return
        except (ValueError,FileNotFoundError):pass
    asof=reg['asof'];members=reg['members'];u=reg['universe']['stocks']
    ar=json.loads((action_base/'registration.json').read_text())
    link_reg=HERE/'ARCHIVE_LINK_CORRECTIONS.json';links=json.loads(link_reg.read_text());correction_path=action_base/'corrections/acquisition.json';corrections=json.loads(correction_path.read_text())
    if corrections['registration_sha256']!=sha(link_reg) or set(corrections['results'])!={r['compressed_url'] for r in links['corrections']}:
        raise ValueError('Actual-link correction registration mismatch')
    if len(action['results'])!=ar['registered_count'] or set(action['results'])!={x['url'] for x in ar['articles']}:raise ValueError('Article registration is not fully represented')
    candidates=[t for t,r in state['results'].items() if r.get('status')!='available' or not r.get('current_session_covered') or not r.get('raw_rows')]
    if set(recovery['registered_recovery_candidates'])!=set(candidates) or set(recovery['results'])!=set(candidates):raise ValueError('Recovery omitted a registered empty/stale/failing source')
    if sha(base/'protocol_snapshot.md')!=reg['protocol_sha256']:raise ValueError('Original source protocol snapshot mismatch')
    if sha(HERE/'ACTION_AUDIT_PROTOCOL.md')!=ar['protocol_sha256'] or sha(HERE/'NASDAQ_ARCHIVE_REGISTRATION.json')!=ar['registration_sha256']:raise ValueError('Action source preregistration changed')
    all_hashes={str(p.resolve()):sha(p) for p in [base/'registration.json',base/'acquisition.json',base/'recovery/acquisition.json',action_base/'registration.json',action_base/'acquisition.json',correction_path]}
    selected={};source_outcomes={}
    for t in members:
        original=state['results'][t];replacement=recovery['results'].get(t)
        for r in [original,replacement]:
            if r is None:continue
            for kind in ['daily','rehab']:
                if r.get(kind+'_path'):
                    p=r[kind+'_path'];digest=r[kind+'_sha256']
                    if sha(p)!=digest:raise ValueError('Source SHA mismatch: '+t+' '+kind)
                    all_hashes[p]=digest
        choice=replacement if replacement and replacement.get('status')=='available' else original
        valid=choice.get('status')=='available' and choice.get('raw_rows',0)>0
        if valid:selected[t]=choice
        source_outcomes[t]={'original_status':original.get('status'),'original_rows':original.get('raw_rows'),'recovery_status':replacement.get('status') if replacement else 'not_required',
                            'selected_status':choice.get('status') if valid else 'no_valid_native_rows','current_session_covered':bool(valid and choice.get('current_session_covered'))}
    if 'SPY' not in selected:raise RuntimeError('Benchmark unavailable after recovery')
    parsed=[]
    exceptions={r['url']:r for r in links['exception_headlines']}
    for r in action['results'].values():
        original=r
        if r['url'] in corrections['results']:
            r=corrections['results'][r['url']]|{'url':corrections['results'][r['url']]['fetch_url'],'original_request':original}
        if r['status']!='available':
            headline=exceptions.get(r['url'],{})
            parsed.append({**r,'status':'source_article_unavailable','tickers':headline.get('tickers',r.get('tickers',[])),
                'published_at':headline.get('published_at',r.get('published_at')),'title':headline.get('kind',r.get('kind',''))+' source terms unavailable'});continue
        if sha(r['path'])!=r['sha256']:raise ValueError('Primary article SHA mismatch')
        all_hashes[r['path']]=r['sha256'];parsed.append(parse_html(Path(r['path']).read_text(),r['url'])|{'source_path':r['path'],'source_sha256':r['sha256'],
            'original_compressed_request_url':original['url'] if r is not original else None,'actual_archive_url':r.get('actual_archive_url')})
    supplemental=json.loads((HERE/'primary_split_supplements.json').read_text())['events'];registry={t:{'verified_splits':[],'unknown_events':[],'whole_history_unknown':False} for t in members}
    aliases={}
    for r in supplemental:
        if u.get(r['ticker'],{}).get('cik')!=r['cik']:raise ValueError('Supplement primary CIK does not match current registered issuer')
        if r.get('historical_symbol'):aliases[r['historical_symbol']]=r['ticker']
        if r['ex_date']<=asof and r['published_at']<=asof:registry[r['ticker']]['verified_splits'].append(r)
    unmatched=[];matched_records=[]
    risky=re.compile(r'split|ratio change|spin[- ]off|stock dividend|special(?: cash)? dividend|merger|reorgani|bankrupt|distribution|exchange offer|business.combination',re.I)
    for record in parsed:
        if not risky.search(record.get('title','')):continue
        if record.get('published_at') and record['published_at']>asof:continue
        affected=record.get('tickers',[])+record.get('explicit_table_symbols',[])
        targets=list(dict.fromkeys(aliases.get(t,t) for t in affected if aliases.get(t,t) in registry))
        if not targets:unmatched.append({k:v for k,v in record.items() if k!='source_text'});continue
        for t in targets:
            r={k:v for k,v in record.items() if k!='source_text'};r['ticker']=t
            explicit=record['status']=='explicit_single_security_split'
            exact_name=bool(t in u and title_company(record['title'])==company_name(u[t]['name']))
            known_alias=any(aliases.get(symbol)==t for symbol in record['tickers'])
            if explicit and (exact_name or known_alias):
                if record['ex_date']<=asof:registry[t]['verified_splits'].append(r|{'status':'verified_split','identity_basis':'SEC_CIK_explicit_alias' if known_alias else 'exact_normalized_current_directory_company_and_symbol'})
            else:
                dates=record.get('candidate_effective_dates',[]) if record.get('effective_date_resolved',True) else []
                if len(dates)==1:
                    if dates[0]<=asof:registry[t]['unknown_events'].append(r|{'ex_date':dates[0],'status':record['status'] if not explicit else 'current_symbol_company_identity_mismatch'})
                else:registry[t]['whole_history_unknown']=True
            matched_records.append(r|{'exact_current_name_match':exact_name,'known_CIK_alias':known_alias})
    cal=calendar('2023-01-01',min('2026-12-31',(pd.Timestamp(asof)+pd.Timedelta(days=65)).date().isoformat()));index=pd.DatetimeIndex([r['session_date'] for r in cal['sessions']])
    gaps=[];reconciliation=[]
    for n,(t,r) in enumerate(selected.items()):
        raw=pd.read_parquet(r['daily_path']);native=pd.read_parquet(r['rehab_path']);events,audit=combine_events(native,registry[t]['verified_splits'],asof)
        reconciliation.extend({'ticker':t,**x} for x in audit)
        found=unresolved_overnight_gaps(raw,index,events,asof,limit=1.8)
        registry[t]['unknown_events'].extend(found);gaps.extend({'ticker':t,**x} for x in found)
        if n%500==0:print('whole pool action source audit',n,'/',len(selected),flush=True)
    rules=[HERE/'actions.py',HERE/'market.py',Path(__file__),HERE/'ACTION_RULES_ADDENDUM.md',HERE/'primary_split_supplements.json',link_reg,
           PARENT.parents[1]/'scripts/model_history_calendar.py']
    rule_hashes={str(p.resolve()):sha(p) for p in rules}
    key=hashlib.sha256(json.dumps([all_hashes,rule_hashes,asof],sort_keys=True).encode()).hexdigest();out=base/'action_audits'/key[:16];out.mkdir(parents=True,exist_ok=True)
    for name,data in [('articles.json',parsed),('reconciliation.json',reconciliation),('unmatched_actions.json',unmatched),('matched_actions.json',matched_records),('unresolved_gaps.json',gaps)]:save(out/name,data)
    audit={'version':'primary_action_source_audit_v3_1','key':key,'asof':asof,'source_snapshot_path':str(base.resolve()),'registered_count':len(members),
      'article_registered_count':ar['registered_count'],'article_status_counts':dict(collections.Counter(x['status'] for x in parsed)),
      'reconciliation_counts':dict(collections.Counter(x['reconciliation'] for x in reconciliation)),
      'unknown_gap_count':len(gaps),'whole_history_unknown_tickers':[t for t,r in registry.items() if r['whole_history_unknown']],
      'selected_sources':selected,'source_outcomes':source_outcomes,'registry':registry,'rule_hashes':rule_hashes,'all_source_hashes':all_hashes,
      'detail_hashes':{str(p.resolve()):sha(p) for p in out.glob('*.json')},'complete_US_corporate_action_coverage':False,'historical_PIT_identity_verified':False,
      'recovery_terminal':True,'full_registered_source_terminal':True,'created_at':datetime.now(timezone.utc).isoformat()}
    path=out/'audit.json';save(path,audit);ready={'ready_for_first_training':True,'path':str(path.resolve()),'sha256':sha(path),'source_snapshot_path':str(base.resolve()),'key':key,'asof':asof}
    save(HERE/'ACTION_AUDIT_READY.json',ready);save(CACHE/'raw_daily_v3_action_audit_status.json',{'status':'complete_source_audit','path':str(path.resolve()),'key':key,'registered_count':len(members),'unknown_gap_count':len(gaps)})
    print(json.dumps({k:audit[k] for k in ['key','article_status_counts','reconciliation_counts','unknown_gap_count','whole_history_unknown_tickers']},indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--wait',action='store_true');ap.add_argument('--wait-timeout-minutes',type=int,default=240);args=ap.parse_args()
    lock=(CACHE/'raw_daily_v3_action_audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=time.monotonic()+args.wait_timeout_minutes*60
    waited=False
    while True:
        base,state,reg,recovery,ab,action,ready=source_state()
        if state.get('error') or action.get('error'):raise RuntimeError('Registered source job failed; inspect actual source state')
        if ready:break
        if not args.wait:raise RuntimeError('Sources not ready; use --wait without evaluating a subset')
        if time.monotonic()>deadline:raise TimeoutError('Bounded source audit wait expired')
        save(CACHE/'raw_daily_v3_action_audit_status.json',{'status':'waiting_for_full_sources','daily_completed':state.get('completed'),'daily_registered':len(reg['members']),
            'article_completed':action.get('completed'),'article_registered':action.get('registered_count'),'recovery_terminal':recovery.get('terminal',False),'updated_at':datetime.now(timezone.utc).isoformat()})
        print('action audit waiting full sources',state.get('completed'),action.get('completed'),recovery.get('terminal'),flush=True);time.sleep(45)
        waited=True
    if waited:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve())])
    prepare()

if __name__=='__main__':main()
