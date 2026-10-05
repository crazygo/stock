#!/usr/bin/env python3
"""Correct entire-directory equity classification and unique official ticker->CIK mapping."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,re,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import urlopen
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from context import CACHE,OLD_CACHE

COMMON=re.compile(r'\b(?:common (?:stock|stcok|new|shares?)|ordinary shares?|ord shares?|class [a-z] (?:common|capital stock|shares|(?:subordinate |limited )?voting shares)|(?:subordinate |limited )voting shares|(?:new york|ny) (?:registry|registered) shares|registered shares|american (?:depositary|depository) (?:shares?|receipts?)|adrs?|adss?)\b|\bcommon\s*$',re.I)
DERIVATIVE=re.compile(r'\b(?:warrants?|rights?|units?|debentures?|notes?|bonds?|etns?|certificates?|certs?|corts|strats)\b|\b(?:preferred|preference|pfd)\b',re.I)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def classify(row,sec_match=False):
    name=row.get('Security Name','');desc=' '.join(name.split(' - ',1)[-1].split())
    if row.get('Test Issue')=='Y':return 'excluded','test_issue'
    if row.get('ETF')=='Y' or row.get('NextShares')=='Y':return 'excluded','fund_etf'
    if '$' in row.get('Symbol',''):return 'excluded','directory_preferred_series_symbol'
    if re.search(r'\b(?:ordinary|ord) shares?\b',desc,re.I) and re.search(r'\b(?:pfd|preferred|preference)\b',desc,re.I):
        return 'review','conflicting_ordinary_and_preferred_description'
    if DERIVATIVE.search(desc):return 'excluded','non_common_security'
    if re.search(r'\b(?:closed end fund|fund|etn)\b',desc,re.I):return 'excluded','investment_fund_or_note'
    if re.search(r'\bdepositary shares\b',desc,re.I) and not re.search(r'\bamerican (?:depositary|depository)\b',desc,re.I):
        return 'review','depositary_underlying_security_not_confirmed'
    if COMMON.search(desc):return 'candidate','explicit_common_or_ADS_or_registered_voting_share'
    if sec_match:return 'review','SEC_listed_ticker_security_description_requires_review'
    return 'review','unclassified_security_retained_for_review'

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--refresh-directory',action='store_true')
    ap.add_argument('--sec-map',type=Path,default=OLD_CACHE/'coverage_v4_probe/sec_tickers_exchange.web.json')
    args=ap.parse_args();CACHE.mkdir(parents=True,exist_ok=True)
    raw_path=CACHE/'nasdaqtraded.txt'
    if args.refresh_directory or not raw_path.exists():
        if args.refresh_directory:
            with urlopen('https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt',timeout=40) as r:b=r.read()
        else:b=(OLD_CACHE/'nasdaqtraded.txt').read_bytes()
        raw_path.write_bytes(b)
    raw=raw_path.read_text();sec=json.loads(args.sec_map.read_text())
    if sec['fields']!=['cik','name','ticker','exchange']:raise ValueError('Official map fields changed')
    mapping=defaultdict(set);details=defaultdict(list)
    for cik,name,t,exchange in sec['data']:
        ticker=t.replace('-','.');mapping[ticker].add(int(cik));details[ticker].append({'cik':int(cik),'SEC_name':name,'exchange':exchange})
    old=json.loads((OLD_CACHE/'universe.json').read_text());accepted={};review=[];excluded=[];changes=[]
    directory=list(csv.DictReader(io.StringIO(raw),delimiter='|'))
    seen=set()
    for row in directory:
        t=row.get('Symbol','')
        if not t or t.startswith('File Creation'):continue
        if t in seen:raise ValueError('Duplicate directory ticker: '+t)
        seen.add(t);ciks=mapping.get(t,set());unique=len(ciks)==1
        category,reason=classify(row,unique)
        item={'ticker':t,'name':row.get('Security Name',''),'exchange':row.get('Listing Exchange'),
            'financial_status':row.get('Financial Status'),'category':category,'classification_reason':reason,
            'membership_mode':'current_retrospective_not_PIT','SEC_ticker_records':details.get(t,[]),
            'security_description_confirmed':category=='candidate'}
        if unique:item.update(cik=next(iter(ciks)),cik_mapping_source='SEC official ticker/exchange unique exact ticker')
        elif ciks:item['cik_mapping_status']='ambiguous_official_ticker_not_guessed'
        else:item['cik_mapping_status']='not_in_official_ticker_map'
        if category=='candidate':accepted[t]={k:v for k,v in item.items() if k!='ticker'}
        elif category=='review':
            review.append(item)
            # Keep uncertain securities in the broad price/financial watch pool.
            # Their type cannot qualify an issued signal until separately confirmed.
            accepted[t]={k:v for k,v in item.items() if k!='ticker'}
        else:excluded.append(item)
        previous=old['stocks'].get(t)
        old_ex=next((x['reason'] for x in old['exclusions'] if x['ticker']==t),None) if not previous else None
        if (category=='candidate')!=(previous is not None) or previous and previous.get('cik')!=item.get('cik'):
            changes.append({'ticker':t,'old_candidate':previous is not None,'old_exclusion':old_ex,'new_category':category,
                'new_reason':reason,'old_cik':previous.get('cik') if previous else None,'new_cik':item.get('cik')})
    metadata={'version':'coverage_repair_v4','registered_at':datetime.now(timezone.utc).isoformat(),
        'directory_count':len(seen),'accepted_count':len(accepted),'explicit_equity_count':len(accepted)-len(review),'review_count':len(review),'excluded_count':len(excluded),
        'raw_sha256':sha(raw_path),'nasdaq_footer':raw.splitlines()[-1],
        'sec_mapping_rows':len(sec['data']),'sec_mapping_status':'official_unique_exact_ticker_web_source_extraction',
        'sec_map_path':str(args.sec_map.resolve()),'sec_map_sha256':sha(args.sec_map),
        'source_extraction_evidence_path':str((args.sec_map.parent/'sec_web_extraction_evidence.json').resolve()),
        'source_extraction_evidence_sha256':sha(args.sec_map.parent/'sec_web_extraction_evidence.json'),
        'protocol_sha256':sha(HERE/'PROTOCOL.md'),'catalog_code_sha256':sha(Path(__file__)),
        'membership_mode':'current_retrospective_not_PIT','new_independent_effect_evidence':False}
    result={'metadata':metadata,'stocks':accepted,'review':review,'exclusions':excluded,'changes':changes}
    stable=json.loads(json.dumps(result));stable['metadata'].pop('registered_at')
    key=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest();target=CACHE/'catalogs'/key[:16]
    if (target/'universe.json').exists():
        result=json.loads((target/'universe.json').read_text());metadata=result['metadata']
    target.mkdir(parents=True,exist_ok=True);(target/'universe.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (target/'nasdaqtraded.txt').write_bytes(raw_path.read_bytes());(target/'protocol.md').write_bytes((HERE/'PROTOCOL.md').read_bytes())
    # This pointer belongs only to v4; old universe and model pointers are untouched.
    (CACHE/'universe.json').write_text(json.dumps(result,ensure_ascii=False))
    (CACHE/'catalog_latest.json').write_text(json.dumps({'path':str(target.resolve()),'key':key},indent=2))
    (HERE/'CATALOG_EVIDENCE.json').write_text(json.dumps({'metadata':metadata,'key':key,
        'path':str(target.resolve()),'old_candidates':len(old['stocks']),'added_candidates':sum(x['new_category']=='candidate' and not x['old_candidate'] for x in changes),
        'removed_from_old_candidates':sum(x['old_candidate'] and x['new_category']=='excluded' for x in changes),
        'exact_CIK_coverage':sum('cik' in x for x in accepted.values()),'changes':changes},ensure_ascii=False,indent=2))
    print(json.dumps({**metadata,'key':key,'changes':len(changes)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
