"""Conservative parser for actual primary corporate-action articles, never price-inferred ratios."""
from __future__ import annotations
import json,re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
import numpy as np,pandas as pd

MONTHS='January|February|March|April|May|June|July|August|September|October|November|December'
DATE=rf'(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}'
VOID={'br','img','input','link','meta','hr','area','base','embed','param','source','track','wbr'}

class Article(HTMLParser):
    def __init__(self):super().__init__();self.stack=[];self.depth=None;self.words=[]
    def handle_starttag(self,tag,attrs):
        if tag in VOID:return
        self.stack.append(tag)
        if dict(attrs).get('id')=='rightWideCOL':self.depth=len(self.stack)
    def handle_endtag(self,tag):
        if tag in self.stack:
            i=len(self.stack)-1-self.stack[::-1].index(tag);self.stack=self.stack[:i]
            if self.depth and len(self.stack)<self.depth:self.depth=None
    def handle_data(self,data):
        if self.depth and 'script' not in self.stack and 'style' not in self.stack and data.strip():self.words.append(data.strip())

def iso_date(text):
    cleaned=re.sub(r'(\d)(st|nd|rd|th)\b',r'\1',text,flags=re.I).replace(',','')
    return datetime.strptime(cleaned,'%B %d %Y').date().isoformat()

def parse_html(html,url):
    p=Article();p.feed(html);words=p.words
    if len(words)<3:return {'url':url,'status':'article_structure_unresolved','tickers':[]}
    stop=next((i for i,x in enumerate(words) if x.startswith('Email Alert Subscriptions')),len(words));text=' '.join(words[:stop])
    title=words[2];symbols=[]
    for group in re.findall(r'\(([^()]*)\)',title):
        # SPAC notation ABC/W/U means derivatives of ABC, not the unrelated companies W and U.
        shorthand=re.fullmatch(r'\s*([A-Z][A-Z0-9.]{1,9})(?:\s*/\s*(?:W|WS|U|R))+\s*',group)
        if shorthand:
            symbols.append(shorthand.group(1));continue
        symbols+=re.findall(r'(?<![A-Za-z])[A-Z][A-Z0-9.]{0,9}(?![A-Za-z])',group)
    symbols=list(dict.fromkeys(x for x in symbols if x not in ['UPDATED','CUSIP','NYSE','NASDAQ','ADR','ADS','OTC']))
    publication=re.search(DATE,words[0],re.I)
    record={'url':url,'title':title,'published_at':iso_date(publication.group()) if publication else None,'tickers':symbols,'status':'non_split_action_retained','source_text':text}
    record['explicit_table_symbols']=list(dict.fromkeys(re.findall(r'(?:Current|New)\s+Symbol\s*:\s*([A-Z][A-Z0-9.]{0,9})(?![A-Za-z0-9])',text)))
    effective=[];date_issues=[]
    for m in re.finditer(rf'(?:become\s+effective|take\s+effect|effective(?:\s+date)?|ex[- ]date)[^.;]{{0,120}}?({DATE})',text,re.I):
        d=iso_date(m.group(1));effective.append(d)
        weekdays=re.findall(r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',m.group(0),re.I)
        actual=datetime.fromisoformat(d).strftime('%A')
        if weekdays and weekdays[-1].lower()!=actual.lower():
            date_issues.append({'date':d,'claimed_weekday':weekdays[-1],'actual_weekday':actual,'reason':'explicit_weekday_date_contradiction'})
        if publication and d<record['published_at'] and re.search(r'\bwill\s+(?:effect|implement|become\s+effective|take\s+effect)\b',text,re.I):
            date_issues.append({'date':d,'published_at':record['published_at'],'reason':'future_tense_action_precedes_publication'})
    days=sorted(set(effective));record['candidate_effective_dates']=days
    if not re.search(r'(?:reverse\s+)?(?:stock\s+)?split',title,re.I):return record
    if date_issues:return record|{'status':'source_effective_date_inconsistent','date_integrity_issues':date_issues,'effective_date_resolved':False}
    if re.search(r'merger|business.combination|spin[- ]off|stock dividend|reorgani|conversion|exchange offer|distribution',title,re.I):
        return record|{'status':'combined_corporate_action_retained_unknown'}
    if re.search(r'cancel(?:led|ed|ation)|postpon',text,re.I):return record|{'status':'cancelled_or_postponed_unresolved'}
    ratios=re.findall(r'\(\s*(\d+(?:\.\d+)?)\s*(?:-|:|for|[-\s]*for[-\s]*)\s*(\d+(?:\.\d+)?)\s*\)',text,re.I)
    if not ratios:ratios=re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:[-\s]+for[-\s]+|:)\s*(\d+(?:\.\d+)?)\b',text,re.I)
    pairs={(float(a),float(b)) for a,b in ratios if float(a)>0 and float(b)>0}
    # Only explicit effective clauses; never use the article publication as the split day.
    record.update(candidate_ratios=sorted(pairs))
    if re.search(r'ratio change|depositary|depository|\bADS\b|\bADR\b',text,re.I):
        return record|{'status':'depositary_or_ratio_change_retained_unknown'}
    if len(pairs)!=1 or len(days)!=1 or len(symbols)!=1:return record|{'status':'split_terms_or_identity_unresolved'}
    a,b=next(iter(pairs));return record|{'status':'explicit_single_security_split','ex_date':days[0],'split_ratio':b/a}

def combine_events(native,records,asof):
    """Append verified missing events; contradictions become unknown, retaining both sources."""
    out=native.copy();add=[];audit=[]
    grouped={}
    for r in records:
        if r.get('status')=='verified_split' and r['ex_date']<=asof:grouped.setdefault(r['ex_date'],[]).append(r)
    for day,group in grouped.items():
        r=group[0];ratio=r['split_ratio'];same=native[native.ex_div_date==day] if 'ex_div_date' in native else pd.DataFrame()
        known=same.split_ratio.dropna().astype(float) if 'split_ratio' in same else pd.Series([],dtype=float)
        if not np.isclose([x['split_ratio'] for x in group],ratio,rtol=1e-8,atol=1e-10).all():
            state='conflicting_primary_splits_retained_unknown';add.append({'ex_div_date':day,'source_action_uncertain':1.})
        elif len(known) and np.isclose(known,ratio,rtol=1e-8,atol=1e-10).all():state='native_confirmed_by_primary'
        elif len(known):
            state='conflicting_split_retained_unknown';add.append({'ex_div_date':day,'source_action_uncertain':1.})
        else:
            state='missing_native_split_supplemented_from_primary';add.append({'ex_div_date':day,'split_ratio':ratio,'primary_source_url':r['url']})
        audit.extend({**x,'reconciliation':state} for x in group)
    if add:out=pd.concat([out,pd.DataFrame(add)],ignore_index=True)
    return out,audit

def audited_events(native,record,asof):
    """Exactly the same frozen action and unresolved-gap state in build and verification."""
    events,audit=combine_events(native,record.get('verified_splits',[]),asof)
    unknown=[{'ex_div_date':r['ex_date'],'source_action_uncertain':1.,'source_issue_reason':r['status']}
             for r in record.get('unknown_events',[]) if r['ex_date']<=asof]
    if unknown:events=pd.concat([events,pd.DataFrame(unknown)],ignore_index=True)
    if record.get('whole_history_unknown'):
        events=pd.concat([events,pd.DataFrame([{'ex_div_date':'2023-01-01','source_all_history_uncertain':1.}])],ignore_index=True)
    return events,audit

def unresolved_overnight_gaps(native,index,events,asof,limit=1.8):
    # Import lazily: market uses native/action fields, not this source registry module.
    from market import normalized
    _,q,_,_,_,_,_=normalized(native,index,events,asof)
    ratio=q.open/q.close.shift(1)
    return [{'ex_date':d.date().isoformat(),'adjusted_open_to_previous_close':float(ratio.loc[d]),'status':'unresolved_overnight_price_basis_or_news'}
            for d in ratio.index[(ratio>=limit)|(ratio<=1/limit)] if d<=pd.Timestamp(asof)]
