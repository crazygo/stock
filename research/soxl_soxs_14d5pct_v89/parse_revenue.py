"""Candidate numerical extraction for review; never authorizes training use."""
import re,json
from datetime import datetime
from catalogue import *

WORDS={'first':1,'second':2,'third':3,'fourth':4}
NUM=r'[0-9][0-9,]*(?:\.[0-9]+)?'
MONEY=re.compile(r'\$\s*('+NUM+r')\s*(billion|million)',re.I)
def million(value,unit):return float(value.replace(',',''))*(1000 if unit.lower()=='billion' else 1)
def quarter(text):
    m=re.search(r'(first|second|third|fourth)[ -]+quarter.*?(20\d{2})',text,re.I)
    if m:return f'{m[2]}Q{WORDS[m[1].lower()]}'
    m=re.search(r'fiscal\s+(20\d{2})\s+(first|second|third|fourth)[ -]+quarter',text,re.I)
    if m:return f'{m[1]}Q{WORDS[m[2].lower()]}'
    m=re.search(r'(?:F?Q)([1-4])(?:\s*FY\s*|\s*[- ]\s*)(\d{2}|20\d{2})\b',text,re.I)
    if m:return f'{int(m[2])+2000 if len(m[2])==2 else int(m[2])}Q{m[1]}'
    return None
def lam_quarter(text):
    m=re.search(r'quarter end(?:ed|ing)\s+([A-Za-z]+\s+\d{1,2},?\s+20\d{2})',text,re.I)
    if not m:return None
    d=datetime.strptime(m[1].replace(',',''),'%B %d %Y')
    return f'{d.year+(d.month>6)}Q{((d.month-7)%12)//3+1}'
def range_value(text):
    # First currency expression after the revenue label, before unrelated metrics.
    m=re.search(r'\$\s*('+NUM+r')\s*(?:billion\s*)?[-–]\s*\$?\s*('+NUM+r')\s*(billion|million)',text,re.I)
    if m:return dict(lower=million(m[1],m[3]),upper=million(m[2],m[3]),form='range')
    m=MONEY.search(text)
    if not m:return None
    first=million(m[1],m[2]);tail=text[m.end():];band=re.match(r'\s*(?:,\s*)?(?:plus or minus|±|\+/-)\s*(.*)',tail,re.I|re.S)
    if band:
        n=MONEY.match(band[1])
        percent=re.match(r'('+NUM+r')\s*%',band[1])
        if n:width=million(n[1],n[2])
        elif percent:width=first*float(percent[1])/100
        else:return None
        return dict(lower=first-width,upper=first+width,form='symmetric_band')
    n=MONEY.search(tail)
    if n and re.fullmatch(r'\s*(?:to|and)\s*',tail[:n.start()],re.I):return dict(lower=first,upper=million(n[1],n[2]),form='range')
    return dict(lower=first,upper=first,form='point')

def actual(r):
    if r['candidate_kind']!='quarterly_results':return None
    s=r['symbol'];label={'NVDA':'Revenue','AMD':'Net revenue','AVGO':'Net revenue','MU':'Revenue','AMAT':'(?:Net (?:sales|revenue)|Revenue)','LRCX':'Revenue','KLAC':'Total revenues','INTC':'Net revenue'}[s]
    matches=list(re.finditer(r'(?m)^'+label+r'\s+\$?\s*('+NUM+r')',r['text'],re.I))
    floor=100000 if s in ['LRCX','KLAC'] else 1000
    m=next((m for m in matches if float(m[1].replace(',',''))>=floor),None)
    if not m:return None
    value=float(m[1].replace(',',''))/(1000 if s in ['LRCX','KLAC'] else 1)
    return dict(fiscal_period=lam_quarter(r['title']) if s=='LRCX' else quarter(r['title']),revenue_million=value,
        evidence_start=m.start(),evidence_end=m.end(),coordinate_space='document_text',unit_assumption='thousands' if s in ['LRCX','KLAC'] else 'millions',review_state='pending_unit_period_and_scope_review')

def guidance(r):
    if r['candidate_kind']!='quarterly_results':return None
    s=r['symbol'];text=' '.join(r['text'].split());match=None;period=None;scope='quarter'
    if s=='NVDA':match=re.search(r'NVIDIA.s outlook for (.{5,80}?) is as follows:\s*Revenue is expected to be (.{1,160})',text,re.I)
    elif s=='AMD':match=re.search(r'For the (.{5,65}?), AMD expects revenue to be (.{1,160})',text,re.I)
    elif s=='AMAT':
        match=re.search(r'In the (.{5,65}?), Applied expects net (?:sales|revenue) to be (.{1,160})',text,re.I)
        if not match:
            table=re.search(r'Business Outlook.{1,550}?(Q[1-4] FY20\d{2})\s*\(In millions,[^)]*\)\s*Total (?:net )?revenue\s*\$\s*('+NUM+r')\s*\+/-\s*\$\s*('+NUM+r')',text,re.I)
            if table:
                center=float(table[2].replace(',',''));width=float(table[3].replace(',',''))
                return dict(fiscal_period=quarter(table[1]),scope='quarter',range_million=dict(lower=center-width,upper=center+width,form='symmetric_band'),evidence_start=table.start(),evidence_end=table.end(),coordinate_space='whitespace_normalized_text',unit_evidence='Explicit In millions table header',review_state='pending_unit_period_and_scope_review')
    elif s=='AVGO':
        h=re.search(r'((?:First |Second |Third |Fourth )?Quarter Fiscal Year 20\d{2}|Fiscal Year 20\d{2}) Business Outlook',text,re.I)
        if h:
            section=text[h.end():h.end()+1000];n=re.search(r'revenue guidance of (.{1,160})',section,re.I)
            if n:
                period=quarter(h[1]);scope='quarter' if period else 'annual'
                value=range_value(n[1]);return dict(fiscal_period=period or re.search(r'20\d{2}',h[1])[0],scope=scope,range_million=value,evidence_start=h.start(),evidence_end=h.end()+n.end(),coordinate_space='whitespace_normalized_text',review_state='pending_unit_period_and_scope_review')
    elif s=='MU':match=re.search(r'table (?:below )?presents Micron.s guidance for (.{5,70}?)[:.]\s*(.{1,700}?)Revenue\s+(.{1,160})',text,re.I)
    elif s=='LRCX':match=re.search(r'Outlook (For the quarter end(?:ed|ing) .{5,40}?), Lam is providing the following guidance:\s*.{1,120}?Revenue\s+(.{1,160})',text,re.I)
    elif s=='KLAC':match=re.search(r'((?:First|Second|Third|Fourth) Quarter Fiscal (?:Year )?20\d{2}) Guidance.{1,200}?Total revenues? (?:is expected to be |are expected to be )?(.{1,160})',text,re.I)
    elif s=='INTC':match=re.search(r'Business Outlook.{1,420}?(Q[1-4] 20\d{2}).{1,120}?Revenue\s+(.{1,160})',text,re.I)
    if match:
        period=lam_quarter(match[1]) if s=='LRCX' else quarter(match[1]);v=range_value(match[3] if s=='MU' else match[2])
        return dict(fiscal_period=period,scope='quarter',range_million=v,evidence_start=match.start(),evidence_end=match.end(),coordinate_space='whitespace_normalized_text',review_state='pending_unit_period_and_scope_review')
    return None

def candidates():
    rows=[]
    for p in sorted((CACHE/'documents').glob('*.json')):
        r=json.loads(p.read_text());a=actual(r);g=guidance(r)
        if g and g['range_million']:
            v=g['range_million'];v['midpoint']=(v['lower']+v['upper'])/2
        rows.append(dict(id=r['id'],symbol=r['symbol'],published_date=r['published_date'],url=r['url'],candidate_kind=r['candidate_kind'],
            actual_candidate=a,guidance_candidate=g,source_file=r['source_file'],source_sha256=r['source_sha256'],document_file=str(p.relative_to(ROOT)),document_sha256=sha(p),training_use_allowed=False))
    rows.sort(key=lambda r:(r['symbol'],r['published_date']))
    dump(SOURCE/'evidence/revenue_candidates.json',dict(records=rows,training_use_allowed=False,stage='mechanical_extraction_not_reviewed_facts'))
    for s in SYMBOLS:
        q=[r for r in rows if r['symbol']==s]
        print(s,'records',len(q),'actual',sum(bool(r['actual_candidate'] and r['actual_candidate']['fiscal_period']) for r in q),'quarter_guidance',sum(bool(r['guidance_candidate'] and r['guidance_candidate']['scope']=='quarter' and r['guidance_candidate']['fiscal_period'] and r['guidance_candidate']['range_million']) for r in q))
    print('unparsed',[(r['symbol'],r['published_date'],bool(r['actual_candidate']),bool(r['guidance_candidate'])) for r in rows if r['candidate_kind']=='quarterly_results' and (not r['actual_candidate'] or not r['guidance_candidate'])])
if __name__=='__main__':candidates()
