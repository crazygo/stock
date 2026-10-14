"""Fixed six-coordinate issuer archive proxies; no market outcomes are used."""
from pathlib import Path
import sys,json,hashlib,math
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v90'))
from issuer_state import snapshot,DATA
import pandas as pd
FIELDS=['fund_matched_count','fund_realization_log','fund_above_upper_centered',
        'fund_outlook_count','fund_outlook_log_growth','fund_result_mean_age']
SOURCE=Path(__file__).resolve().parent
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v91'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def coordinates(state):
    if not state['known'] or len(state['issuers'])!=8:raise ValueError('Incomplete issuer query coverage')
    paired=[r for r in state['issuers'] if r['status']=='matched_regular_outlook']
    forward=[r for r in state['issuers'] if r.get('scope_status')=='reported_issuer_revenue_proxy' and r.get('next_guidance_scope')=='quarter']
    n=len(paired);k=len(forward)
    # Empty means have explicit zero count; unresolved company values are not
    # included in the denominator and never become zero-valued observations.
    values=[float(n),sum(math.log1p(r['realization']) for r in paired)/n if n else 0.,
            2*sum(r['actual_revenue_million']>r['reference_upper_million'] for r in paired)/n-1 if n else 0.,
            float(k),sum(math.log(r['next_guidance_midpoint_million']/r['actual_revenue_million']) for r in forward)/k if k else 0.,
            sum(r['age_calendar_days'] for r in paired)/n if n else 0.]
    return dict(zip(FIELDS,values)),dict(paired_ids=[r['latest_result_id'] for r in paired],
        paired_reference_ids=[r['reference_id'] for r in paired],forward_ids=[r['latest_result_id'] for r in forward],
        unknown=[dict(symbol=r['symbol'],status=r['status']) for r in state['issuers'] if r['status']!='matched_regular_outlook'])


def verify_inputs():
    directory=ROOT/'research/soxl_soxs_14d5pct_v90';manifest=read(directory/'evidence/delivery_manifest.json')
    for f,h in manifest['delivery_files_sha256'].items():assert sha(directory/f)==h,f
    for f,h in manifest['input_sha256'].items():assert sha(ROOT/f)==h,f
    return sha(directory/'evidence/delivery_manifest.json')


def prepare(panel,output=None):
    data=read(DATA);rows=[]
    for at in panel.index:
        s=snapshot(data,at.isoformat());values,meta=coordinates(s)
        rows.append(dict(at=at,**values,paired_ids=json.dumps(meta['paired_ids']),
            paired_reference_ids=json.dumps(meta['paired_reference_ids']),forward_ids=json.dumps(meta['forward_ids']),unknown=json.dumps(meta['unknown'])))
    table=pd.DataFrame(rows).set_index('at');out=panel.copy()
    for name in FIELDS:out[name]=table[name]
    out['fundamental_known']=True
    if output is not None:table.to_csv(output,index=True)
    return out
