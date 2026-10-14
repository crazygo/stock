"""Freeze a results/outlook document inventory from preserved official directories."""
from pathlib import Path
import hashlib,json,re
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).resolve().parent
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v89'
SYMBOLS=['NVDA','AMD','AVGO','MU','AMAT','LRCX','KLAC','INTC']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def inventory():
    target=SOURCE/'catalogue.json';assert not target.exists()
    path=ROOT/'research/soxl_soxs_14d5pct_v86/evidence/indexes.json'
    source=json.loads(path.read_text());rows=[];excluded=[];hashes={str(path.relative_to(ROOT)):sha(path)}
    for symbol in SYMBOLS:
        for row in source[symbol]['all_records']:
            if not '2022-07-01'<=row['published_date']<='2026-04-30':continue
            if not re.search(r'results|outlook|guidance|revenue|business update|preliminary',row['title'],re.I):continue
            if re.search(r'to report|to announce|sets|conference call|earnings date',row['title'],re.I):continue
            item=dict(row,symbol=symbol,id=hashlib.sha256(row['url'].encode()).hexdigest()[:24])
            if re.search(r'patent valid|private cloud outlook|cash tender offer|medical federated|direct funding under CHIPS',row['title'],re.I):
                excluded.append(dict(item,reason='Directory title concerns a different subject, not issuer revenue results or guidance'));continue
            evidence=ROOT/row['index_evidence'];hashes[row['index_evidence']]=sha(evidence)
            item['candidate_kind']='update_or_preliminary' if re.search(r'updates|revises|reaffirms|preliminary',row['title'],re.I) else 'quarterly_results'
            rows.append(item)
    assert len({r['url'] for r in rows})==len(rows)
    out=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='Fixed eight issuers; official directory results and outlook documents 2022-07-01 through 2026-04-30; pointers only, body review required.',records=rows,excluded=excluded,source_sha256=hashes,
        not_complete_guidance_revision_history=True,no_market_outcomes_read=True)
    dump(target,out);print({s:sum(r['symbol']==s for r in rows) for s in SYMBOLS});print('documents',len(rows),'excluded',len(excluded))
if __name__=='__main__':inventory()
