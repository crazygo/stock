"""Capture the issuer's full QQQ holdings, preserving cash and derivatives."""
from __future__ import annotations
import argparse,html,json,re
from urllib.request import Request,urlopen
from urllib.parse import urlparse
from common import *
from acquire import coverage_report

def fetch(url):
 if not urlparse(url).hostname.endswith('invesco.com'):raise ValueError('Unexpected issuer host')
 return urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=20).read()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--from-capture',type=Path);parser.add_argument('--source-url');args=parser.parse_args()
 page='https://www.invesco.com/qqq-etf/en/about.html'
 if args.from_capture:
  url=args.source_url
  if not url or not urlparse(url).hostname.endswith('invesco.com'):raise ValueError('Captured response requires its issuer URL')
  raw=args.from_capture.read_bytes()
 else:
  source=fetch(page).decode();api=re.findall(r'data-holding-api="([^"]+)"',source)
  if not api:raise RuntimeError('Issuer page no longer exposes holdings endpoint')
  enabled='true' in re.findall(r'data-enable-file-data-source="([^"]+)"',source)
  url='https://www.invesco.com/content/dam/invesco/qqq-etf/en/bulk-uploads/holdings/sector-allocation.json' if enabled else html.unescape(api[0]);raw=fetch(url)
 j=json.loads(raw);equities=[];excluded=[]
 for r in j['holdings']:
  code=r.get('ticker');kind=r.get('securityTypeCode')
  if kind in ['COM','ADR','DRNY'] and code and re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}',code):equities.append(r)
  else:excluded.append(r)
 if len(equities)<90 or len({r['ticker'] for r in equities})!=len(equities):raise RuntimeError('Incomplete or duplicated QQQ equity capture')
 u=json.loads((OUT/'universe.json').read_text());frozen=OUT/'frozen_universe_before_qqq_source.json'
 if not frozen.exists():frozen.write_text(json.dumps(u,ensure_ascii=False,indent=2))
 stamp=datetime.now(ET).isoformat();path=OUT/'universe_sources'/('QQQ_'+datetime.now(ET).strftime('%Y%m%d_%H%M%S')+'.json');path.parent.mkdir(exist_ok=True);path.write_bytes(raw)
 for row in u['members'].values():row['groups']=[g for g in row['groups'] if g!='ETF成分:QQQ']
 for r in equities:
  s=r['ticker'];m=u['members'].setdefault(s,dict(symbol=s,groups=[]));m.setdefault('name',html.unescape(r['issuerName']));m.setdefault('stock_type','STOCK');m['groups'].append('ETF成分:QQQ')
 u['funds']['QQQ']=dict(status='full_issuer_api_us_equities',source_url=url,issuer_page=page,source_effective_date=j.get('effectiveDate'),source_business_date=j.get('effectiveBusinessDate'),observed_at=stamp,raw_sha256=sha(path),raw_file=str(path.relative_to(OUT)),members=[r['ticker'] for r in equities],equity_count=len(equities),excluded_non_equity=excluded,total_holdings=len(j['holdings']),previous_capture_status='unresolved_full_constituents')
 write(OUT/'universe.json',u);coverage_report(u);print(json.dumps(dict(source=url,observed_at=stamp,equities=len(equities),excluded=len(excluded),known_pool=len(u['members'])),ensure_ascii=False))

if __name__=='__main__':main()
