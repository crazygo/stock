"""Free original-accession financial facts. No current consensus in historical inputs."""
import json,time,urllib.request
from collections import Counter
import pandas as pd
from common import OUT,write,sha,now
TAGS={'revenue':['RevenueFromContractWithCustomerExcludingAssessedTax','Revenues','SalesRevenueNet','RevenueFromContractWithCustomerIncludingAssessedTax'],'eps':['EarningsPerShareDiluted','EarningsPerShareBasic'],'income':['NetIncomeLoss','ProfitLoss']}

def main():
 audit=json.loads((OUT/'sec_audit.json').read_text());events=json.loads((OUT/'cache/sec/events.json').read_text())['events'];by_symbol={};records=[];statuses=[];consecutive=0
 for r in events:by_symbol.setdefault(r['symbol'],[]).append(r)
 for symbol in sorted(by_symbol):
  filings=by_symbol[symbol];cik=filings[0]['cik'];url=f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json';path=OUT/'cache/sec_facts'/f'{symbol}.json'
  try:
   if path.exists():payload=json.loads(path.read_text())
   else:
    request=urllib.request.Request(url,headers={'User-Agent':'StockResearch local public data research','Accept':'application/json'})
    with urllib.request.urlopen(request,timeout=25) as response:payload=json.load(response)
    write(path,payload);write(path.with_suffix('.receipt.json'),dict(received_at=now(),source=url,sha256=sha(path)));time.sleep(.5)
   if int(payload['cik'])!=cik:raise ValueError('CIK mismatch')
   flat=[]
   for metric,tags in TAGS.items():
    for priority,tag in enumerate(tags):
     for tax in ['us-gaap','ifrs-full']:
      node=payload.get('facts',{}).get(tax,{}).get(tag,{})
      unit='USD/shares' if metric=='eps' else 'USD'
      for row in node.get('units',{}).get(unit,[]):
       if not all(k in row for k in ['start','end','val','accn','filed']):continue
       duration=(pd.Timestamp(row['end'])-pd.Timestamp(row['start'])).days
       if not 70<=duration<=110:continue
       flat.append(dict(metric=metric,tag=tag,priority=priority,duration=duration,**row))
   candidates=pd.DataFrame(flat);count=0
   if not candidates.empty:
    for filing in filings:
     same=candidates[candidates.accn==filing['accession']]
     if same.empty or not filing['accepted_at'].endswith('Z'):continue
     item=dict(symbol=symbol,cik=cik,accession=filing['accession'],accepted_at=filing['accepted_at'],form=filing['form'],source=url,metrics={})
     for metric in TAGS:
      m=same[same.metric==metric].sort_values(['end','priority'],ascending=[False,True]).drop_duplicates('end')
      if m.empty:continue
      current=m.iloc[0];prior=m[(pd.to_datetime(current.end)-pd.to_datetime(m.end)).dt.days.between(350,380)]
      v=dict(value=float(current.val),start=current.start,end=current.end,tag=current.tag,unit='USD/shares' if metric=='eps' else 'USD',original_accession=filing['accession'])
      if len(prior):
       previous=prior.iloc[0];den=abs(float(previous.val))
       v.update(prior_value=float(previous.val),prior_end=previous.end,yoy_change=(float(current.val)-float(previous.val))/den if den>1e-6 else None)
      item['metrics'][metric]=v
     if item['metrics']:records.append(item);count+=1
   statuses.append(dict(symbol=symbol,status='ok',quarterly_accession_events=count,raw_sha256=sha(path)));consecutive=0
   print(json.dumps(dict(symbol=symbol,quarterly_events=count)),flush=True)
  except Exception as e:
   statuses.append(dict(symbol=symbol,status='unavailable',error=type(e).__name__,http_status=getattr(e,'code',None)));consecutive+=1
   if consecutive>=3:break
 write(OUT/'cache/sec_facts/events.json',dict(at=now(),events=records,sec_source_events_sha256=sha(OUT/'cache/sec/events.json')))
 write(OUT/'sec_facts_audit.json',dict(at=now(),status='partial_free_public',symbols=statuses,events=len(records),fields=TAGS,accepted_timestamp_rule='exact original accession join, UTC acceptance+5minutes in R04',
  scope='standard GAAP quarter facts indexed to original accession; no adjusted EPS, analyst expectation, guidance or earliest earnings release',
  revision='later amended/restated accessions cannot populate an earlier accession; current API extraction is not a historical delivery snapshot',
  primary_document_sample_verification='pending; not claimed complete'))
if __name__=='__main__':main()
