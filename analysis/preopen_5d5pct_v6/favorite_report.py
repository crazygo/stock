"""Every current Special Focus stock, including zero-signal/unregistered cases."""
import json
from common import OUT,write,now
from research import metrics,FIVE

def main():
 snapshot=json.loads((OUT/'watchlist_snapshot.json').read_text());group=next(s for s in snapshot['snapshots'] if s['group']=='特别关注')
 members=[r for r in group['members'] if r['code'].startswith('US.') and r['stock_type']=='STOCK']
 for rid,base in [('R00',OUT),('R01',OUT/'R01'),('R02',OUT/'R02'),('R03',OUT/'R03'),('R04',OUT/'R04'),('R05',OUT/'R05')]:
  file=base/'results.json'
  if not file.exists():continue
  result=json.loads(file.read_text())
  for arm in result['arms']:
   metas=[json.loads(p.read_text()) for p in (base/'cache/runs').glob(f"{arm['arm']}_*/meta.json")]
   for variant in arm['variants']:
    source=base/f"{arm['arm']}_{variant['variant']}_signals.json";events=json.loads(source.read_text())['events'];ticks=[]
    for m in metas:
     p=base/'cache/runs'/f"{arm['arm']}_{m['month']}"/f"{variant['variant']}_replay.json"
     if p.exists():ticks+=json.loads(p.read_text())['ticks']
    rows=[]
    for member in members:
     symbol=member['code'][3:];registered=[m['month'] for m in metas if symbol in m.get('registered',[])]
     rows.append(dict(symbol=symbol,code=member['code'],name=member.get('name'),required_five=symbol in FIVE,registered_months=sorted(registered),status='registered' if registered else 'no_registered_model',**metrics([e for e in events if e['symbol']==symbol],ticks)))
    codes={r['symbol'] for r in rows};variant['favorites']=dict(membership='current_snapshot_retrospective_not_PIT',stocks=rows,group_metrics=metrics([e for e in events if e['symbol']in codes],ticks),required_five=FIVE)
  write(file,result)
 print(json.dumps(dict(status='completed',current_us_focus_stocks=len(members))),flush=True)
if __name__=='__main__':main()
