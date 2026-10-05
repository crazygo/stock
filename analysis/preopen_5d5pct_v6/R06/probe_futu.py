"""Finite read-only Futu capability probes; each call isolated and timeout bounded."""
import sys,json,subprocess,datetime,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def worker(spec):
 import futu as ft,pandas as pd
 ft.SysConfig.enable_proto_encrypt(False)
 q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
 out=dict(name=spec['name'],method=spec['method'],kwargs=spec['kwargs'],requested_at=stamp())
 try:
  values=getattr(q,spec['method'])(**spec['kwargs']);out['received_at']=stamp();out['ret']=int(values[0]);out['parts']=[]
  for i,value in enumerate(values[1:]):
   dest=ROOT/'cache/probes'/f"{spec['name']}_{i}"
   if isinstance(value,pd.DataFrame):
    path=dest.with_suffix('.parquet');value.to_parquet(path,index=False,compression='zstd',compression_level=7)
    out['parts'].append(dict(kind='frame',rows=len(value),columns=list(value),attrs=value.attrs,file=str(path.relative_to(ROOT)),sha256=sha(path),preview=value.head(1).to_dict('records')))
   elif isinstance(value,(dict,list)):
    path=dest.with_suffix('.json');path.write_text(json.dumps(value,ensure_ascii=False,default=str))
    out['parts'].append(dict(kind=type(value).__name__,file=str(path.relative_to(ROOT)),sha256=sha(path),keys=list(value) if isinstance(value,dict) else None,preview=str(value)[:500]))
   else:out['parts'].append(dict(kind=type(value).__name__,value=str(value)[:800]))
 except Exception as e:out.update(ret=-999,error=repr(e),received_at=stamp())
 finally:q.close()
 path=ROOT/'cache/probes'/f"{spec['name']}_metadata.json";path.write_text(json.dumps(out,ensure_ascii=False,indent=2,default=str)+'\n')

SPECS=[
 ('option_statistics','get_option_underlying_his_statistic',dict(code='US.AMD',begin_time='2024-10-04',end_time='2026-09-30')),
 ('option_volatility_history','get_option_underlying_his_volatility',dict(code='US.AMD',begin_time='2024-10-04',end_time='2026-09-30')),
 ('short_volume','get_daily_short_volume',dict(code='US.AMD',num=50)),
 ('short_interest','get_short_interest',dict(code='US.AMD',num=50)),
 ('earnings_price_history','get_financials_earnings_price_history',dict(code='US.AMD')),
 ('earnings_price_move','get_financials_earnings_price_move',dict(code='US.AMD',period_count=8)),
 ('financial_statements','get_financials_statements',dict(code='US.AMD',num=20)),
 ('analyst_consensus','get_research_analyst_consensus',dict(code='US.AMD')),
 ('rating_detail','get_research_rating_summary',dict(code='US.AMD',num=20)),
 ('news_search','get_search_news',dict(keyword='AMD',max_count=10)),
 ('short_volume_alab','get_daily_short_volume',dict(code='US.ALAB',num=50)),
 ('option_statistics_alab','get_option_underlying_his_statistic',dict(code='US.ALAB',begin_time='2024-10-04',end_time='2026-09-30')),
 ('option_volatility_history_alab','get_option_underlying_his_volatility',dict(code='US.ALAB',begin_time='2024-10-04',end_time='2026-09-30')),
 ('capital_distribution','get_capital_distribution',dict(stock_code='US.AMD')),
 ('capital_flow_intraday','get_capital_flow',dict(stock_code='US.AMD',period_type='INTRADAY',start='2026-05-01',end='2026-05-01')),
 ('market_snapshot','get_market_snapshot',dict(code_list=['US.AMD','US.ALAB'])),
 ('order_book','get_order_book',dict(code='US.AMD',num=5)),
 ('recent_ticker','get_rt_ticker',dict(code='US.AMD',num=10)),
 ('owner_plates','get_owner_plate',dict(code_list=['US.AMD','US.ALAB'])),
 ('dividends','get_corporate_actions_dividends',dict(code='US.AMD')),
 ('splits','get_corporate_actions_stock_splits',dict(code='US.AMD',num=20)),
 ('buybacks','get_corporate_actions_buybacks',dict(code='US.AMD',num=20)),
 ('insider_trades','get_insider_trade_list',dict(code='US.AMD',num=20)),
 ('institutional_holdings','get_shareholders_institutional',dict(code='US.AMD',num=20)),
 ('macro_list','get_macro_indicator_list',dict(region='US')),
 ('economic_calendar','get_economic_calendar',dict(begin_date='2026-05-01',end_date='2026-05-07',market_list=['US'],count=100)),
 ('earnings_calendar','get_earnings_calendar',dict(market='US',begin_date='2026-05-01',end_date='2026-05-07')),
]

def main():
 if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(json.loads(sys.argv[2]));return
 (ROOT/'cache/probes').mkdir(parents=True,exist_ok=True);records=[]
 for name,method,kwargs in SPECS:
  spec=dict(name=name,method=method,kwargs=kwargs);meta=ROOT/'cache/probes'/f'{name}_metadata.json'
  if not meta.exists():
   try:
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',json.dumps(spec)],capture_output=True,text=True,timeout=25)
    (ROOT/'logs'/f'probe_{name}.log').write_text(result.stdout+result.stderr)
   except subprocess.TimeoutExpired:
    meta.write_text(json.dumps(dict(**spec,ret=-998,status='timeout_rejected_this_round',received_at=stamp()))+'\n')
   time.sleep(1.1)
  if not meta.exists():meta.write_text(json.dumps(dict(**spec,ret=-997,status='worker_failed_rejected_this_round',received_at=stamp()))+'\n')
  data=json.loads(meta.read_text());records.append(data)
  print(json.dumps(dict(name=name,ret=data['ret'],parts=[{k:p.get(k) for k in ['kind','rows','columns','keys','value']} for p in data.get('parts',[])]),ensure_ascii=False),flush=True)
  (ROOT/'probe_results.json').write_text(json.dumps(dict(at=stamp(),status='running',records=records),ensure_ascii=False,indent=2)+'\n')
 (ROOT/'probe_results.json').write_text(json.dumps(dict(at=stamp(),status='completed',records=records),ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
