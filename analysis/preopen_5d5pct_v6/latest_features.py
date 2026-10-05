"""Latest available completed-session research features, never pretend weekend is live."""
import json,subprocess,sys
import pandas as pd
from common import OUT,OLD,SESSIONS,save,write,now
from prepare import daily_features,extend_peers

def main():
    latest=[]
    for p in (OLD/'raw').glob('*.parquet'):
        f=pd.read_parquet(p,columns=['day']);latest += [d for d in f.day.unique() if d in SESSIONS]
    day=max(latest);path=OUT/'cache/latest_base.parquet';path.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(OUT/'legacy_features_adapter.py'),'--day',day,'--output',str(path)],check=True)
    frame=pd.read_parquet(path);parts=[]
    for symbol in sorted(frame.symbol.unique()):
        raw=pd.read_parquet(OLD/'raw'/f'{symbol}.parquet');d=daily_features(raw);d['symbol']=symbol;parts.append(d)
    daily_history=pd.concat(parts,ignore_index=True);save(daily_history,OUT/'cache/latest_daily.parquet')
    frame=frame.merge(daily_history[daily_history.day==day],on=['symbol','day'],how='left',validate='many_to_one')
    frame=extend_peers(frame)
    for c in ['e_filing_age_hours','e_earnings_filing','e_quarterly_filing','e_post_return','e_post_volume_ratio']:frame[c]=float('nan')
    frame['e_missing']=1.
    cut=570+SESSIONS[day]['duration_minutes']-30;frame=frame[frame.minute==cut].copy();save(frame,OUT/'cache/latest_features.parquet')
    write(OUT/'latest_features_metadata.json',dict(at=now(),day=day,minute=cut,rows=len(frame),source='latest available completed session; causal prefix',current=False,
        availability='historical bar_end+1s assumption; no real-time receipt claim',price_basis='NONE'))
if __name__=='__main__':main()
