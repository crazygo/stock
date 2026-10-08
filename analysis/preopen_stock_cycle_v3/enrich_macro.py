"""Previous official-session macro values, never same-day closes."""
import json
import pandas as pd
import numpy as np
import build as b
OUT=b.OUT

def macro_frame(calendar):
 days=[r['session_date'] for r in calendar if '2024-01-01'<=r['session_date']<=b.END];m=pd.DataFrame(index=days)
 for s,key in [('VIX','CLOSE'),('VVIX','VVIX'),('VIX9D','CLOSE')]:
  f=pd.read_csv(OUT/'raw'/(s+'_History.csv'));f['day']=pd.to_datetime(f.DATE,format='%m/%d/%Y').dt.strftime('%Y-%m-%d');m[s]=f.set_index('day')[key].reindex(days)
 out=pd.DataFrame(index=days);out['env_vix']=m.VIX.shift(1);out['env_vvix']=m.VVIX.shift(1);out['env_term']=m.VIX9D.div(m.VIX).shift(1)
 for n in [3,10,30]:out['env_vix_change_'+str(n)]=m.VIX.pct_change(n).shift(1)
 out['env_vix_percentile_60']=m.VIX.rolling(60,min_periods=30).rank(pct=True).shift(1)
 return out

def main():
 f=pd.read_parquet(OUT/'panel.parquet');calendar=json.loads((b.legacy.HISTORY/'calendar.json').read_text())['sessions'];m=macro_frame(calendar)
 if not (OUT/'panel_nomacro.parquet').exists():f.to_parquet(OUT/'panel_nomacro.parquet',index=False,compression='zstd',compression_level=7)
 for c in m:f[c]=f.day.map(m[c])
 f.to_parquet(OUT/'panel.parquet',index=False,compression='zstd',compression_level=7)
 a=json.loads((OUT/'audit.json').read_text());a['panel_sha256']=b.sha(OUT/'panel.parquet');a['macro_sources']=json.loads((OUT/'free_macro_sources.json').read_text());(OUT/'audit.json').write_text(json.dumps(a,ensure_ascii=False,indent=2));print(json.dumps({'macro_columns':m.columns.tolist(),'missing_current_vix':int(f.env_vix.isna().sum()),'last_date':m.index[-1]}))
if __name__=='__main__':main()
