#!/usr/bin/env python3
"""Read-only cross-provider overlap audit; never splice price bases or relabel a backtest."""
import hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from data import ROOT,CACHE
from scripts.model_history_calendar import calendar

def main():
    daily=pd.read_parquet(CACHE/'panel.parquet');parts=[];lineage=[]
    cal=calendar('2026-01-01','2026-12-31')['sessions'];schedule={r['session_date']:r for r in cal}
    for p in sorted((ROOT/'market_data/us_60m').glob('*/*.parquet')):
        ticker=p.parent.name
        if ticker not in set(daily.ticker.unique()):continue
        q=pd.read_parquet(p);q['date']=q.time_key.astype(str).str[:10];q['clock']=q.time_key.astype(str).str[11:16]
        lineage.append({'ticker':ticker,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        for day,rows in q.groupby('date'):
            session=schedule.get(day)
            if not session:continue
            cl=session['close_at_et'][11:16]
            regular=rows[(rows.clock>'09:30')&(rows.clock<=cl)].sort_values('time_key')
            # OpenD archive timestamps are bar ends; reject incomplete regular prefixes.
            expected=math.ceil(session['duration_minutes']/60)
            if len(regular)!=expected or regular.clock.iloc[-1]!=cl:continue
            parts.append({'ticker':ticker,'date':pd.Timestamp(day),'hourly_rth_high':float(regular.high.max()),
                          'hourly_rth_close':float(regular.iloc[-1].close),'hourly_rth_bars':len(regular)})
    audit=pd.DataFrame(parts).merge(daily[['ticker','date','high','close']],on=['ticker','date'],how='inner',validate='one_to_one')
    audit['close_relative_difference']=audit.hourly_rth_close/audit.close-1
    audit['high_relative_difference']=audit.hourly_rth_high/audit.high-1
    aligned=audit[audit.close_relative_difference.abs()<=.001]
    audit.to_parquet(CACHE/'cross_source_overlap.parquet',compression='zstd',compression_level=7,index=False)
    summary={'run_at':datetime.now(timezone.utc).isoformat(),'hourly_symbols':len(lineage),'overlap_stock_days':len(audit),
        'close_within_0_1pct_stock_days':len(aligned),'close_difference_over_0_1pct':int((audit.close_relative_difference.abs()>.001).sum()),
        'aligned_close_but_massive_high_over_futu_rth_by_0_5pct':int((aligned.high_relative_difference<-.005).sum()),
        'aligned_close_but_futu_rth_high_over_massive_by_0_5pct':int((aligned.high_relative_difference>.005).sum()),
        'close_abs_difference_quantiles':{str(k):float(v) for k,v in audit.close_relative_difference.abs().quantile([.5,.9,.99,1]).items()},
        'scope':'existing locally cached current-member subset; neither complete broad pool nor independent PIT sample',
        'basis':'Futu QFQ vs Massive split adjusted mixed vintages; same close is not proof of matching corporate actions or session scope',
        'result':'audit only; no rescaling, splicing, original-label replacement, or qualification upgrade','raw_comparison_path':str(CACHE/'cross_source_overlap.parquet')}
    (CACHE/'cross_source_overlap_lineage.json').write_text(json.dumps(lineage,indent=2))
    (CACHE/'cross_source_overlap_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
