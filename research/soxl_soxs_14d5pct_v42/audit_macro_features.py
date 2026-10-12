"""Independently reconstruct each window by multiplying raw daily gross returns."""
from pathlib import Path
import math,json
import numpy as np
import pandas as pd
from macro_features import ROOT,SYMBOLS,MACRO,load_raw
from verify_continuation import near


def audit_macro_features(run):
    run=Path(run);manifest=json.loads((run/'input_manifest.json').read_text());reg=json.loads((run/'registration.json').read_text());p=reg['preregistration']['protocol']
    raw,source=load_raw(ROOT/p['macro_acquisition']);assert source==manifest['macro_source']
    dates=[pd.Timestamp(x['session_date']) for x in manifest['macro_calendar']['sessions']];cal={pd.Timestamp(x['session_date']):x for x in manifest['macro_calendar']['sessions']}
    inputs=pd.read_parquet(run/'macro_panel.parquet');old=pd.read_parquet(Path(manifest['source'])/'financial_panel.parquet');pd.testing.assert_frame_equal(inputs[old.columns],old)
    added=pd.read_parquet(run/'macro_features.parquet');stored=pd.read_parquet(run/'macro_returns.parquet');receipt=json.loads((run/'macro_receipts.json').read_text())
    prices={};actions={};gross={};checks=0;dividends=0;splits=0
    for s in SYMBOLS:
        prices[s]={pd.Timestamp(r.time_key).normalize():r for r in raw[s]['daily'].itertuples()};assert set(prices[s])==set(dates)
        actions[s]={pd.Timestamp(r.ex_div_date):r for r in raw[s]['actions'].itertuples()};gross[s]={}
        for i,date in enumerate(dates):
            row=prices[s][date];assert np.isfinite([row.open,row.high,row.low,row.close,row.volume]).all() and min(row.open,row.high,row.low,row.close)>0 and row.volume>=0
            assert row.low<=min(row.open,row.close) and row.high>=max(row.open,row.close)
            act=actions[s].get(date);mult=1.;cash=0.
            if act is not None:
                assert all(pd.isna(getattr(act,c)) or getattr(act,c)==0 for c in ['per_share_div_ratio','per_share_trans_ratio','allotment_ratio','stk_spo_ratio','spin_off_ratio'])
                if pd.notna(act.split_ratio):mult=1./act.split_ratio;splits+=int(mult!=1)
                if pd.notna(act.per_cash_div):cash=float(act.per_cash_div);dividends+=int(cash!=0)
                assert mult==1 or cash==0
            if i:
                value=(row.close*mult+cash)/prices[s][dates[i-1]].close;gross[s][date]=value
                near(stored.loc[date,s],math.log(value));checks+=1
            else:assert pd.isna(stored.loc[date,s])
    for day,group in inputs.groupby('date'):
        j=dates.index(day);assert j>=21;ready=added.loc[group.index,'macro_ready'];assert ready.all()
        record=next(x for x in receipt['days'] if pd.Timestamp(x['date'])==day)
        assert record['ready'] and pd.Timestamp(record['previous_session'])==dates[j-1]
        assert [pd.Timestamp(x) for x in record['used_close_dates']]==dates[j-21:j]
        latest=pd.Timestamp(cal[dates[j-1]]['close_at']);assert pd.Timestamp(record['latest_close_at'])==latest and latest<day.tz_localize('America/New_York')
        for h in [5,20]:
            # Multiplication rather than production summation of cached log returns.
            totals={s:math.log(math.prod(gross[s][d] for d in dates[j-h:j])) for s in SYMBOLS}
            for s,val in totals.items():near(record['totals'][str(h)][s],val);checks+=1
            for stem,left,right in [('spy','SPY',None),('small_large','IWM','SPY'),('long_short','TLT','SHY'),('credit','HYG','LQD')]:
                col='macro_'+stem+'_'+str(h);value=totals[left]-(totals[right] if right else 0.)
                near(record['values'][col],value)
                for at in group.index:near(inputs.loc[at,col],value);near(added.loc[at,col],value);checks+=2
    info={'passed':True,'raw_daily_returns_and_window_checks':checks,'sessions':len(dates),'symbols':len(SYMBOLS),'dividend_events':dividends,'split_events':splits,
          'feature_queries':len(inputs),'all_original_ready_dates_retained':True,'no_current_session_macro_prices':True,'historical_received_at_not_proven':True}
    (run/'macro_feature_audit.json').write_text(json.dumps(info,indent=2)+'\n');return info
