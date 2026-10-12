"""Previous-session total-return proxy inputs, with explicit raw/action receipts."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.model_history_calendar import calendar
SYMBOLS=['SPY','IWM','TLT','SHY','HYG','LQD'];HORIZONS=[5,20]
EQUITY=['macro_spy_'+str(h) for h in HORIZONS]+['macro_small_large_'+str(h) for h in HORIZONS]
BONDS=['macro_long_short_'+str(h) for h in HORIZONS]+['macro_credit_'+str(h) for h in HORIZONS]
MACRO=EQUITY+BONDS
UNSUPPORTED=['per_share_div_ratio','per_share_trans_ratio','allotment_ratio','stk_spo_ratio','spin_off_ratio']


def load_raw(manifest_path):
    manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text());assert not manifest['partial']
    assert [x['symbol'] for x in manifest['symbols']]==SYMBOLS
    raw={};hashes={}
    for item in manifest['symbols']:
        s=item['symbol'];raw[s]={}
        for kind in ['daily','actions']:
            p=ROOT/item[kind]['path'];h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==item[kind]['sha256']
            raw[s][kind]=pd.read_parquet(p);hashes[item[kind]['path']]=h
    return raw,{'acquisition_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'files_sha256':hashes,'manifest':manifest}


def daily_returns(raw,sessions):
    """Ex-date cash is a price-return adjustment, not available trading cash."""
    dates=pd.DatetimeIndex([pd.Timestamp(s['session_date']) for s in sessions]);out=pd.DataFrame(index=dates);audit=[];coverage=[]
    for symbol in SYMBOLS:
        df=raw[symbol]['daily'].copy();df['date']=pd.to_datetime(df.time_key).dt.normalize()
        assert not df.date.duplicated().any();df=df.set_index('date').sort_index();a=raw[symbol]['actions'].copy()
        missing=list(dates.difference(df.index));extra=list(df.index.difference(dates));df=df.reindex(dates)
        prices=df[['open','high','low','close']].to_numpy(float)
        valid=np.isfinite(prices).all(axis=1)&(prices>0).all(axis=1)&df.high.ge(df[['open','close','low']].max(axis=1))&df.low.le(df[['open','close','high']].min(axis=1))&df.volume.ge(0)
        coverage.append({'symbol':symbol,'missing_dates':missing,'extra_dates_excluded':extra,'invalid_price_dates':list(dates[~valid])})
        a['date']=pd.to_datetime(a.ex_div_date).dt.normalize();assert not a.date.duplicated().any()
        a=a.set_index('date');logret=np.full(len(dates),np.nan)
        for j,date in enumerate(dates):
            act=a.loc[date] if date in a.index else None;cash=0.;shares=1.
            if act is not None:
                for col in UNSUPPORTED:
                    value=act.get(col,np.nan)
                    if pd.notna(value) and abs(float(value))>1e-12:valid.loc[date]=False
                ratio=act.get('split_ratio',np.nan)
                if pd.notna(ratio):assert float(ratio)>0;shares=1./float(ratio)
                dividend=act.get('per_cash_div',np.nan)
                if pd.notna(dividend):cash=float(dividend);assert cash>=0
                # Cash amount unit is ambiguous if combined with a split: fail closed.
                if shares!=1 and cash!=0:valid.loc[date]=False
            if j and valid.iloc[j] and valid.iloc[j-1]:
                gross=(float(df.loc[date,'close'])*shares+cash)/float(df.iloc[j-1]['close']);assert gross>0
                logret[j]=np.log(gross)
            if act is not None:audit.append({'symbol':symbol,'date':date,'shares_per_previous_share':shares,'cash_per_previous_share':cash})
        out[symbol]=logret
    return out,{'action_rows':audit,'dates':len(dates),'coverage':coverage,'missing_first_return_only':out.iloc[0].isna().all() and out.iloc[1:].notna().all().all(),
        'availability':'prior official session close assumed available by next ET midnight; actions used only at/past ex-date; not historical first-public or observed received_at'}


def build_features(features,raw,sessions):
    returns,quality=daily_returns(raw,sessions);dates=returns.index;calendar_map={pd.Timestamp(s['session_date']):s for s in sessions};rows=[];receipts=[]
    for day in sorted(set(features.date)):
        j=int(dates.searchsorted(day,side='left'));past=dates[max(0,j-21):j]
        ready=j>=21 and day in dates and returns.iloc[j-20:j].notna().all().all();values={k:np.nan for k in MACRO};totals={}
        if ready:
            for h in HORIZONS:
                total=returns.iloc[j-h:j].sum();totals[str(h)]={s:float(total[s]) for s in SYMBOLS}
                for stem,left,right in [('spy','SPY',None),('small_large','IWM','SPY'),('long_short','TLT','SHY'),('credit','HYG','LQD')]:
                    values['macro_'+stem+'_'+str(h)]=float(total[left]-(total[right] if right else 0.))
        receipts.append({'date':day,'ready':bool(ready),'previous_session':dates[j-1] if j else None,
            'used_close_dates':list(past) if ready else [],'latest_close_at':calendar_map[dates[j-1]]['close_at'] if j else None,'totals':totals,'values':values})
        rows.append({'date':day,'macro_ready':bool(ready),**values})
    bydate=pd.DataFrame(rows).set_index('date');result=features[['date']].join(bydate,on='date').drop(columns='date')
    return result,returns,{'quality':quality,'days':receipts}
