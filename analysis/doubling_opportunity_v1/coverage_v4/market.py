"""Causal split-aware ratios from native NONE prices; no QFQ dividend conversion."""
from __future__ import annotations
import numpy as np,pandas as pd

PRICE_FEATURES=['ret5','ret20','ret60','vol20','downvol20','range20','volume_ratio',
 'peak_ratio','trough_ratio','position183','ma20_ratio','ma60_ratio',
 'dollar_volume20','price','coverage183','rs20','rs60']
UNSUPPORTED=['per_share_div_ratio','per_share_trans_ratio','allotment_ratio','stk_spo_ratio','spin_off_ratio','special_dividend']

def split_state(index,events,asof):
    """Future split scaling cancels in ratios; actual quoted price stays untouched."""
    index=pd.DatetimeIndex(index);factor=np.ones(len(index));unknown=np.zeros(len(index),bool)
    feature_unknown=np.zeros(len(index),bool);issues=[]
    if events.empty:return factor,unknown,feature_unknown,issues
    if 'ex_div_date' not in events:raise ValueError('Corporate-action response lacks ex-date')
    for row in events.to_dict('records'):
        if pd.notna(v:=pd.to_numeric(row.get('source_all_history_uncertain'),errors='coerce')) and abs(v)>1e-12:
            unknown[:]=True;feature_unknown[:]=True;issues.append({'reason':'undated_primary_source_action_or_identity_uncertain'});continue
        day=pd.to_datetime(row['ex_div_date'],errors='coerce')
        if pd.isna(day):
            unknown[:]=True;feature_unknown[:]=True;issues.append({'reason':'invalid_ex_date'});continue
        if day>pd.Timestamp(asof):continue
        ratio=pd.to_numeric(row.get('split_ratio'),errors='coerce')
        bad=any(pd.notna(v:=pd.to_numeric(row.get(k),errors='coerce')) and abs(v)>1e-12 for k in UNSUPPORTED+['source_action_uncertain'])
        if pd.notna(ratio):
            if not np.isfinite(ratio) or ratio<=0:bad=True
            else:factor[index<day]*=float(ratio)
        if bad:
            pos=index.searchsorted(day)
            if pos<len(index):unknown[pos]=True
            feature_unknown|=(index>=day)&(index<day+pd.Timedelta(days=183))
            issues.append({'ex_date':day.date().isoformat(),'reason':'unsupported_or_invalid_corporate_action'})
    if not np.isfinite(factor).all() or (factor<=0).any():raise ValueError('Unrepresentable split scaling')
    return factor,unknown,feature_unknown,issues

def normalized(raw,index,events,asof):
    frame=raw.copy();frame['date']=pd.to_datetime(frame.time_key.astype(str).str[:10])
    if frame.date.duplicated().any():raise ValueError('Duplicate native daily date')
    q=frame.set_index('date').reindex(index)
    values=q[['open','high','low','close','volume']].astype(float)
    invalid=(~np.isfinite(values[['open','high','low','close']]).all(axis=1)|
             (values[['open','high','low','close']]<=0).any(axis=1)|
             (values.high+1e-8<values[['open','low','close']].max(axis=1))|
             (values.low-1e-8>values[['open','high','close']].min(axis=1))|
             ~np.isfinite(values.volume)|(values.volume<0))
    original=values.copy();values.loc[invalid,:]=np.nan
    factor,unknown,feature_unknown,issues=split_state(index,events,asof)
    adjusted=values.copy()
    for k in ['open','high','low','close']:adjusted[k]=values[k]*factor
    adjusted.volume=values.volume/factor
    return values,adjusted,factor,unknown,feature_unknown,issues,original

def price_features(raw,index,events,asof,spy_returns,expected_counts):
    rawq,q,factor,unknown,feature_unknown,issues,original=normalized(raw,index,events,asof)
    c=q.close;r=c.pct_change(fill_method=None);f=pd.DataFrame(index=index)
    for n in (5,20,60):f[f'ret{n}']=c/c.shift(n)-1
    f['vol20']=r.rolling(20,min_periods=20).std();f['downvol20']=r.clip(upper=0).rolling(20,min_periods=20).std()
    f['range20']=q.high.rolling(20).max()/q.low.rolling(20).min()-1
    f['volume_ratio']=q.volume.rolling(5).mean()/q.volume.rolling(20).mean()
    peak=q.high.rolling('183D',min_periods=60).max();trough=q.low.rolling('183D',min_periods=60).min()
    f['peak_ratio']=c/peak;f['trough_ratio']=c/trough;f['position183']=(c-trough)/(peak-trough)
    f['ma20_ratio']=c/c.rolling(20).mean();f['ma60_ratio']=c/c.rolling(60).mean()
    # Raw historical dollar turnover and absolute price cannot use future split factors.
    f['dollar_volume_actual']=(rawq.close*rawq.volume).rolling(20).mean()
    f['dollar_volume20']=np.log1p(f.dollar_volume_actual);f['price']=np.log(rawq.close)
    counts=c.rolling('183D').count();f['coverage183']=counts/expected_counts.reindex(index)
    for n in (20,60):f[f'rs{n}']=f[f'ret{n}']-spy_returns[n].reindex(index)
    f['close']=rawq.close;f['split_basis_multiplier']=factor
    f['peak183']=peak/factor;f['trough183']=trough/factor
    f['unsupported_action_feature_window']=feature_unknown
    warmup=index>=pd.Timestamp('2023-01-01')+pd.Timedelta(days=183)
    f['eligible']=(rawq.close>=1)&(f.dollar_volume_actual>=500_000)&(counts>=60)&warmup&~feature_unknown&f[PRICE_FEATURES].notna().all(axis=1)
    f['invalid_native_bar']=original.close.notna()&rawq.close.isna()
    return f,q,unknown,issues

def labels(adjusted,index,unknown,asof,horizon,cost=.002):
    """Mature only after the last official session in the calendar-day window."""
    index=pd.DatetimeIndex(index);expiry=index+pd.Timedelta(days=horizon)
    stop=index.searchsorted(expiry,side='right');high=adjusted.high.to_numpy();close=adjusted.close.to_numpy()
    missing=np.r_[0,np.cumsum(~np.isfinite(high)|~np.isfinite(close))]
    unsupported=np.r_[0,np.cumsum(unknown)];y=np.full(len(index),np.nan);end=np.full(len(index),np.nan)
    maturity=[];touch=[];reason=[];cutoff=pd.Timestamp(asof)
    for i,j in enumerate(stop):
        mature=index[j-1] if j>i+1 else pd.NaT;maturity.append(mature);first=None
        if not np.isfinite(close[i]):status='missing_or_invalid_reference'
        elif j<=i+1:status='no_future_session'
        elif expiry[i]>index[-1] or mature>cutoff:status='immature'
        elif missing[j]>missing[i+1]:status='missing_or_invalid_future_session'
        elif unsupported[j]>unsupported[i+1]:status='unsupported_corporate_action'
        else:
            status='mature_provider_daily_split_economic_touch';target=close[i]*(1+cost)*2
            hits=np.flatnonzero(high[i+1:j]>=target);y[i]=int(bool(len(hits)));end[i]=int(close[j-1]>=target)
            if len(hits):first=index[i+1+hits[0]]
        touch.append(first);reason.append(status)
    return pd.DataFrame({f'y{horizon}':y,f'mature_at{horizon}':maturity,f'first_touch{horizon}':touch,
                         f'end_close_double{horizon}':end,f'label_status{horizon}':reason},index=index)
