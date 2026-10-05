"""Causal daily/peer extensions bound to the exact immutable v5 label rows."""
import json, sys
import numpy as np
import pandas as pd
from common import OUT, OLD, DATES, SESSIONS, sha, save, write
sys.path.insert(0,str(OLD))
from feature_primitives import GROUPS, GROUP_OF
LABELS=['entry','exit','exit_minute','y','net','mfe','mae','label_end','complete_future']

def read_raw(symbol):
    # Use the exact registry cited by the old manifest, never a path fallback.
    path=OLD/'raw'/f'{symbol}.parquet'
    if not path.exists():raise FileNotFoundError(f'Bound raw registry missing: {symbol}')
    return pd.read_parquet(path)

def daily_features(raw):
    rows=[]
    for day,g in raw.groupby('day',sort=True):
        if day not in SESSIONS:continue
        close=570+SESSIONS[day]['duration_minutes'];g=g[(g.minute>=570)&(g.minute<close)]
        if not np.array_equal(g.minute.to_numpy(),np.arange(570,close,5)):continue
        prices=g[['open','high','low','close']].to_numpy(float)
        if not (np.isfinite(prices).all() and (prices>0).all() and (g.high>=g[['open','close']].max(axis=1)).all() and (g.low<=g[['open','close']].min(axis=1)).all()):continue
        rows.append(dict(day=day,open=g.open.iloc[0],high=g.high.max(),low=g.low.min(),close=g.close.iloc[-1],volume=g.volume.sum()))
    d=pd.DataFrame(rows).set_index('day').reindex(DATES);c=d.close;ret=np.log(c).diff();tr=pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1)
    f=pd.DataFrame(index=d.index)
    for n in [5,10,20,60]:
        f[f'd_return_{n}']=c.pct_change(n,fill_method=None)
        f[f'd_ma_gap_{n}']=c/c.rolling(n,min_periods=n).mean()-1
    f['d_vol_10']=ret.rolling(10,min_periods=10).std()
    f['d_vol_20']=ret.rolling(20,min_periods=20).std()
    f['d_atr_14']=tr.rolling(14,min_periods=14).mean()/c
    f['d_up_fraction_10']=(ret>0).astype(float).where(ret.notna()).rolling(10,min_periods=10).mean()
    f['d_volume_ratio_5_20']=d.volume.rolling(5,min_periods=5).mean()/d.volume.rolling(20,min_periods=20).mean()
    f['d_range_position_20']=(c-d.low.rolling(20,min_periods=20).min())/(d.high.rolling(20,min_periods=20).max()-d.low.rolling(20,min_periods=20).min())
    f['d_drawdown_20']=c/d.high.rolling(20,min_periods=20).max()-1
    f=f.shift(1);f['d_missing']=f.d_return_60.isna().astype(float)
    return f.reset_index(names='day')

def extend_peers(panel):
    # Current fixed groups are retrospective; missing peers remain explicit.
    group=panel.symbol.map(GROUP_OF).fillna('unmapped');panel['_group']=group
    grouped=panel.groupby(['day','minute','_group'],observed=True,sort=False)
    panel['b_peer_count']=grouped.symbol.transform('size').astype(float)
    panel['b_peer_expected']=group.map({k:len(v) for k,v in GROUPS.items()}).astype(float)
    panel['b_peer_coverage']=panel.b_peer_count/panel.b_peer_expected
    for source in ['i15_ret','i30_ret','rth_ret','pre_gap']:
        valid=panel[source].notna();sign=(panel[source]>0).astype(float).where(valid)
        temp=panel.assign(_sign=sign).groupby(['day','minute','_group'],observed=True,sort=False)
        panel['b_breadth_'+source]=temp._sign.transform('mean')
        med=grouped[source].transform('median');panel['b_relative_'+source]=panel[source]-med
        panel['b_dispersion_'+source]=grouped[source].transform('std')
    aggregates=panel.groupby(['day','minute','_group'],observed=True).agg(b_leader_return=('i30_ret','max'),b_mean_return=('i30_ret','mean'),b_positive=('b_breadth_i30_ret','first')).reset_index()
    aggregates=aggregates.sort_values(['day','_group','minute'])
    g=aggregates.groupby(['day','_group'],observed=True,sort=False)
    aggregates['b_positive_change_30']=aggregates.b_positive-g.b_positive.shift(6)
    aggregates['b_leader_change_30']=aggregates.b_leader_return-g.b_leader_return.shift(6)
    # Actual repeated leader identity, with only completed causal prefixes.
    leader=panel.sort_values(['day','minute','_group','i30_ret','symbol'],ascending=[True,True,True,False,True],na_position='last').drop_duplicates(['day','minute','_group'])[['day','minute','_group','symbol']]
    aggregates=aggregates.merge(leader.rename(columns={'symbol':'_leader'}),on=['day','minute','_group'],validate='one_to_one')
    previous=aggregates.groupby(['day','_group'],observed=True)._leader.shift(6)
    aggregates['b_leader_persist_30']=(aggregates._leader==previous).astype(float).where(previous.notna())
    added=[c for c in aggregates if c.startswith('b_') and c not in panel]
    panel=panel.merge(aggregates[['day','minute','_group']+added],on=['day','minute','_group'],how='left',validate='many_to_one',sort=False)
    panel['b_group_missing']=(panel._group=='unmapped').astype(float)
    cols=[c for c in panel if c.startswith('b_') and c!='b_group_missing']
    panel.loc[panel._group=='unmapped',cols]=np.nan
    return panel.drop(columns='_group')

def extend_events(panel):
    path=OUT/'cache/sec/events.json'
    columns=['e_filing_age_hours','e_earnings_filing','e_quarterly_filing','e_post_return','e_post_volume_ratio','e_missing']
    for c in columns:panel[c]=np.nan
    panel['e_missing']=1.
    if not path.exists():return panel,dict(status='unavailable',eligible_rows=0)
    events=json.loads(path.read_text())['events'];ev=pd.DataFrame(events)
    if ev.empty:return panel,dict(status='unavailable',eligible_rows=0)
    ev=ev[ev.accepted_at.astype(str).str.len()>10].copy()
    ev['available']=pd.to_datetime(ev.accepted_at,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)+pd.Timedelta(seconds=60)
    # Unknown/malformed timestamps never become midnight events.
    ev['earnings']=(ev.form.eq('8-K')&ev['items'].astype(str).str.contains('2.02',regex=False)).astype(float)
    ev['quarterly']=ev.form.isin(['10-Q','10-K','20-F']).astype(float)
    ev=ev[(ev.earnings>0)|(ev.quarterly>0)]
    for symbol,ix in panel.groupby('symbol',sort=False).groups.items():
        es=ev[ev.symbol==symbol].sort_values('available')
        if es.empty:continue
        rows=panel.loc[ix];cut=pd.to_datetime(rows.day)+pd.to_timedelta(rows.minute,unit='m')+pd.Timedelta(seconds=30)
        positions=np.searchsorted(es.available.to_numpy(),cut.to_numpy(),side='right')-1
        valid=positions>=0
        if not valid.any():continue
        selected=es.iloc[positions[valid]];ids=np.asarray(ix)[valid]
        hours=(cut.to_numpy()[valid]-selected.available.to_numpy())/np.timedelta64(1,'h')
        active=hours<=168;ids=ids[active];selected=selected.iloc[np.flatnonzero(active)]
        panel.loc[ids,'e_filing_age_hours']=hours[active]
        panel.loc[ids,'e_earnings_filing']=selected.earnings.to_numpy();panel.loc[ids,'e_quarterly_filing']=selected.quarterly.to_numpy()
        panel.loc[ids,'e_post_return']=panel.loc[ids,'rth_ret'];panel.loc[ids,'e_post_volume_ratio']=panel.loc[ids,'rth_rvol'];panel.loc[ids,'e_missing']=0.
    return panel,dict(status='filing_confirmation_only',eligible_rows=int((panel.e_missing==0).sum()),
          fields=columns,availability='original filing acceptance + 60s assumption; not original earnings announcement, no consensus')

def main():
    coverage=json.loads((OLD/'horizon_v1/coverage.json').read_text())
    source=OLD/'weekly_v1/panel.parquet';label=OLD/'horizon_v1/labels_5d5pct.parquet'
    assert sha(source)==coverage['panel_sha256'],'legacy feature hash drift'
    assert sha(label)==coverage['targets']['5d5pct']['label_sha256'],'legacy label hash drift'
    panel=pd.read_parquet(source);labels=pd.read_parquet(label)
    assert len(panel)==len(labels) and np.array_equal(labels.row_id,np.arange(len(panel)))
    panel=panel.drop(columns=LABELS,errors='ignore');panel['symbol']=panel.symbol.astype(str);panel['day']=panel.day.astype(str)
    panel['_row_id']=np.arange(len(panel))
    daily=[]
    for i,symbol in enumerate(sorted(panel.symbol.unique()),1):
        f=daily_features(read_raw(symbol));f['symbol']=symbol;daily.append(f)
        if i%10==0:print(json.dumps(dict(phase='daily',symbols=i)),flush=True)
    panel=panel.merge(pd.concat(daily,ignore_index=True),on=['symbol','day'],how='left',validate='many_to_one',sort=False)
    panel=extend_peers(panel);panel,event_info=extend_events(panel)
    panel=panel.sort_values('_row_id').reset_index(drop=True)
    assert np.array_equal(panel._row_id.to_numpy(),labels.row_id.to_numpy())
    for c in ['y','entry','target','label_end','future_status']:panel[c]=labels[c].to_numpy()
    for c in panel.select_dtypes('number'):
        if c not in ['_row_id','minute','entry_minute','entry','target']:panel[c]=panel[c].astype('float32')
    panel.symbol=panel.symbol.astype('category');panel.day=panel.day.astype('category')
    save(panel,OUT/'cache/panel.parquet')
    write(OUT/'coverage.json',dict(requested=['2024-10-04','2026-09-30'],actual_features=[min(panel.day.astype(str)),max(panel.day.astype(str))],
        rows=len(panel),symbols=panel.symbol.nunique(),panel_sha256=sha(OUT/'cache/panel.parquet'),
        legacy_panel_sha256=coverage['panel_sha256'],legacy_labels_sha256=coverage['targets']['5d5pct']['label_sha256'],
        protocol_sha256=sha(OUT/'PROTOCOL.md'),membership='current_snapshot_retrospective_not_PIT',events=event_info,
        daily_fields=[c for c in panel if c.startswith('d_')],peer_fields=[c for c in panel if c.startswith('b_')],
        labels=dict(panel.future_status.value_counts()),price_basis='NONE'))
    print(json.dumps(dict(phase='prepared',rows=len(panel),events=event_info)),flush=True)
if __name__=='__main__':main()
