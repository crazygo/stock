"""As-of T-1 features and next-session outcomes, with strict session grids."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAL = ROOT / 'market_data/calendars/nasdaq_sessions_2026_v1.json'
SESSIONS = [s for s in json.loads(CAL.read_text())['sessions'] if '2026-01-01' <= s['session_date'] <= '2026-09-25']
DATES = [s['session_date'] for s in SESSIONS]


def clean(frame, start, end):
    expected = pd.date_range(start, end, freq='5min', inclusive='left')
    if len(frame) != len(expected) or not pd.DatetimeIndex(frame.start).equals(expected):
        return None
    x = frame[['open', 'high', 'low', 'close', 'volume', 'turnover']].to_numpy(float)
    good = (np.isfinite(x).all() and (x[:, :4] > 0).all() and (x[:, 4:] >= 0).all()
            and (x[:, 1] >= x[:, [0, 2, 3]].max(axis=1)).all()
            and (x[:, 2] <= x[:, [0, 1, 3]].min(axis=1)).all()
            and frame.price_basis.eq('NONE').all()
            and frame.end.eq(frame.start + pd.Timedelta(minutes=5)).all()
            and frame.available.le(end + pd.Timedelta(seconds=1)).all()
            and frame.available.ge(frame.end).all())
    return x if good else None


def summarize(symbol, raw_override=None):
    path = HERE / f'data_cache/us_5m/{symbol}/2026.parquet'
    raw = pd.read_parquet(path) if raw_override is None else raw_override.copy()
    for col, source in [('start','start_at'), ('end','end_at'), ('available','available_at')]:
        raw[col] = pd.to_datetime(raw[source], utc=True)
    raw.sort_values('start', inplace=True)
    bydate = dict(tuple(raw.groupby('session_date')))
    action_path = ROOT / f'market_data/corporate_actions/{symbol}.parquet'
    actions = pd.read_parquet(action_path)
    action_dates = set(actions.ex_div_date.astype(str))
    split_dates = set(actions.loc[actions.split_ratio.fillna(1).ne(1), 'ex_div_date'].astype(str)) if 'split_ratio' in actions else set()
    rows = []
    for session in SESSIONS:
        day = session['session_date']
        f = bydate.get(day, raw.iloc[:0])
        op, cl = pd.Timestamp(session['open_at']), pd.Timestamp(session['close_at'])
        rth = f[f.start.ge(op) & f.start.lt(cl)]
        a = clean(rth, op, cl)
        r = {'date': day, 'symbol': symbol, 'rth_valid': a is not None,
             'rth_bars':len(rth), 'action_today': day in action_dates,
             'split_today':day in split_dates,
             'feature_cutoff': pd.Timestamp(day+' 20:00:01',tz='America/New_York').isoformat()}
        if a is not None:
            o,h,l,c,v,t = a.T
            width = h.max()-l.min()
            r.update(open=o[0],high=h.max(),low=l.min(),close=c[-1],volume=v.sum(),
                     oc=c[-1]/o[0]-1,range=width/o[0],location=(c[-1]-l.min())/width if width else .5,
                     pullback=c[-1]/h.max()-1, first60=c[11]/o[0]-1,
                     last60=c[-1]/o[-12]-1,last30=c[-1]/o[-6]-1,
                     afternoon=c[-1]/o[-24]-1,up_fraction=float((c>o).mean()),
                     low_time=rth.iloc[int(l.argmin())].start_at_et[11:16],
                     high_time=rth.iloc[int(h.argmax())].start_at_et[11:16],
                     max_available_at=rth.available.max().isoformat())
            # RTH-only audited true VWAP. Invalid sessions are left missing.
            perbar = np.divide(t,v,out=np.full(len(t),np.nan),where=v>0)
            valid = (v==0) | ((perbar>=l-.01)&(perbar<=h+.01))
            r['vwap_quality_fraction']=float(valid.mean())
            if valid.all() and v.sum()>0:
                r['close_vs_vwap']=c[-1]/(t.sum()/v.sum())-1
            for name,begin,end,n in [('pre','04:00','09:30',66),('post','16:00','20:00',48)]:
                st=pd.Timestamp(day+' '+begin,tz='America/New_York').tz_convert('UTC')
                en=pd.Timestamp(day+' '+end,tz='America/New_York').tz_convert('UTC')
                ext=clean(f[f.start.ge(st)&f.start.lt(en)],st,en) if session['duration_minutes']==390 else None
                r[name+'_valid']=ext is not None
                if ext is not None:
                    r[name+'_return']=ext[-1,3]/(c[-1] if name=='post' else ext[0,0])-1
                    r[name+'_volume']=ext[:,4].sum()
        rows.append(r)
    d=pd.DataFrame(rows).set_index('date').reindex(DATES)
    d['cc']=d.close/d.close.shift()-1
    d.loc[d.action_today, 'cc']=np.nan
    d['relvol']=d.volume/d.volume.shift().rolling(20,min_periods=10).median()
    d['range_ratio']=d['range']/d['range'].shift().rolling(20,min_periods=10).median()
    d['return5']=d.close/d.close.shift(5)-1
    action5=d.action_today.astype(int).rolling(5,min_periods=1).sum()>0
    d.loc[action5,'return5']=np.nan
    # Share changes invalidate historical volume comparisons; no adjustment using future factors.
    split21=d.split_today.astype(int).rolling(21,min_periods=1).sum()>0
    d.loc[split21,'relvol']=np.nan
    d['post_vol_ratio']=d.post_volume/d.post_volume.shift().rolling(20,min_periods=10).median()
    d.loc[split21,'post_vol_ratio']=np.nan
    return d.reset_index(), {'symbol':symbol,'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                           'rows':len(raw),'last_bar':raw.end.max().isoformat(),
                           'actions_sha256':hashlib.sha256(action_path.read_bytes()).hexdigest()}


def main():
    universe=json.loads((HERE/'universe.json').read_text())['members']
    daily,manifest=[],[]
    for symbol in [*sorted(universe),'QQQ']:
        d,m=summarize(symbol);daily.append(d);manifest.append(m)
    daily=pd.concat(daily,ignore_index=True)
    q=daily[daily.symbol.eq('QQQ')].set_index('date').cc
    daily['relative_qqq']=daily.cc-daily.date.map(q)
    # Predeclared finer disjoint value-chain labels, no price-based grouping.
    sector={s:'platform_software' for s in universe}
    for name,syms in {
        'compute_chips':'NVDA AMD ARM INTC QCOM',
        'interconnect':'ALAB CRDO LITE MRVL AVGO CSCO',
        'memory_storage':'MU SNDK STX WDC',
        'semi_equipment':'AMAT ASML KLAC LRCX TER',
        'analog_power':'TXN ADI NXPI MPWR MCHP',
        'ai_cloud':'CRWV NBIS',
        'eda':'CDNS SNPS',
        'power_industrial':'CEG HON',
    }.items():
        sector.update(dict.fromkeys(syms.split(),name))
    rows=[]
    cols=['cc','oc','range','location','pullback','first60','last60','last30','afternoon','up_fraction',
          'relvol','range_ratio','return5','post_return','post_vol_ratio','relative_qqq','close_vs_vwap',
          'low_time','high_time','post_valid','pre_return','feature_cutoff','max_available_at']
    for symbol in sorted(universe):
        d=daily[daily.symbol.eq(symbol)].set_index('date').reindex(DATES)
        p=d.shift(1)
        out=pd.DataFrame({'date':DATES,'prev_date':[None]+DATES[:-1],'symbol':symbol,'sector':sector[symbol]})
        for col in cols:out['prev_'+col]=p[col].to_numpy()
        valid=(d.rth_valid & p.rth_valid.fillna(False).astype(bool) & ~d.action_today).to_numpy()
        out['valid_label']=valid
        out['close_return']=(d.close/p.close-1).to_numpy()
        out['high_return']=(d.high/p.close-1).to_numpy()
        out['gap']=(d.open/p.close-1).to_numpy()
        out['open_close']=(d.close/d.open-1).to_numpy()
        out['open_high']=(d.high/d.open-1).to_numpy()
        out['open_low']=(d.low/d.open-1).to_numpy()
        out['close8']=np.where(valid,out.close_return.ge(.08),np.nan)
        out['high10']=np.where(valid,out.high_return.ge(.10),np.nan)
        out['openhigh5']=np.where(valid,out.open_high.ge(.05),np.nan)
        rows.append(out)
    panel=pd.concat(rows,ignore_index=True)
    daily.to_parquet(HERE/'daily.parquet',index=False,compression='zstd')
    panel.to_parquet(HERE/'panel.parquet',index=False,compression='zstd')
    coverage=daily.groupby('symbol').agg(first=('date','min'),last=('date','max'),valid_days=('rth_valid','sum'),observed_days=('rth_bars',lambda x:int((x>0).sum())))
    coverage.to_csv(HERE/'coverage.csv')
    (HERE/'data_manifest.json').write_text(json.dumps({'inputs':manifest,'calendar_sha256':hashlib.sha256(CAL.read_bytes()).hexdigest(),
        'bar_availability':'historical end_at + 1 second assumption','price_basis':'NONE','sectors':sector},indent=2))
    recent=panel[panel.date.ge('2026-09-21') & panel.valid_label]
    recent.to_csv(HERE/'discovery_all.csv',index=False)
    winners=recent[recent.close8.eq(1)|recent.high10.eq(1)].sort_values(['date','close_return'],ascending=[True,False])
    winners.to_csv(HERE/'discovery_events.csv',index=False)
    print('Recent baseline',recent.groupby('date').agg(n=('symbol','size'),close8=('close8','sum'),high10=('high10','sum')).to_string())
    display=['date','symbol','close_return','high_return','gap','open_close','prev_cc','prev_oc','prev_location','prev_relvol','prev_last60','prev_post_return','prev_relative_qqq']
    print(winners[display].round(4).to_string(index=False))


if __name__=='__main__':main()
