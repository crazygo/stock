"""Daily inputs with explicit official-session lags; no current/future records."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from common import OUT,ROOT,OLD,DATES,SESSIONS,sha,save,write,now

DEST=OUT/'R06'
FAMILIES={'capital':'c','options':'o','volatility':'v','short_volume':'s'}

def number(f,key,nonnegative=False):
    s=pd.to_numeric(f[key],errors='coerce').replace([np.inf,-np.inf],np.nan)
    return s.where(s>=0) if nonnegative else s

def ratio(a,b):return a/b.where(b>0)

def measurements(family,frame,turnover=None,grid=DATES):
    f=frame.set_index('source_day').reindex(grid);x=pd.DataFrame(index=f.index)
    if family=='capital':
        denominator=turnover.reindex(grid)
        for name in ['in_flow','super_in_flow','big_in_flow','mid_in_flow','sml_in_flow','main_in_flow']:
            x[name]=ratio(number(f,name),denominator)
        for n in [5,20]:
            for name in ['in_flow','main_in_flow']:
                z=x[name];x[f'{name}_mean_{n}']=z.rolling(n,min_periods=3 if n==5 else 10).mean()
                x[f'{name}_positive_{n}']=(z>0).astype(float).where(z.notna()).rolling(n,min_periods=3 if n==5 else 10).mean()
    elif family=='options':
        for name in ['call_volume','put_volume','call_open_interest','put_open_interest']:
            x['log_'+name]=np.log1p(number(f,name,True))
        for name in ['put_call_volume_ratio','put_call_open_interest_ratio']:x[name]=number(f,name,True)
        volume=number(f,'option_volume',True);oi=number(f,'option_open_interest',True)
        x['volume_relative_20']=ratio(volume,volume.shift(1).rolling(20,min_periods=10).mean())
        x['oi_change_5']=ratio(oi,oi.shift(5))-1
    elif family=='volatility':
        x['iv']=number(f,'iv',True);x['hv']=number(f,'hv',True);x['iv_hv_spread']=x.iv-x.hv
        x['iv_change_5']=x.iv-x.iv.shift(5);x['hv_change_5']=x.hv-x.hv.shift(5)
        x['iv_percentile_20']=x.iv.rolling(20,min_periods=10).rank(pct=True)
    else:
        pct=number(f,'short_percent',True);x['short_percent']=pct.where(pct<=100)/100
        qty=number(f,'total_shares_short',True)
        x['quantity_relative_20']=ratio(qty,qty.shift(1).rolling(20,min_periods=10).mean())
        for n in [5,20]:x[f'short_percent_mean_{n}']=x.short_percent.rolling(n,min_periods=3 if n==5 else 10).mean()
    # A rolling statistic cannot manufacture a new observation on a missing day.
    observed=pd.Series(f.index.isin(frame.source_day),index=f.index)
    x=x.where(observed,axis=0)
    return x.replace([np.inf,-np.inf],np.nan),observed

def lagged(family,values,observed,lag,grid=DATES):
    prefix=FAMILIES[family];tag=f'f{lag}_{prefix}_';out=pd.DataFrame({'day':grid})
    indices=np.arange(len(grid),dtype=float);source=pd.Series(indices,index=grid).where(observed).ffill().shift(lag)
    delayed=values.ffill().shift(lag)
    # Preserve each observed row's within-family missing fields, rather than
    # borrowing an individual field from an earlier provider record.
    for i in range(len(grid)):
        if pd.notna(source.iloc[i]):delayed.iloc[i]=values.iloc[int(source.iloc[i])]
    stale=indices-source.to_numpy()-lag
    for c in values:out[tag+c]=delayed[c].to_numpy()
    out[tag+'source_day']=[grid[int(i)] if pd.notna(i) else None for i in source]
    out[tag+'available_day']=[grid[int(i)+lag] if pd.notna(i) and int(i)+lag<len(grid) else None for i in source]
    out[tag+'missing']=source.isna().astype(float).to_numpy()
    out[tag+'stale_sessions']=stale
    out[tag+'coverage_20']=observed.astype(float).rolling(20,min_periods=1).mean().shift(lag).to_numpy()
    out[tag+'value_missing_fraction']=delayed.isna().mean(axis=1).to_numpy()
    return out

def real_turnover(symbol,raw):
    paths=sorted((ROOT/'market_data/us_5m'/symbol).glob('*.parquet'))+sorted((OUT/'cache/r2/us_5m'/symbol).glob('*.parquet'))
    # The separate original monthly archive retains turnover that normalized
    # price caches intentionally omitted. Use the already-local capital period.
    paths+=sorted(p for p in (ROOT/'market_data/model_training_history_v1/parts'/symbol).glob('*/bars.parquet') if '2025-10'<=p.parent.name<='2026-09')
    parts=[];sources=[];rejected=0
    for p in paths:
        f=pd.read_parquet(p)
        if not {'turnover','start_at_et','price_basis'}<=set(f) or set(f.price_basis.dropna())!={'NONE'}:continue
        f['start']=pd.to_datetime(f.start_at_et,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        parts.append(f[['start','turnover','open','high','low','close','volume']]);sources.append({'file':str(p.relative_to(ROOT)), 'sha256':sha(p)})
    if not parts:return pd.Series(dtype=float),sources,0
    f=pd.concat(parts).merge(raw[['start','day','minute','open','high','low','close','volume']],on='start',suffixes=('_archive',''),validate='many_to_one')
    cols=['open','high','low','close','volume']
    identical=np.isclose(f[[c+'_archive' for c in cols]].to_numpy(float),f[cols].to_numpy(float),rtol=0,atol=1e-8,equal_nan=False).all(1)
    f['valid']=identical & pd.to_numeric(f.turnover,errors='coerce').ge(0) & np.isfinite(pd.to_numeric(f.turnover,errors='coerce'))
    candidate_days=set(f.loc[(f.minute>=570)&(f.minute<960),'day']);f=f[f.valid].drop_duplicates('start',keep='first');totals={}
    for day,g in f.groupby('day',observed=True):
        if day not in SESSIONS:continue
        g=g[(g.minute>=570)&(g.minute<570+SESSIONS[day]['duration_minutes'])].sort_values('minute')
        if np.array_equal(g.minute.to_numpy(),np.arange(570,570+SESSIONS[day]['duration_minutes'],5)) and g.valid.all():totals[day]=float(g.turnover.sum())
    rejected=len(candidate_days-set(totals))
    return pd.Series(totals,dtype=float),sources,rejected

def prepare():
    acquisition=json.loads((DEST/'acquisition.json').read_text());assert acquisition['status']=='completed'
    from quality import main as quality_audit
    quality_audit()
    parent=json.loads((OUT/'R03/coverage.json').read_text());source=OUT/'R03/cache/panel.parquet';assert sha(source)==parent['panel_sha256']
    daily=[];sources=[];grid=[d for d in DATES if d<='2026-10-02']
    for symbol in acquisition['fixed_symbols']:
        p=OUT/'cache/backfill'/f'{symbol}.parquet'
        if not p.exists():p=OLD/'raw'/f'{symbol}.parquet'
        raw=pd.read_parquet(p);turnover,ts,rejected=real_turnover(symbol,raw)
        allf=pd.DataFrame({'day':grid});fs=[]
        for family in FAMILIES:
            fp=DEST/'cache/history'/family/f'{symbol}.parquet'
            if fp.exists():
                f=pd.read_parquet(fp);fs.append(dict(family=family,rows=len(f),first=f.source_day.min() if len(f) else None,last=f.source_day.max() if len(f) else None,sha256=sha(fp)))
            else:
                names={'capital':['in_flow','super_in_flow','big_in_flow','mid_in_flow','sml_in_flow','main_in_flow'],'options':['call_volume','put_volume','call_open_interest','put_open_interest','put_call_volume_ratio','put_call_open_interest_ratio','option_volume','option_open_interest'],'volatility':['iv','hv'],'short_volume':['short_percent','total_shares_short']}[family]
                f=pd.DataFrame(columns=['source_day']+names);fs.append(dict(family=family,rows=0,status='unavailable'))
            assert not f.source_day.duplicated().any()
            x,obs=measurements(family,f,turnover,grid)
            for lag in [2,5]:allf=allf.merge(lagged(family,x,obs,lag,grid),on='day',validate='one_to_one')
        allf['symbol']=symbol;daily.append(allf);sources.append(dict(symbol=symbol,families=fs,turnover_sources=ts,turnover_days=len(turnover),turnover_rejected_days=rejected))
        if len(daily)%20==0:print(json.dumps(dict(stage='daily_features',symbols=len(daily))),flush=True)
    d=pd.concat(daily,ignore_index=True);save(d,DEST/'cache/daily_features.parquet')
    numeric=[c for c in d.select_dtypes('number')];controls=[c for c in numeric if c.endswith(('_missing','_stale_sessions','_coverage_20','_value_missing_fraction'))]
    families={k:[c for c in numeric if c.startswith(f'f2_{v}_') and c not in controls] for k,v in FAMILIES.items()}
    delayed=[c for c in numeric if c.startswith('f5_') and c not in controls]
    panel=pd.read_parquet(source);panel.symbol=panel.symbol.astype(str);panel.day=panel.day.astype(str)
    # Source/available dates are persisted once per stock-day in daily_features,
    # not replicated across millions of minute rows. Only numerical inputs join.
    keys=panel[['symbol','day','minute']].to_numpy();panel=panel.merge(d[['symbol','day']+numeric],on=['symbol','day'],how='left',sort=False,validate='many_to_one');assert np.array_equal(keys,panel[['symbol','day','minute']].to_numpy())
    for c in numeric:panel[c]=panel[c].astype('float32')
    reference=json.loads((OUT/'R03/cache/runs/H3_2026-09/meta.json').read_text())
    panel.symbol=panel.symbol.astype('category');panel.day=panel.day.astype('category');save(panel,DEST/'cache/panel.parquet')
    coverage=dict(parent,at=now(),rows=len(panel),panel_sha256=sha(DEST/'cache/panel.parquet'),parent_panel_sha256=parent['panel_sha256'],protocol_sha256=sha(DEST/'PROTOCOL.md'),base_features=reference['features'],control_features=controls,family_features=families,delayed_features=delayed,new_sources=sources,daily_features_sha256=sha(DEST/'cache/daily_features.parquet'),input_time='record source_day plus 2 or 5 official sessions; provider historical vintages unverified; exposed development assumptions',turnover='actual archive RTH turnover only, complete bars matching immutable R03 OHLCV; missing denominator retained, never estimated')
    write(DEST/'coverage.json',coverage);print(json.dumps(dict(prepared=len(panel),numeric=len(numeric))),flush=True)

if __name__=='__main__':prepare()
