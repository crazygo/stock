"""Completed daily response after as-known planned dates; no result-text inference."""
from pathlib import Path
from functools import lru_cache
from datetime import datetime,timedelta,time
from zoneinfo import ZoneInfo
import hashlib,json,sys
import numpy as np
import pandas as pd

SOURCE=Path(__file__).resolve().parent
ROOT=SOURCE.parents[1]
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v88'
PANEL=['NVDA','AMD','AVGO','MU','AMAT','LRCX','KLAC','INTC']
ET=ZoneInfo('America/New_York')
UNSUPPORTED=['special_dividend','per_share_div_ratio','per_share_trans_ratio','allotment_ratio','stk_spo_ratio','spin_off_ratio']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def normalize(raw,actions,sessions):
    """Forward share units; later actions never revise an earlier normalized row."""
    raw=raw.copy();raw.index=raw.time_key.astype(str).str[:10]
    if raw.index.duplicated().any():raise ValueError('Duplicate daily rows')
    dates=[x['session_date'] for x in sessions]
    frame=raw.reindex(dates)[['open','high','low','close','volume']].astype(float)
    frame['valid']=np.isfinite(frame[['open','high','low','close','volume']]).all(axis=1)
    frame['valid']&=(frame[['open','high','low','close']]>0).all(axis=1)&(frame.volume>0)
    frame['valid']&=(frame.high>=frame[['open','low','close']].max(axis=1))&(frame.low<=frame[['open','high','close']].min(axis=1))
    frame['split_ratio']=1.;frame['cash']=0.;frame['share_factor']=1.
    selected=actions[actions.ex_div_date.between(dates[0],dates[-1])].sort_values('ex_div_date')
    if selected.ex_div_date.duplicated().any():raise ValueError('Ambiguous simultaneous action rows')
    for a in selected.to_dict('records'):
        day=a['ex_div_date']
        if day not in frame.index:raise ValueError('Corporate action on a non-session date')
        ratio=a.get('split_ratio');cash=a.get('per_cash_div')
        ratio=float(ratio) if pd.notna(ratio) else 1.;cash=float(cash) if pd.notna(cash) else 0.
        bad=any(pd.notna(a.get(k)) and a[k]!=0 for k in UNSUPPORTED)
        bad|=not np.isfinite([ratio,cash]).all() or ratio<=0 or cash<0 or (ratio!=1 and cash!=0)
        if bad:
            frame.loc[frame.index>=day,'valid']=False
            continue
        frame.loc[day,'split_ratio']=ratio;frame.loc[day,'cash']=cash
        frame.loc[frame.index>=day,'share_factor']/=ratio
    frame['economic_volume']=frame.volume/frame.share_factor
    frame['return_valid']=frame.valid&frame.valid.shift(1,fill_value=False)
    gross=((frame.close/frame.split_ratio+frame.cash)/frame.close.shift(1)).where(frame.return_valid)
    frame['economic_log_return']=np.log(gross)
    frame['available_at']=[datetime.combine(datetime.fromisoformat(d).date()+timedelta(days=1),time(),ET).isoformat() for d in dates]
    frame.index.name='date'
    return frame


def verify_prices():
    path=SOURCE/'evidence/price_inputs.json';manifest=read(path)
    assert manifest['coverage_passed'] and manifest['price_basis']=='NONE'
    for name,digest in manifest['files_sha256'].items():assert sha(ROOT/name)==digest,name
    return sha(path)


@lru_cache(maxsize=1)
def load_prices():
    verify_prices();manifest=read(SOURCE/'evidence/price_inputs.json')
    tables={s:pd.read_parquet(ROOT/manifest['symbols'][s]['normalized_path']) for s in PANEL+['SPY']}
    return tables,manifest['calendar']['sessions']


def response(previous_days,at,tables,sessions):
    at=pd.Timestamp(at)
    if at.tzinfo is None:raise ValueError('Explicit timezone required')
    if len({tuple(x) for x in previous_days})!=len(previous_days):raise ValueError('Issuer day must be deduplicated')
    now=at.tz_convert(ET).date()
    if any(not 1<=(now-datetime.fromisoformat(d).date()).days<=14 for _,d in previous_days):
        raise ValueError('Only prior one to fourteen calendar days are eligible')
    dates=[x['session_date'] for x in sessions];observed=[];pending=[];missing=[]
    for symbol,planned in previous_days:
        if symbol not in PANEL:raise ValueError('Unregistered panel member')
        future=[d for d in dates if d>planned]
        if not future:
            missing.append([symbol,planned,'calendar_next_session_missing']);continue
        reaction_day=future[0]
        available=pd.Timestamp(reaction_day,tz=ET)+pd.DateOffset(days=1)
        if available>at:
            pending.append([symbol,planned,reaction_day]);continue
        pos=dates.index(reaction_day);history=dates[pos-20:pos]
        if len(history)!=20:
            missing.append([symbol,planned,'twenty_prior_sessions_missing']);continue
        required=PANEL+['SPY'];reason=None
        for name in required:
            t=tables[name]
            if reaction_day not in t.index or not t.loc[reaction_day,'return_valid']:
                reason='mature_return_missing:'+name;break
        own=tables[symbol]
        if reason is None and (not set(history)<=set(own.index) or not own.loc[history,'valid'].all()):
            reason='prior_volume_missing:'+symbol
        if reason:
            missing.append([symbol,planned,reason]);continue
        spy=float(tables['SPY'].loc[reaction_day,'economic_log_return'])
        issuer=float(own.loc[reaction_day,'economic_log_return'])-spy
        peers=np.mean([float(tables[s].loc[reaction_day,'economic_log_return'])-spy for s in PANEL if s!=symbol])
        vol=float(np.log(own.loc[reaction_day,'economic_volume']/own.loc[history,'economic_volume'].mean()))
        observed.append(dict(symbol=symbol,planned_date=planned,reaction_date=reaction_day,
            base_date=dates[pos-1],available_at=available.isoformat(),volume_history=history,
            issuer_excess=issuer,peer_excess=float(peers),log_volume_ratio=vol))
    values=None if missing else [len(observed),*[float(np.mean([r[key] for r in observed])) if observed else 0.
        for key in ['issuer_excess','peer_excess','log_volume_ratio']]]
    return dict(known=not missing,values=values,observed=observed,pending=pending,missing=missing)


def prepare_reactions(panel,table):
    tables,sessions=load_prices();out=panel.copy();records=[]
    for at,row in table.iterrows():
        r=response(json.loads(row.previous_days),at,tables,sessions)
        assert r['known'],('Required mature reaction missing',at,r['missing'])
        records.append(dict(at=at,**dict(zip(['reaction_count','issuer_reaction','peer_reaction','volume_reaction'],r['values'])),
            reaction_pending_count=len(r['pending']),reaction_observed=json.dumps(r['observed']),reaction_pending=json.dumps(r['pending'])))
    values=pd.DataFrame(records).set_index('at')
    for name in values.columns:table[name]=values[name]
    for name in ['reaction_count','issuer_reaction','peer_reaction','volume_reaction','reaction_pending_count']:out[name]=values[name]
    return out
