"""Add strictly past premarket, after-hours and multi-day hourly context."""
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "expanded_v2"
SOURCE = ROOT / "research/group_expectation_matrix/outputs/20260925_v5"

EXTRA = {
    "premarket_gap": ("09:30盘前末价相对前收盘涨幅", "pct", "premarket_price"),
    "premarket_return": ("04:00至09:30盘前涨幅", "pct", "premarket_price"),
    "premarket_range": ("盘前振幅相对前收盘价", "pct", "premarket_range"),
    "premarket_location": ("盘前末价在盘前高低区间的位置", "ratio", "premarket_position"),
    "premarket_pullback": ("盘前末价距盘前最高价的回落", "pct", "premarket_position"),
    "premarket_last60": ("08:30至09:30盘前最后1小时涨幅", "pct", "premarket_hour"),
    "premarket_hour_acceleration": ("盘前末1小时涨幅减此前1小时涨幅", "pct", "premarket_hour"),
    "premarket_relative_volume": ("盘前总量/过去20日盘前量中位数", "multiple", "premarket_volume"),
    "premarket_volume_vs_regular": ("盘前总量/过去20日常规盘量中位数", "ratio", "premarket_volume"),
    "prev_postmarket_return": ("前一交易日16:00至20:00盘后涨幅", "pct", "afterhours"),
    "prev_postmarket_range": ("前一交易日盘后振幅", "pct", "afterhours"),
    "overnight_open_gap": ("04:00开价相对前日20:00末价涨幅", "pct", "overnight_gap"),
    "prior1_session_return": ("前一交易日常规盘开收涨幅", "pct", "prior_day"),
    "prior3_return": ("前3交易日收盘收益", "pct", "history"),
    "prior10_return": ("前10交易日收盘收益", "pct", "history"),
    "prior60_return": ("前60交易日收盘收益", "pct", "history"),
    "prior5_vol": ("过去5日收盘收益日波动率", "pct", "volatility"),
    "prior60_vol": ("过去60日收盘收益日波动率", "pct", "volatility"),
    "prev_last60_return": ("前一交易日收盘前1小时涨幅", "pct", "prior_hour"),
    "prior3_hour_up_fraction": ("前3日常规盘小时上涨占比", "ratio", "prior_hour"),
    "prior3_hour_efficiency": ("前3日小时净价格变化/绝对变化和", "ratio", "prior_hour"),
    "prior3_hour_vol": ("前3日常规盘小时收益波动率", "pct", "prior_hour"),
    "premarket_qqq_relative": ("盘前涨幅减QQQ盘前涨幅", "pct", "premarket_relative"),
    "qqq_premarket_gap": ("QQQ盘前末价相对前收盘涨幅", "pct", "premarket_market"),
}


def clean_session(frame, expected, cutoff):
    if len(frame) != expected or frame.start.duplicated().any():
        return None
    a = frame[["open", "high", "low", "close", "volume", "turnover"]].to_numpy(float)
    if not (np.isfinite(a[:, :4]).all() and (a[:, 2] > 0).all()
            and (a[:, 1] >= np.maximum(a[:, 0], a[:, 3])).all()
            and (a[:, 2] <= np.minimum(a[:, 0], a[:, 3])).all()
            and frame.available.le(cutoff).all() and frame.price_basis.eq("NONE").all()):
        return None
    if not (frame.end.to_numpy() == (frame.start + pd.Timedelta(minutes=5)).to_numpy()).all():
        return None
    # A full grid, rather than merely a matching row count.
    if expected > 1 and not (np.diff(frame.start.astype("int64")) == 300_000_000).all():
        # pandas timezone dtype can be microseconds or nanoseconds.
        diffs = frame.start.diff().dropna()
        if not diffs.eq(pd.Timedelta(minutes=5)).all():
            return None
    return a


def build_symbol(raw, sessions, actions):
    raw = raw.copy()
    raw["start"] = pd.to_datetime(raw.start_at, utc=True)
    raw["end"] = pd.to_datetime(raw.end_at, utc=True)
    raw["available"] = pd.to_datetime(raw.available_at, utc=True)
    raw.sort_values("start", inplace=True)
    byday = {d: f for d, f in raw.groupby("session_date")}
    history_rth, history_pre, history_post, closes = [], [], [], []
    rows = []
    dates = [s["session_date"] for s in sessions]
    for i, session in enumerate(sessions):
        day = session["session_date"]
        f = byday.get(day, raw.iloc[:0])
        before = pd.Timestamp(f"{day} 09:30:01", tz="America/New_York").tz_convert("UTC")
        pre = clean_session(f[f.session_type.eq("pre_market")], 66, before)
        close_time = pd.Timestamp(session["close_at"]) + pd.Timedelta(seconds=1)
        regular = clean_session(f[f.session_type.eq("regular")], session["duration_minutes"]//5, close_time)
        post_end = pd.Timestamp(f"{day} 20:00:01", tz="America/New_York").tz_convert("UTC")
        post = clean_session(f[f.session_type.eq("post_market")], 48, post_end)
        r = {k: np.nan for k in EXTRA}
        prev = history_rth[-1] if history_rth else None
        prev_post = history_post[-1] if history_post else None
        prev_close = closes[-1] if closes else np.nan
        recent_action = any(dates[max(0,i-61)] < a <= day for a in actions)
        if pre is not None:
            o,h,l,c,v,t = pre.T
            width = h.max()-l.min()
            r.update(premarket_return=c[-1]/o[0]-1,
                     premarket_location=(c[-1]-l.min())/width if width>0 else .5,
                     premarket_pullback=1-c[-1]/h.max(),
                     premarket_last60=c[-1]/o[-12]-1,
                     premarket_hour_acceleration=(c[-1]/o[-12]-1)-(c[-13]/o[-24]-1))
            if np.isfinite(prev_close) and not recent_action:
                r.update(premarket_gap=c[-1]/prev_close-1, premarket_range=width/prev_close)
            pre_vols = [p[:,4].sum() for p in history_pre[-20:] if p is not None and np.isfinite(p[:,4]).all() and (p[:,4]>=0).all()]
            rth_vols = [p[:,4].sum() for p in history_rth[-20:] if p is not None and np.isfinite(p[:,4]).all() and (p[:,4]>=0).all()]
            if np.isfinite(v).all() and (v>=0).all() and not recent_action:
                if len(pre_vols)>=10 and np.median(pre_vols)>0:
                    r['premarket_relative_volume']=v.sum()/np.median(pre_vols)
                if len(rth_vols)>=10 and np.median(rth_vols)>0:
                    r['premarket_volume_vs_regular']=v.sum()/np.median(rth_vols)
            if prev_post is not None and not recent_action:
                r['overnight_open_gap']=o[0]/prev_post[-1,3]-1
        if prev_post is not None and prev is not None and not recent_action:
            r.update(prev_postmarket_return=prev_post[-1,3]/prev[-1,3]-1,
                     prev_postmarket_range=(prev_post[:,1].max()-prev_post[:,2].min())/prev[-1,3])
        if prev is not None:
            r.update(prior1_session_return=prev[-1,3]/prev[0,0]-1,
                     prev_last60_return=prev[-1,3]/prev[-12,0]-1)
        if not recent_action:
            for n in (3,5,10,60):
                history=np.array(closes[-n-1:])
                if len(history)==n+1 and np.isfinite(history).all():
                    if n in (3,10,60):r[f'prior{n}_return']=history[-1]/history[0]-1
                    if n in (5,60):r[f'prior{n}_vol']=float(np.std(np.diff(np.log(history)),ddof=1))
            if len(history_rth)>=3 and all(p is not None for p in history_rth[-3:]):
                hourly=[]
                for p in history_rth[-3:]:
                    hourly.extend((p[j,0],p[min(j+11,len(p)-1),3]) for j in range(0,len(p),12))
                h=np.array(hourly)
                returns=h[:,1]/h[:,0]-1
                change=np.diff(np.r_[h[0,0],h[:,1]])
                r.update(prior3_hour_up_fraction=float((returns>0).mean()),
                         prior3_hour_efficiency=float(change.sum()/np.abs(change).sum()) if np.abs(change).sum()>0 else 0.,
                         prior3_hour_vol=float(np.std(returns,ddof=1)))
        rows.append({'date':day,**r,'premarket_complete':pre is not None})
        history_rth.append(regular)
        history_pre.append(pre)
        history_post.append(post)
        closes.append(regular[-1,3] if regular is not None else np.nan)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'frozen_candidates.json').exists():raise FileExistsError('Expanded run already frozen')
    from build_dataset import corporate_dates
    panel=pd.read_parquet(HERE/'features.parquet')
    cfg=json.loads((SOURCE/'config.json').read_text())
    sessions=json.loads((ROOT/cfg['calendar']).read_text())['sessions']
    records=[]
    for i,symbol in enumerate(sorted(panel.symbol.unique())):
        raw=pd.read_parquet(ROOT/cfg['bars_dir']/symbol/'2026.parquet')
        actions=corporate_dates(pd.read_parquet(ROOT/cfg['actions_dir']/(symbol+'.parquet')))
        records.extend({'symbol':symbol,**r} for r in build_symbol(raw,sessions,actions))
        if (i+1)%20==0:print(f'Premarket/history {i+1}/105',flush=True)
    extra=pd.DataFrame(records)
    q=extra[extra.symbol.eq('QQQ')][['date','premarket_return','premarket_gap']].rename(columns={'premarket_return':'qqq_pre_return','premarket_gap':'qqq_pre_gap'})
    extra=extra.merge(q,on='date',how='left',validate='many_to_one')
    extra['premarket_qqq_relative']=extra.premarket_return-extra.qqq_pre_return
    extra['qqq_premarket_gap']=extra.qqq_pre_gap
    extra.drop(columns=['qqq_pre_return','qqq_pre_gap'],inplace=True)
    panel=panel.merge(extra,on=['symbol','date'],how='left',validate='many_to_one')
    panel.to_parquet(OUT/'features.parquet',index=False,compression='zstd',compression_level=7)
    defs=json.loads((HERE/'feature_definitions.json').read_text())|EXTRA
    (OUT/'feature_definitions.json').write_text(json.dumps(defs,ensure_ascii=False,indent=2))
    protocol=(HERE/'PROTOCOL.md').read_text()+'''\n## 用户补充后的扩展（后段结果打开前）\n\n新增盘前04:00—09:30、前日盘后16:00—20:00、前3日小时聚合、前3/5/10/20/60日历史特征。没有20:00—04:00夜盘逐bar数据，仅保留跨时段端点跳空，不能称夜盘路径。盘前66根5m必须完整且价格有效，缺数据不补零；扩展时段不使用未经审计的VWAP。\n\n除原先时间切分候选外，追加最近30和60个完整标签已成熟的起始交易日独立探索，各自重新选条件和Top5。阈值沿用早期发现段目录，避免近期分位数漂移改变条件含义；每个近期候选至少20股票日、8日期。近期结果是窗口内探索比例，不当独立复核。另将同一固定规则同时放到最近60/最近30/此前30日，量化迁移；60日包含30日，比较变化用不重叠的前30/后30。所有58群尝试；小样本、只有单股、群成员变化单独报告。\n'''
    (OUT/'PROTOCOL.md').write_text(protocol)
    audit=json.loads((HERE/'dataset_audit.json').read_text())
    audit.update(expansion_nonmissing=panel[list(EXTRA)].notna().mean().to_dict(),expanded_feature_count=len(defs),
                 expansion_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 protocol_sha256=hashlib.sha256(protocol.encode()).hexdigest())
    (OUT/'dataset_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({'features':len(defs),'availability':audit['expansion_nonmissing']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
