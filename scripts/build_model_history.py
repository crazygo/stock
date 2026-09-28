#!/usr/bin/env python3
"""Validate and assemble annual 5m, derived hourly and RTH daily archives.

Partial aggregate bars retain counts and a false completeness flag, but their
OHLCV values are null. Nothing is interpolated. This is not a feature-training
job and never changes the source snapshots used by previous experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.backfill_model_history import DEST, START, END, MONTHS, SESSIONS, FIELDS, history_starts, quality, save, sha, symbols, write


def load_bars(symbol, start=START, end=END, root=DEST):
    """Read all required year partitions; reject duplicate or adjusted bars."""
    paths=[Path(root)/"us_5m"/symbol/f"{year}.parquet" for year in range(int(start[:4]),int(end[:4])+1)]
    listed=history_starts(root).get(symbol)
    missing=[str(p) for p in paths if not p.exists() and not (listed and p.stem+"-12-31"<listed)]
    if missing:raise FileNotFoundError("Missing year partitions: "+", ".join(missing))
    frames=[pd.read_parquet(p) for p in paths if p.exists()]
    if not frames:return pd.DataFrame(columns=["symbol","session_date","start_at","end_at","price_basis",*FIELDS])
    f=pd.concat(frames,ignore_index=True)
    f=f[(f.session_date>=start)&(f.session_date<=end)].sort_values("start_at")
    if len(f) and (set(f.price_basis)!={"NONE"} or f.start_at.duplicated().any()):
        raise ValueError("Mixed basis or duplicate cross-year source bars")
    return f.reset_index(drop=True)


def aggregate(f, daily=False):
    if f.empty:return pd.DataFrame()
    f=f.copy().sort_values("start_at")
    starts=pd.to_datetime(f.start_at,utc=True).dt.tz_convert("America/New_York")
    dates=starts.dt.strftime("%Y-%m-%d")
    m=starts.dt.hour*60+starts.dt.minute
    close=dates.map({d:570+s["duration_minutes"] for d,s in SESSIONS.items()}).fillna(960).astype(int)
    if daily:
        keep=dates.isin(SESSIONS)&(m>=570)&(m<close)
        f=f[keep].copy();dates=dates[keep];m=m[keep];close=close[keep]
        a=pd.Series(570,index=f.index);b=close
        kind=pd.Series("regular",index=f.index)
    else:
        a=pd.Series(np.select([m<240,m<570,m<close,m<1200],
            [(m//60)*60,240+((m-240)//60)*60,570+((m-570)//60)*60,close+((m-close)//60)*60],
            default=(m//60)*60),index=f.index).astype(int)
        b=pd.Series(np.select([m<240,m<570,m<close,m<1200],
            [np.minimum(a+60,240),np.minimum(a+60,570),np.minimum(a+60,close),np.minimum(a+60,1200)],
            default=a+60),index=f.index).astype(int)
        kind=pd.Series(np.select([m<240,m<570,m<close,m<1200],
            ["overnight","pre_market","regular","post_market"],default="overnight"),index=f.index)
    if f.empty:return pd.DataFrame()
    f["aggregate_date"]=dates;f["a"]=a;f["b"]=b;f["kind"]=kind
    v=f[FIELDS].to_numpy(float)
    f["valid"]=(np.isfinite(v).all(1)&(v[:,:4]>0).all(1)&(v[:,4:]>=0).all(1)&
                (v[:,1]>=v[:,[0,2,3]].max(1))&(v[:,2]<=v[:,[0,1,3]].min(1)))
    clock=pd.to_datetime(f.start_at,utc=True)
    f["valid"] &= clock.dt.minute.mod(5).eq(0)&clock.dt.second.eq(0)&clock.dt.microsecond.eq(0)
    out=f.groupby(["aggregate_date","a","b","kind"],sort=True).agg(
        symbol=("symbol","first"),open=("open","first"),high=("high","max"),low=("low","min"),
        close=("close","last"),volume=("volume","sum"),turnover=("turnover","sum"),
        source_bars=("start_at","size"),unique_bars=("start_at","nunique"),valid=("valid","all")).reset_index()
    out["duration_minutes"]=out.b-out.a
    out["expected_bars"]=out.duration_minutes//5
    out["is_complete"]=(out.source_bars==out.expected_bars)&(out.unique_bars==out.expected_bars)&out.valid
    out.loc[~out.is_complete,FIELDS]=np.nan
    op=(pd.to_datetime(out.aggregate_date)+pd.to_timedelta(out.a,unit="m")).dt.tz_localize("America/New_York")
    cl=(pd.to_datetime(out.aggregate_date)+pd.to_timedelta(out.b,unit="m")).dt.tz_localize("America/New_York")
    for key,val in [("start_at",op.dt.tz_convert("UTC")),("end_at",cl.dt.tz_convert("UTC")),
                    ("start_at_et",op),("end_at_et",cl),
                    ("available_at",(cl+pd.Timedelta(seconds=1)).dt.tz_convert("UTC"))]:
        out[key]=val.map(lambda x:x.isoformat())
    out["time_key"]=cl.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["price_basis"]="NONE";out["source"]="derived_from_futu_5m_NONE"
    return out.rename(columns={"aggregate_date":"session_date","kind":"session_type"}).drop(columns=["a","b","valid","unique_bars"])


def build():
    listing=history_starts()
    reports=[];objects=[];incomplete=[]
    for symbol in symbols():
        for year in (2023,2024,2025,2026):
            months=[m for m in MONTHS if m.startswith(str(year))]
            outstanding=[m for m in months if not (DEST/"parts"/symbol/m/"complete.json").exists()]
            incomplete.extend([f"{symbol}/{m}" for m in outstanding])
            paths=[DEST/"parts"/symbol/m/"bars.parquet" for m in months]
            frames=[pd.read_parquet(p) for p in paths if p.exists()]
            f=pd.concat(frames,ignore_index=True).sort_values("start_at") if frames else pd.DataFrame()
            start=max(START,f"{year}-01-01");end=min(END,f"{year}-12-31")
            q=quality(f,start,end,listing.get(symbol))
            if q["duplicates"] or (len(f) and q["basis"]!=["NONE"]):
                raise ValueError(f"Source integrity failure {symbol}/{year}: {q}")
            invalid_count=q["invalid_rows"]
            if invalid_count:
                v=f[FIELDS].to_numpy(float)
                valid=(np.isfinite(v).all(1)&(v[:,:4]>0).all(1)&(v[:,4:]>=0).all(1)&
                       (v[:,1]>=v[:,[0,2,3]].max(1))&(v[:,2]<=v[:,[0,1,3]].min(1)))
                save(f.loc[~valid],DEST/"quarantine"/symbol/f"{year}.parquet")
                f=f.loc[valid].copy()
                q=quality(f,start,end,listing.get(symbol))
            q["quarantined_invalid_source_rows"]=invalid_count
            if len(f):
                outputs={"us_5m":f,"derived_60m":aggregate(f),"derived_day":aggregate(f,daily=True)}
                for folder,frame in outputs.items():
                    dest=DEST/folder/symbol/f"{year}.parquet";save(frame,dest)
                    objects.append(dict(path=str(dest.relative_to(DEST)),rows=len(frame),sha256=sha(dest),bytes=dest.stat().st_size))
            reports.append(dict(symbol=symbol,year=year,listing_date=listing.get(symbol),pending_months=outstanding,**q))
        print(json.dumps({"assembled":symbol,"securities_done":len(reports)//4}),flush=True)
    summary=dict(dataset_id=DEST.name,requested_start=START,requested_end=END,
                 symbols=len(symbols()),annual_partitions=len(objects),
                 acquisition_complete=not incomplete,pending_months=incomplete,
                 total_5m_rows=sum(x["rows"] for x in objects if x["path"].startswith("us_5m/")),
                 full_rth_symbol_years=sum(x["rth_complete"] for x in reports),
                 missing_rth_bars=sum(x["missing_rth_bars"] for x in reports),
                 source_quality="Unadjusted; no synthetic or interpolated bars; extended-session completeness is not certified",
                 availability_quality="historical assumed bar end plus 1s, not observed historical delivery",
                 universe_mode="current candidate snapshot, retrospective, not point-in-time",
                 all_2026_dates_previously_exposed=True,trained_models=False,
                 files=objects,coverage=reports)
    write(DEST/"manifest.json",summary)
    write(DEST/"coverage.json",reports)
    lines=["# 五模型历史行情数据验收", "",f"请求范围：{START} 至 {END}（ET），108 只候选股加 QQQ。", "",
           "2023 年从 4 月开始，仅作为 2024 年的特征预热；不构造上市前行情。", "",
           f"采集请求完成：{not incomplete}；待完成月分区：{len(incomplete)}。",
           f"5 分钟数据：{summary['total_5m_rows']:,} 行；经上市日期过滤后仍缺少 {summary['missing_rth_bars']:,} 根常规盘柱。", "",
           "小时线与日线从同一份 NONE 五分钟行情聚合；日线只包含常规盘，小时线不跨交易时段。",
           "小时或日聚合缺少任何基础柱时，is_complete=false，OHLCV 留空，不以 0 或插值冒充真实行情。", "",
           "早年盘前、盘后、夜盘仅归档来源实际返回的柱，不保证 24 小时每个时槽都有历史数据。",
           "当前股票池为回溯快照；不代表 2023–2025 当时可知的成分股。已有实验输入及结果未改写，未启动训练。", "",
           "逐股票逐年份覆盖、上市日、常规盘缺口及各时段柱数见 coverage.json；来源与文件 SHA-256 见 manifest.json。"]
    (DEST/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"files","coverage","pending_months"}}),flush=True)


if __name__=="__main__":
    build()
