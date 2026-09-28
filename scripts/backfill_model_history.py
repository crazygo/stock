#!/usr/bin/env python3
"""Resumable model-history acquisition, Local -> R2 -> single locked OpenD.

No training, orders or automatic cloud writes. Historical receipt timestamps
are audit metadata, never represented as historical real-time availability.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.model_history_calendar import calendar
from scripts.r2_client import R2Client

DEST = ROOT / "market_data/model_training_history_v1"
CACHE = ROOT / ".cache/model_history_20260927"
V8 = ROOT / "research/after_open_3d5pct/runs/focus_v8_20260927"
R03 = ROOT / "research/after_open_3d5pct/runs/precision_v9_20260927/R03"
OPTICS = ROOT / "research/strategy_group_lab/runs/optics_3round_20260926/market_data"
START, END = "2023-04-01", "2026-09-25"
FIELDS = ["open", "high", "low", "close", "volume", "turnover"]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False, compression="zstd", compression_level=7)
    temp.replace(path)


def symbols():
    u = json.loads((V8 / "inputs/market_data/universe/qqq_retrospective_v1.json").read_text())
    all_symbols = {x["symbol"] for x in u["members"] if x["role"] == "candidate"} | {"QQQ"}
    groups = json.loads((ROOT / "research/after_open_3d5pct/focus_v8/protocol.json").read_text())["groups"]
    focus = set(sum(groups.values(), [])) | {"QQQ"}
    return sorted(focus) + sorted(all_symbols - focus)


CAL = calendar()
SESSIONS = {s["session_date"]: s for s in CAL["sessions"]}
MONTHS = pd.period_range(START[:7], END[:7], freq="M").astype(str).tolist()


def history_starts(root=DEST):
    path=Path(root)/"security_metadata.json"
    rows=json.loads(path.read_text()) if path.exists() else []
    out={r["code"].removeprefix("US."):str(r["listing_date"])[:10] for r in rows
         if r.get("listing_date") and str(r["listing_date"])[:10]>"1970-01-01"}
    overrides=Path(root)/"security_history.json"
    if overrides.exists():
        out.update({symbol:r["required_from"] for symbol,r in json.loads(overrides.read_text()).items()})
    return out


def normalize(frame, symbol):
    """Vectorized 5m normalization; midnight and half days are explicit."""
    if frame.empty:
        return pd.DataFrame()
    f = frame.copy()
    if f.time_key.duplicated().any():
        raise ValueError(f"Duplicate provider timestamp for {symbol}")
    end = pd.to_datetime(f.time_key).dt.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise")
    start = end - pd.Timedelta(minutes=5)
    date = end.dt.strftime("%Y-%m-%d")
    a = start.dt.hour*60 + start.dt.minute
    b = end.dt.hour*60 + end.dt.minute
    close = date.map({d: 570+s["duration_minutes"] for d,s in SESSIONS.items()}).fillna(960)
    same = start.dt.date == end.dt.date
    f["symbol"] = symbol
    for key, val in [("start_at", start.dt.tz_convert("UTC")), ("end_at", end.dt.tz_convert("UTC")),
                     ("available_at", (end+pd.Timedelta(seconds=1)).dt.tz_convert("UTC")),
                     ("start_at_et", start), ("end_at_et", end)]:
        f[key] = val.map(lambda x: x.isoformat())
    f["session_date"] = date
    f["session_type"] = np.select([same&(a>=570)&(b<=close), same&(a>=240)&(b<=570),
                                   same&(a>=close)&(b<=1200)],
                                  ["regular", "pre_market", "post_market"], default="overnight")
    f["price_basis"] = "NONE"
    for c in FIELDS + ["pe_ratio", "turnover_rate", "change_rate", "last_close"]:
        f[c] = pd.to_numeric(f[c], errors="raise").astype(float) if c in f else np.nan
    cols = ["symbol", "time_key", "start_at", "end_at", "available_at", "start_at_et", "end_at_et",
            "session_date", "session_type", *FIELDS, "pe_ratio", "turnover_rate", "change_rate", "last_close", "price_basis"]
    return f[cols].sort_values("start_at").reset_index(drop=True)


def quality(f, start, end, listed=None):
    sessions = [s for d,s in SESSIONS.items() if start <= d <= end and (not listed or d >= listed)]
    expected = pd.DatetimeIndex([x for s in sessions for x in pd.date_range(
        s["open_at"], periods=s["duration_minutes"]//5, freq="5min")])
    if f.empty:
        actual = pd.DatetimeIndex([], tz="UTC")
        bad = duplicates = 0
        basis = []
    else:
        f = f[(f.session_date>=start)&(f.session_date<=end)]
        actual = pd.DatetimeIndex(pd.to_datetime(f.start_at, utc=True))
        v = f[FIELDS].to_numpy(float)
        bad = int((~np.isfinite(v).all(1)|(v[:,0]<=0)|(v[:,2]<=0)|(v[:,4]<0)|(v[:,5]<0)|
                   (v[:,1]<np.max(v[:,[0,2,3]],axis=1))|(v[:,2]>np.min(v[:,[0,1,3]],axis=1))).sum())
        duplicates = int(f.start_at.duplicated().sum())
        basis = sorted(f.price_basis.unique().tolist())
    missing = expected.difference(actual)
    missing_dates = Counter(missing.tz_convert("America/New_York").strftime("%Y-%m-%d")) if len(missing) else {}
    return dict(rows=len(f), required_sessions=len(sessions), expected_rth_bars=len(expected),
                missing_rth_bars=len(missing), missing_by_date=dict(missing_dates), invalid_rows=bad,
                duplicates=duplicates, basis=basis, rth_complete=not len(missing) and not bad and not duplicates,
                session_counts=f.session_type.value_counts().to_dict() if len(f) else {},
                first_time=f.time_key.min() if len(f) else None, last_time=f.time_key.max() if len(f) else None)


def import_frame(frame, symbol, source, fresh_provider=False):
    if frame.empty:
        return
    if set(frame.price_basis.dropna()) != {"NONE"}:
        raise ValueError(f"Wrong price basis: {source}")
    # Normalize a separate version, never mutate the source or experiment inputs.
    frame = normalize(frame, symbol)
    record = {"source":str(source), "sha256":sha(source)}
    for month, part in frame.groupby(frame.session_date.str[:7]):
        if month not in MONTHS:
            continue
        dest = DEST / "parts" / symbol / month / "bars.parquet"
        lineage = dest.parent/"imports.json"
        items = json.loads(lineage.read_text()) if lineage.exists() else []
        if record in items:
            continue
        if dest.exists():
            old = pd.read_parquet(dest)
            common = old.merge(part, on="start_at", suffixes=("_old", "_new"))
            conflict = np.any([~np.isclose(common[c+"_old"], common[c+"_new"], rtol=0, atol=1e-8, equal_nan=True) for c in FIELDS],axis=0)
            if conflict.any():
                if not fresh_provider:
                    raise ValueError(f"Conflicting observations: {symbol}/{month}: {source}")
                # A new ALL response can revise a previously archived extended
                # feed. Preserve both values and explicitly select the new
                # response only in this independent dataset version.
                evidence=CACHE/"source_revisions"/symbol/month
                evidence.mkdir(parents=True,exist_ok=True)
                save(common.loc[conflict],evidence/"differences.parquet")
                write(evidence/"resolution.json",dict(previous_sha256=sha(dest),new_source=str(source),
                      new_source_sha256=sha(source),differing_rows=int(conflict.sum()),
                      policy="fresh complete NONE ALL provider response; original shared and experiment files unchanged"))
            part = pd.concat([old, part]).drop_duplicates("start_at", keep="last" if fresh_provider else "first").sort_values("start_at")
        save(part, dest)
        if record not in items:
            items.append(record)
            write(lineage, items)


def prepare():
    DEST.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    write(DEST/"calendar.json", CAL)
    write(DEST/"registration.json", dict(dataset_id=DEST.name, symbols=symbols(), source_start=START,
          training_start="2024-01-02", end=END, interval="5m", session="ALL", price_basis="NONE",
          warmup_reason="126 daily candles + 20-session relative-volume history and six weekly group states",
          universe_basis="current 108 candidate snapshot plus QQQ; retrospective, not point-in-time membership",
          availability_basis="historical_assumed_bar_end_plus_1s; request receipt only in audit",
          no_training=True, no_automatic_cloud_writes=True, script_sha256=sha(Path(__file__))))
    for symbol in symbols():
        for base in [ROOT/"market_data", OPTICS]:
            for year in (2023,2024,2025,2026):
                src = base/"us_5m"/symbol/f"{year}.parquet"
                # The shared store was freshly audited through Sep 25. Older
                # frozen optics snapshots can differ in extended-session feed
                # semantics. Use them only when the shared year is absent.
                preferred = ROOT/"market_data/us_5m"/symbol/f"{year}.parquet"
                if src.exists() and (base == ROOT/"market_data" or not preferred.exists()):
                    import_frame(pd.read_parquet(src), symbol, src)
            ca = base/"corporate_actions"/f"{symbol}.parquet"
            if ca.exists() and not (DEST/"corporate_actions"/ca.name).exists():
                (DEST/"corporate_actions").mkdir(exist_ok=True)
                shutil.copy2(ca, DEST/"corporate_actions"/ca.name)
        for done in (R03/"parts"/symbol).glob("*/result.json"):
            result = json.loads(done.read_text())
            src = done.parent/"bars.parquet"
            if result["pagination_complete"] and src.exists():
                import_frame(pd.read_parquet(src), symbol, src)
                write(DEST/"parts"/symbol/done.parent.name/"complete.json",
                      dict(source=str(done), source_sha256=sha(done), pagination_complete=True))
    client = R2Client()
    inventory = client.list_objects("us_5m/")
    write(CACHE/"r2_inventory.json", inventory)
    wanted = set(symbols())
    for obj in inventory:
        key = obj["key"]
        pieces = key.split("/")
        if len(pieces)!=3 or pieces[1] not in wanted or pieces[2] not in {"2023.parquet","2024.parquet","2025.parquet"}:
            continue
        cache = CACHE/"r2"/key
        if not cache.exists():
            client.get_object(key, cache)
        import_frame(pd.read_parquet(cache), pieces[1], cache)
    print(json.dumps({"status":"prepared", "symbols":len(wanted), "completed_r03_parts":len(list(DEST.glob('parts/*/*/complete.json')))}), flush=True)


def acquire():
    import futu as ft
    lock = open("/tmp/stock_futu_acquisition.lock", "a")
    print('{"status":"waiting_for_shared_futu_lock"}', flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    # Re-read completed output from the earlier task after it releases the lock.
    prepare()
    ft.SysConfig.enable_proto_encrypt(False)
    quote = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
    results = []
    try:
        ret, basic = quote.get_stock_basicinfo(ft.Market.US, ft.SecurityType.STOCK, code_list=["US."+s for s in symbols()])
        listing = {}
        if ret == ft.RET_OK:
            write(DEST/"security_metadata.json", basic.to_dict("records"))
            listing = history_starts()
        ret, q = quote.get_history_kl_quota(get_detail=False)
        if ret == ft.RET_OK: write(CACHE/"quota_before.json", {"used":int(q[0]),"remaining":int(q[1])})
        for symbol in symbols():
            for year in (2025,2024,2023,2026):
                missing = []
                for month in [m for m in MONTHS if m.startswith(str(year))]:
                    p = DEST/"parts"/symbol/month
                    if (p/"complete.json").exists(): continue
                    start = max(month+"-01",START)
                    end = min(str(pd.Period(month).end_time.date()),END)
                    if (p/"bars.parquet").exists():
                        f = pd.read_parquet(p/"bars.parquet")
                        if quality(f,start,end,listing.get(symbol))["rth_complete"]:
                            write(p/"complete.json",dict(source="existing_local_or_r2",pagination_complete=True));continue
                    if listing.get(symbol) and listing[symbol] > end:
                        write(p/"complete.json",dict(source="security_metadata_and_official_history",status="before_current_regular_trading",required_from=listing[symbol],pagination_complete=True));continue
                    missing.append(month)
                # Consecutive months share a paginated request, without requesting existing months again.
                groups = []
                for month in missing:
                    if not groups or pd.Period(month) != pd.Period(groups[-1][-1])+1: groups.append([])
                    groups[-1].append(month)
                for months in groups:
                    start=max(months[0]+"-01",START);end=min(str(pd.Period(months[-1]).end_time.date()),END)
                    part=CACHE/"requests"/symbol/f"{start}_{end}"/f"attempt_{time.time_ns()}";part.mkdir(parents=True,exist_ok=False)
                    pages=[];cursor=None;audit=[];failure=None
                    for page in range(160):
                        time.sleep(1.2 if page==0 else .3)
                        for attempt in range(3):
                            sent=datetime.now(timezone.utc).isoformat()
                            ret,chunk,nxt=quote.request_history_kline("US."+symbol,start=start,end=end,
                                ktype=ft.KLType.K_5M,autype=ft.AuType.NONE,max_count=1000,extended_time=True,
                                session=ft.Session.ALL,page_req_key=cursor)
                            event=dict(symbol=symbol,start=start,end=end,page=page+1,attempt=attempt+1,
                                       requested_at=sent,received_at=datetime.now(timezone.utc).isoformat(),ok=ret==ft.RET_OK)
                            if ret==ft.RET_OK:
                                raw=part/f"raw_{page+1:03d}.parquet";save(chunk,raw)
                                event.update(rows=len(chunk),sha256=sha(raw),has_more=nxt is not None);audit.append(event);break
                            event["error"]=str(chunk)[:350];audit.append(event)
                            if attempt<2:time.sleep(30)
                        write(part/"audit.json",audit)
                        if ret!=ft.RET_OK:failure=str(chunk)[:350];break
                        pages.append(chunk);cursor=nxt
                        if cursor is None:break
                    if cursor is not None and not failure:failure="page_cap"
                    if failure:
                        result=dict(symbol=symbol,start=start,end=end,status="failed",error=failure)
                        results.append(result);write(part/"result.json",result)
                        print(json.dumps(result),flush=True);continue
                    raw=pd.concat(pages,ignore_index=True) if pages else pd.DataFrame()
                    f=normalize(raw,symbol)
                    normalized=part/"normalized.parquet"
                    if len(f):save(f,normalized);import_frame(f,symbol,normalized,fresh_provider=True)
                    for month in months:
                        write(DEST/"parts"/symbol/month/"complete.json",dict(source=str(part),pagination_complete=True,
                              status="source_returned" if len(f) else "source_empty",request_received_at=audit[-1]["received_at"]))
                    result=dict(symbol=symbol,start=start,end=end,status="fetched",quality=quality(f,start,end,listing.get(symbol)))
                    results.append(result);write(part/"result.json",result)
                    write(DEST/"progress.json",dict(status="acquiring",results=results))
                    print(json.dumps({k:v for k,v in result.items() if k!="quality"}|{"rows":len(f),"missing_rth":result["quality"]["missing_rth_bars"]}),flush=True)
        ret,q=quote.get_history_kl_quota(get_detail=False)
        if ret==ft.RET_OK:write(CACHE/"quota_after.json",{"used":int(q[0]),"remaining":int(q[1])})
        write(DEST/"progress.json",dict(status="acquisition_finished",results=results))
    finally:
        quote.close();fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["prepare","acquire"])
    args=parser.parse_args()
    {"prepare":prepare,"acquire":acquire}[args.command]()
