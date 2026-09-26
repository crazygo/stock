"""Point-in-time features and separate multi-horizon outcomes; offline only."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.group_expectation_matrix.build import (calendar_grid, corporate_dates,
    file_hash, outcome_at, prepare_bars, write_json)

ROOT = Path(__file__).resolve().parents[2]
ET = "America/New_York"

def epoch(values):
    return pd.DatetimeIndex(values).as_unit("ns").asi8

def provenance(cfg):
    source = ROOT / cfg["group_source"]
    old = json.loads((source / "manifest.json").read_text())
    result = []
    for item in old["source_files"]:
        path = ROOT / item["path"]
        actual = file_hash(path)
        if actual != item["sha256"]:
            raise ValueError(f"Frozen group source changed: {path}")
        result.append({"path": item["path"], "sha256": actual})
    for name in ["groups.json", "memberships.json", "versions.json", "instruments.json", "manifest.json"]:
        path = source / name
        result.append({"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)})
    return result

def encode_segment(ohlcv, mask, reference, durations):
    """Each slot is independent; missing observations remain masked, never backfilled."""
    x = np.zeros((len(mask), 7), np.float32)
    good = mask & np.isfinite(ohlcv).all(axis=1) & (ohlcv[:, 0] > 0) & (ohlcv[:, 3] > 0)
    a = ohlcv[good]
    x[good, 0] = np.log(a[:, 3] / a[:, 0])
    x[good, 1] = np.log(a[:, 3] / reference)
    x[good, 2] = (a[:, 1] - a[:, 2]) / a[:, 0]
    x[good, 3] = np.log1p(np.maximum(a[:, 4], 0)) / 20
    x[good, 4] = np.log1p(np.maximum(a[:, 5], 0)) / 25
    x[:, 5] = np.asarray(durations) / 390
    x[:, 6] = good
    return x

def aggregate(a, mask, factor):
    chunks, good, durations = [], [], []
    for p in range(0, len(a), factor):
        z, m = a[p:p+factor], mask[p:p+factor]
        # Preserve the observed tail before cutoff (e.g. 55 minutes of the second
        # RTH hour). Internal missing bars still invalidate an aggregate.
        if p+factor >= len(a) and m.any():
            last = int(np.flatnonzero(m)[-1])+1
            z, m = z[:last], m[:last]
        ok = bool(m.all())
        chunks.append([z[0, 0], np.max(z[:, 1]), np.min(z[:, 2]), z[-1, 3],
                       np.sum(z[:, 4]), np.sum(z[:, 5])] if ok else [np.nan]*6)
        good.append(ok)
        durations.append(len(z)*5)
    return np.asarray(chunks), np.asarray(good), durations

def tabular(branches):
    out = []
    for x in branches:
        good = x[:, 6] > 0
        z = x[good, :5]
        out.append(float(good.mean()))
        if len(z):
            out.extend(np.concatenate([z[-1], z.mean(0), z.std(0), z.min(0), z.max(0), z.sum(0)]))
        else:
            out.extend([np.nan]*30)
    return np.asarray(out, np.float32)

def snapshot_segments(raw_frame, daily, day_index, sessions, cutoff, history=20):
    """Construct only from records available by cutoff, before resolving duplicates."""
    known = raw_frame[(raw_frame._available <= cutoff) & (raw_frame._end <= cutoff)].copy()
    known = known[~known._start.duplicated(keep=False)].set_index("_start")
    basis = known.price_basis.eq("NONE")
    known.loc[~basis, ["open", "high", "low", "close", "volume", "turnover"]] = np.nan
    def slots(start, count):
        grid = pd.date_range(start, periods=count, freq="5min")
        f = known.reindex(grid)
        a = f[["open", "high", "low", "close", "volume", "turnover"]].to_numpy(float)
        valid = (np.isfinite(a).all(1) & (a[:, 2] > 0) & (a[:, 4] >= 0) &
                 (a[:, 2] <= np.minimum(a[:, 0], a[:, 3])) &
                 (a[:, 1] >= np.maximum(a[:, 0], a[:, 3])))
        end_ok = epoch(f._end) == epoch(grid + pd.Timedelta(minutes=5))
        return a, valid & end_ok
    h = np.full((history, 6), np.nan)
    hm = np.zeros(history, bool)
    for j, s in enumerate(sessions[max(0, day_index-history):day_index], start=max(0, history-day_index)):
        opening, closing = pd.Timestamp(s["open_at"]), pd.Timestamp(s["close_at"])
        a, m = slots(opening, int((closing-opening).total_seconds()/300))
        if m.all():
            h[j] = [a[0, 0], a[:, 1].max(), a[:, 2].min(), a[-1, 3], a[:, 4].sum(), a[:, 5].sum()]
            hm[j] = True
    d = sessions[day_index]["session_date"]
    prev = sessions[day_index-1]["session_date"] if day_index else d
    post = slots(pd.Timestamp(prev+" 16:00", tz=ET).tz_convert("UTC"), 48)
    pre = slots(pd.Timestamp(d+" 04:00", tz=ET).tz_convert("UTC"), 66)
    intraday = slots(pd.Timestamp(d+" 09:30", tz=ET).tz_convert("UTC"), 24)
    if intraday[1].sum() < 12 or hm.sum() < 10:
        return None
    reference = intraday[0][intraday[1]][-1, 3]
    return [(h, hm), post, pre, intraday], reference

def build(cfg, output):
    source = ROOT / cfg["group_source"]
    groups = json.loads((source / "groups.json").read_text())
    instruments = json.loads((source / "instruments.json").read_text())
    symbols = sorted(x["symbol"] for x in instruments if x.get("candidate"))
    members = json.loads((source / "memberships.json").read_text())
    versions = {v["version_id"]: v for v in json.loads((source / "versions.json").read_text())}
    group_index = {g["group_id"]: i for i, g in enumerate(groups)}
    member_map = {}
    for m in members:
        v = versions[m["version_id"]]
        member_map.setdefault((m["symbol"], m["week_id"]), []).append((m, v))
    sessions = json.loads((ROOT/cfg["calendar"]).read_text())["sessions"]
    grid = calendar_grid(sessions)
    starts, ends = epoch(grid.start), epoch(grid.end)
    dates = grid.date.to_numpy()
    asof = pd.Timestamp(cfg["as_of"])
    arrays = {str(g): {s: [] for s in ["H", "post", "pre", "B"]} for g in cfg["granularities"]}
    tab = {str(g): [] for g in cfg["granularities"]}
    rows, labels, memberships, availability = [], [], [], []
    paths = {}
    for si, sym in enumerate(symbols):
        files = sorted((ROOT/cfg["bars_dir"]/sym).glob("*.parquet"))
        action_path = ROOT/cfg["actions_dir"]/f"{sym}.parquet"
        if not files or not action_path.exists():
            availability.append({"symbol": sym, "status": "missing_bars_or_actions", "feature_rows": 0})
            continue
        raw = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        raw["_start"] = pd.to_datetime(raw.start_at, utc=True)
        raw["_end"] = pd.to_datetime(raw.end_at, utc=True)
        raw["_available"] = pd.to_datetime(raw.available_at, utc=True)
        aligned, arr, valid, entry_ok, duplicates, known_n = prepare_bars(raw, grid, asof)
        avail = epoch(aligned.available_at)
        actions = corporate_dates(pd.read_parquet(action_path))
        volume = pd.to_numeric(aligned.volume, errors="coerce").to_numpy(float)
        daily = {}
        for d, ids in grid.groupby("date", sort=False).indices.items():
            if valid[ids].all():
                z = arr[ids]
                daily[d] = ([z[0, 0], z[:, 1].max(), z[:, 2].min(), z[-1, 3],
                             volume[ids].sum(), z[:, 4].sum()], pd.Timestamp(int(avail[ids].max()), tz="UTC"))
        # Path arrays support 5-minute mark-to-market without consulting labels to select stocks.
        paths[sym] = np.column_stack([arr[:, :4], valid.astype(float), entry_ok.astype(float)])
        before = len(rows)
        for di, session in enumerate(sessions):
            day = session["session_date"]
            if not cfg["start_date"] <= day <= cfg["source_end"]:
                continue
            cutoff = pd.Timestamp(day+" "+cfg["cutoff_et"], tz=ET).tz_convert("UTC")
            earliest = cutoff + pd.Timedelta(seconds=150)
            entry = earliest.ceil("5min")
            if entry >= pd.Timestamp(session["close_at"]):
                continue
            history_start = sessions[max(0, di-cfg["history_sessions"])]["session_date"]
            if any(history_start < d <= day for d in actions):
                continue
            snap = snapshot_segments(raw, daily, di, sessions, cutoff, cfg["history_sessions"])
            if snap is None:
                continue
            segments, reference = snap
            iso = pd.Timestamp(day).isocalendar()
            week = f"{iso.year}-W{iso.week:02d}"
            mask = np.zeros(len(groups), bool)
            mids = []
            for m, v in member_map.get((sym, week), []):
                if pd.Timestamp(v["feature_cutoff_at"]) > cutoff or pd.Timestamp(v["effective_from"]) > cutoff:
                    raise AssertionError("Future membership")
                if v.get("effective_to") and cutoff >= pd.Timestamp(v["effective_to"]):
                    raise AssertionError("Expired membership")
                mids.append(m["version_id"])
                for gid in m["group_ids"]:
                    mask[group_index[gid]] = True
            pos = int(np.searchsorted(starts, entry.value))
            rid = len(rows)
            rows.append({"row_id": rid, "symbol": sym, "date": day, "week_id": week,
                         "cutoff_at": cutoff.isoformat(), "entry_at": entry.isoformat(),
                         "pos": pos, "reference_price": reference, "version_ids": "|".join(mids)})
            memberships.append(mask)
            for gran in cfg["granularities"]:
                branches = []
                for bi, (a, m) in enumerate(segments):
                    if bi == 0:
                        aa, mm, duration = a, m, [390]*len(a)
                    else:
                        aa, mm, duration = aggregate(a, m, gran//5)
                    x = encode_segment(aa, mm, reference, duration)
                    arrays[str(gran)][["H", "post", "pre", "B"][bi]].append(x)
                    branches.append(x)
                tab[str(gran)].append(tabular(branches))
            for expectation in cfg["expectations"]:
                o = outcome_at(arr, valid, entry_ok, avail, starts, ends, dates, pos,
                               {"trading_days": expectation["days"], "target_return": expectation["target"]},
                               asof.value, actions, cfg["portfolio"]["cost_bps"]/10000)
                labels.append({"row_id": rid, "expectation_id": expectation["id"], **o})
        availability.append({"symbol": sym, "status": "available", "feature_rows": len(rows)-before,
                             "duplicate_bars": duplicates, "known_bars": known_n})
        if si % 10 == 0:
            print(f"features {si+1}/{len(symbols)} {sym}: {len(rows)} rows", flush=True)
    pd.DataFrame(rows).to_parquet(output/"rows.parquet", index=False)
    pd.DataFrame(labels).to_parquet(output/"labels.parquet", index=False)
    grid.to_parquet(output/"grid.parquet", index=False)
    np.savez_compressed(output/"memberships.npz", mask=np.asarray(memberships))
    np.savez_compressed(output/"paths.npz", **paths)
    for gran, parts in arrays.items():
        np.savez_compressed(output/f"features_{gran}.npz", tab=np.asarray(tab[gran]),
                            **{k: np.asarray(v) for k, v in parts.items()})
    write_json(output/"groups.json", groups)
    write_json(output/"availability.json", availability)
    print(f"DATA COMPLETE: {len(rows)} feature rows; {len(labels)} independent labels", flush=True)

def split_masks(rows, labels, fold):
    mature = labels.status.eq("mature").to_numpy()
    ready = pd.to_datetime(labels.label_available_at, utc=True)
    cutoff = pd.to_datetime(rows.cutoff_at, utc=True)
    fit_time, outer = pd.Timestamp(fold["fit_before"], tz="UTC"), pd.Timestamp(fold["outer_start"], tz="UTC")
    fit = mature & (cutoff < fit_time) & (ready < fit_time)
    cal = mature & (cutoff >= fit_time) & (cutoff < outer) & (ready < outer)
    test = rows.date.between(fold["outer_start"], fold["outer_end"])
    return np.asarray(fit), np.asarray(cal), np.asarray(test)
