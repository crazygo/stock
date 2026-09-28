#!/usr/bin/env python3
"""Offline, versioned stock-group × expectation research.

Run from the repository root with the research Python environment. No network,
broker, or scheduling APIs are used. Output directories are immutable by default.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def scalar(value):
    return None if not np.isfinite(value) else round(float(value), 8)


def calendar_grid(sessions):
    parts = []
    for s in sessions:
        starts = pd.date_range(s["open_at"], s["close_at"], freq="5min", inclusive="left").tz_convert("UTC")
        parts.append(pd.DataFrame({"start": starts, "end": starts + pd.Timedelta(minutes=5), "date": s["session_date"]}))
    grid = pd.concat(parts, ignore_index=True)
    return grid


def corporate_dates(frame):
    """Cash dividends are not split events; unsupported capital changes excluded."""
    fields = ["split_ratio", "join_ert", "bonus_ert", "per_share_div_ratio", "transfer_ert", "per_share_trans_ratio", "allotment_ratio", "stk_spo_ratio", "spin_off_ratio"]
    mask = np.zeros(len(frame), dtype=bool)
    for k in fields:
        if k in frame:
            mask |= pd.to_numeric(frame[k], errors="coerce").fillna(0).ne(0).to_numpy()
    return sorted(frame.loc[mask, "ex_div_date"].astype(str).str[:10].tolist())


def unavailable_outcome(reason="market_data_unavailable"):
    """Explicit unknown label for an entity without the required 5m/action data."""
    return {"status": "missing", "reason": reason, "hit": None, "entry_price": None,
            "label_end_at": None, "label_available_at": None, "terminal_return": None,
            "mfe": None, "mae": None, "hit_minutes": None, "holding_minutes_proxy": None,
            "net_proxy": None}


def validate_refinement(taxonomy):
    """Validate either flat replacement groups or explicit parent group entities."""
    if "refinement" not in taxonomy:
        return
    groups = {g["group_id"]: g for g in taxonomy["groups"]}
    mode = taxonomy.get("refinement_mode", "replace")
    if mode not in ("replace", "parent_entities"):
        raise ValueError("Unknown refinement mode")
    retain_parents = mode == "parent_entities"
    seen_sources, seen_groups = set(), set()
    for r in taxonomy["refinement"]:
        source, ids = r["source_group_id"], r["subgroup_ids"]
        if source in seen_sources or (not retain_parents and source in groups):
            raise ValueError("Refinement source must be unique and absent from flat groups")
        seen_sources.add(source)
        if retain_parents:
            parent = groups.get(source)
            if not parent or parent.get("group_kind") != "parent" or parent.get("parent_group_id") is not None:
                raise ValueError("Refinement parent must be a root group entity")
            if parent["name"] != r["source_name"] or parent.get("child_group_ids") != ids:
                raise ValueError("Parent entity name or child relations are inconsistent")
            if parent.get("membership_rule") != "union_of_child_memberships":
                raise ValueError("Parent membership must use a deduplicated child union")
        if not 2 <= len(ids) <= 3 or len(ids) != len(set(ids)):
            raise ValueError("Each source requires 2–3 unique subgroups")
        symbols = set()
        for gid in ids:
            if gid not in groups or gid in seen_groups:
                raise ValueError("Each subgroup must exist and belong to one refinement source")
            seen_groups.add(gid)
            g = groups[gid]
            if retain_parents and (g.get("parent_group_id") != source or g.get("group_kind") != "subgroup" or g.get("child_group_ids") != []):
                raise ValueError("Subgroup parent entity relation is inconsistent")
            if g.get("derived_from_group_id") != source or not g.get("segment") or g["name"] != r["source_name"] + " - " + g["segment"]:
                raise ValueError("Flat subgroup name or refinement lineage is inconsistent")
            if not g["symbols"] or g.get("children") or g.get("subgroups"):
                raise ValueError("Subgroups require members and must remain flat")
            supported = {s for x in g.get("sources", []) if x.get("url") and x.get("observed_at") for s in x.get("supports", [])}
            if not set(g["symbols"]) <= supported:
                raise ValueError("Every refined member requires an observed source")
            symbols.update(g["symbols"])
        if symbols != set(r["source_symbols"]):
            raise ValueError("Refinement must preserve the source membership union")
        if retain_parents and set(groups[source]["symbols"]) != symbols:
            raise ValueError("Parent members must equal the deduplicated child union")
    if (seen_groups | (seen_sources if retain_parents else set())) != set(groups):
        raise ValueError("All flat groups must be covered by refinement")


def validate_inputs(cfg, taxonomy, universe, calendar, etf_snapshot=None):
    def unique(values, name):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name}")
    unique([e["expectation_id"] for e in cfg["expectations"]], "expectation_id")
    unique([g["group_id"] for g in taxonomy["groups"]], "group_id")
    validate_refinement(taxonomy)
    unique([m["symbol"] for m in universe["members"]], "symbol")
    unique([s["session_date"] for s in calendar["sessions"]], "calendar session")
    unique(cfg["trend_windows"], "trend window")
    for e in cfg["expectations"]:
        if type(e["trading_days"]) is not int or e["trading_days"] < 1 or not 0 < e["target_return"] < 10:
            raise ValueError("Expectations require positive integer days and positive return")
    for n in cfg["trend_windows"]+[cfg["volatility_window"], cfg["liquidity_window"]]:
        if type(n) is not int or n < 2:
            raise ValueError("Lookback must be an integer >= 2")
    for key in ["volatility_annual_cuts", "liquidity_dollars_cuts"]:
        cuts = cfg[key]
        if len(cuts) != 2 or not 0 < cuts[0] < cuts[1]:
            raise ValueError("Cuts must contain two ascending positive values")
    candidates = {m["symbol"] for m in universe["members"] if m["role"] == "candidate"}
    if etf_snapshot is not None:
        etfs = {m["symbol"] for m in etf_snapshot["members"]}
        if len(etfs) != len(etf_snapshot["members"]) or not etfs or not etfs <= candidates:
            raise ValueError("ETF snapshot members must be unique candidate entities")
        if any(next(m for m in universe["members"] if m["symbol"] == s).get("instrument_type") != "etf" for s in etfs):
            raise ValueError("ETF group members must have ETF instrument type")
    for g in taxonomy["groups"]:
        unique(g["symbols"], f"members in {g['group_id']}")
        if not g["group_id"].startswith("industry:") or not set(g["symbols"]) <= candidates:
            raise ValueError("Industry groups must have industry: IDs and candidate-only members")
    sessions = calendar["sessions"]
    if [s["session_date"] for s in sessions] != sorted(s["session_date"] for s in sessions):
        raise ValueError("Calendar must be ordered")
    if pd.Timestamp(cfg["evaluation_as_of"]).tzinfo is None:
        raise ValueError("evaluation_as_of requires a timezone")
    if cfg["end_date"] < cfg["start_date"] or cfg["trend_z_threshold"] <= 0:
        raise ValueError("Invalid dates or trend threshold")


def align_bars(raw, grid):
    raw = raw.copy()
    raw["start"] = pd.to_datetime(raw["start_at"], utc=True)
    raw["end"] = pd.to_datetime(raw["end_at"], utc=True)
    raw["available"] = pd.to_datetime(raw["available_at"], utc=True)
    # Duplicate timestamps are invalid, never silently choose a favorable revision.
    duplicate = raw["start"].duplicated(keep=False)
    raw = raw.loc[~duplicate].set_index("start")
    aligned = raw.reindex(pd.DatetimeIndex(grid["start"]))
    arr = aligned[["open", "high", "low", "close", "turnover"]].to_numpy(float)
    o, h, l, c = (arr[:, i] for i in range(4))
    valid = np.isfinite(arr[:, :4]).all(axis=1) & (l > 0) & (l <= np.minimum(o, c)) & (h >= np.maximum(o, c)) & (h >= l)
    valid &= aligned["end"].to_numpy() == grid["end"].to_numpy()
    valid &= aligned["price_basis"].eq("NONE").to_numpy()
    valid &= aligned["available"].notna().to_numpy()
    valid &= (aligned["available"] >= aligned["end"]).to_numpy()
    # A bar reported at zero volume is not a tradable entry proxy.
    entry_ok = valid & pd.to_numeric(aligned["volume"], errors="coerce").gt(0).to_numpy()
    return aligned, arr, valid, entry_ok, int(duplicate.sum())


def prepare_bars(raw, grid, asof):
    """Keep known prices; late-only availability metadata explains pending labels.

    A later revision cannot erase a known bar. Late prices never enter daily
    features or outcomes; only their availability timestamp is used in auditing.
    """
    times = pd.to_datetime(raw.available_at, utc=True)
    known = raw.loc[times <= asof].copy()
    aligned, arr, valid, entry_ok, duplicates = align_bars(known, grid)
    late = raw.loc[times > asof].copy()
    if len(late):
        late["start"] = pd.to_datetime(late.start_at, utc=True)
        late["available"] = pd.to_datetime(late.available_at, utc=True)
        first_late = late.groupby("start").available.min().reindex(aligned.index)
        aligned["available"] = aligned.available.fillna(first_late)
    return aligned, arr, valid, entry_ok, duplicates, len(known)


def daily_history(grid, aligned, arr, valid):
    rows = []
    for day, idx in grid.groupby("date", sort=True).indices.items():
        ids = np.asarray(idx)
        ok = bool(valid[ids].all())
        available = aligned.iloc[ids]["available"].max() if ok else pd.NaT
        turnover = arr[ids, 4]
        rows.append({"date": day, "close": float(arr[ids[-1], 3]) if ok else np.nan,
                     "dollar_volume": float(turnover.sum()) if ok and np.isfinite(turnover).all() and (turnover >= 0).all() else np.nan,
                     "available": available, "complete": ok})
    return pd.DataFrame(rows).set_index("date")


def classify_history(daily, prior_dates, cutoff, strategy, actions):
    n = strategy["lookback_sessions"]
    need = n if strategy["family"] == "liquidity" else n + 1
    dates = prior_dates[-need:]
    if len(dates) < need:
        return None, "insufficient_history", {}
    if any(dates[0] < a <= dates[-1] for a in actions):
        return None, "corporate_action_in_lookback", {}
    d = daily.reindex(dates)
    if not d["complete"].fillna(False).all():
        return None, "incomplete_history", {}
    if not (d["available"] < cutoff).all():
        return None, "history_not_available", {}
    if strategy["family"] == "liquidity":
        value = float(d.dollar_volume.median()) if d.dollar_volume.notna().all() else np.nan
        if not np.isfinite(value):
            return None, "missing_turnover", {}
        label = ["low", "medium", "high"][int(value >= strategy["cuts"][0]) + int(value >= strategy["cuts"][1])]
        return label, "classified", {"median_daily_turnover": value}
    rets = np.diff(np.log(d.close.to_numpy(float)))
    sigma = float(np.std(rets, ddof=1))
    value = sigma * np.sqrt(252)
    facts = {"return": scalar(np.expm1(rets.sum())), "annualized_volatility": scalar(value), "history_end": dates[-1]}
    if strategy["family"] == "volatility":
        label = ["low", "medium", "high"][int(value >= strategy["cuts"][0]) + int(value >= strategy["cuts"][1])]
        return label, "classified", facts
    displacement = float(rets.sum())
    z = displacement / (sigma * np.sqrt(n)) if sigma > 1e-12 else (0.0 if abs(displacement) < 1e-12 else np.sign(displacement) * 1e9)
    label = "up" if z > strategy["threshold"] else "down" if z < -strategy["threshold"] else "range"
    facts["trend_z"] = scalar(z)
    return label, "classified", facts


def outcome_at(arr, valid, entry_ok, available_ns, starts, ends, dates, pos, expectation, asof_ns, actions, cost):
    count = int(expectation["trading_days"] * 78)
    last = pos + count - 1
    result = {"status": "pending", "reason": None, "hit": None, "entry_price": None,
              "label_end_at": None, "label_available_at": None, "terminal_return": None,
              "mfe": None, "mae": None, "hit_minutes": None, "holding_minutes_proxy": None, "net_proxy": None}
    if last >= len(arr):
        result["reason"] = "future_calendar_not_covered"
        return result
    result["label_end_at"] = pd.Timestamp(ends[last], tz="UTC").isoformat()
    if ends[last] + 1_000_000_000 > asof_ns:
        result["reason"] = "full_horizon_not_mature"
        return result
    if any(dates[pos] < a <= dates[last] for a in actions):
        result.update(status="corporate_action", reason="unsupported_capital_change")
        return result
    ids = slice(pos, last + 1)
    if (available_ns[ids] > asof_ns).any():
        result["reason"] = "horizon_not_yet_available"
        return result
    if not entry_ok[pos]:
        result.update(status="missing", reason="entry_missing_invalid_or_zero_volume")
        return result
    if not valid[ids].all():
        result.update(status="missing", reason="incomplete_or_invalid_horizon")
        return result
    ready = int(available_ns[ids].max())
    if ready > asof_ns:
        result["reason"] = "horizon_not_yet_available"
        return result
    entry = arr[pos, 0]
    hit_positions = np.flatnonzero(arr[ids, 1] >= entry * (1 + expectation["target_return"]))
    hit = bool(len(hit_positions))
    terminal = arr[last, 3] / entry - 1
    gross = expectation["target_return"] if hit else terminal
    result.update(status="mature", reason=None, hit=int(hit), entry_price=float(entry),
                  label_available_at=pd.Timestamp(ready, tz="UTC").isoformat(),
                  terminal_return=float(terminal), mfe=max(0., float(arr[ids, 1].max() / entry - 1)),
                  mae=min(0., float(arr[ids, 2].min() / entry - 1)),
                  hit_minutes=int((hit_positions[0] + 1) * 5) if hit else None,
                  holding_minutes_proxy=int((hit_positions[0] + 1) * 5) if hit else count * 5,
                  net_proxy=float((1 + gross) * (1 - cost) / (1 + cost) - 1))
    return result


def es95_loss(values):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        return None
    mass = len(values) * .05
    n = int(np.floor(mass))
    total = float(values[:n].sum())
    if n < len(values):
        total += (mass - n) * values[n]
    return max(0., -total / mass)


def describe(frame):
    mature = frame[frame.status == "mature"]
    n = len(mature)
    res = {"sample_count": int(len(frame)), "mature_n": n, "hit_n": int(mature.hit.sum()),
           "pending_n": int((frame.status == "pending").sum()),
           "missing_n": int((~frame.status.isin(["mature", "pending"])).sum()),
           "status_counts": {str(k): int(v) for k, v in frame.status.value_counts().items()},
           "stock_n": int(mature.symbol.nunique()), "issuer_n": int(mature.issuer.nunique()),
           "date_n": int(mature.date.nunique()), "date_from": mature.date.min() if n else None,
           "date_to": mature.date.max() if n else None}
    if not n:
        return res
    res.update(hit_rate=scalar(mature.hit.mean()), stock_equal_hit_rate=scalar(mature.groupby("symbol").hit.mean().mean()),
               issuer_equal_hit_rate=scalar(mature.groupby("issuer").hit.mean().mean()),
               max_stock_share=scalar(mature.symbol.value_counts().max()/n),
               mean_terminal_return=scalar(mature.terminal_return.mean()), mean_net_proxy=scalar(mature.net_proxy.mean()),
               es95_loss=scalar(es95_loss(mature.net_proxy)), mae_q10=scalar(mature.mae.quantile(.1)),
               mean_holding_days_proxy=scalar(mature.holding_minutes_proxy.mean()/390),
               hit_median_days=scalar(mature.loc[mature.hit == 1, "hit_minutes"].median()/390))
    return res


def paired_baseline(cell, baseline, all_dates, expectation, cfg, bootstrap):
    """Date-matched rate; same resampled calendar blocks for group and baseline."""
    g = cell[cell.status == "mature"].groupby("date").hit.agg(["sum", "count"])
    b = baseline[baseline.status == "mature"].groupby("date").hit.mean()
    if g.empty:
        return {}
    common = g.index.intersection(b.index)
    g = g.loc[common]
    expected = b.loc[common] * g["count"]
    n = g["count"].sum()
    r = {"baseline_rate": scalar(expected.sum()/n), "lift": scalar((g["sum"].sum()-expected.sum())/n),
         "lift_ci95": None, "ci_status": "not_requested_for_weekly_scope"}
    if not bootstrap:
        return r
    block = max(cfg["bootstrap_block_sessions"], expectation["trading_days"] + 1)
    if len(g) < max(cfg["minimum_inference_dates"], block*4) or cell.loc[cell.status == "mature", "symbol"].nunique() < cfg["minimum_inference_stocks"]:
        r["ci_status"] = "insufficient_dates_or_stocks"
        return r
    date_axis = [d for d in all_dates if g.index.min() <= d <= g.index.max()]
    sums = g["sum"].reindex(date_axis, fill_value=0).to_numpy(float)
    counts = g["count"].reindex(date_axis, fill_value=0).to_numpy(float)
    expected = expected.reindex(date_axis, fill_value=0).to_numpy(float)
    seed = cfg["seed"] + int(digest([expectation["expectation_id"], sorted(cell.symbol.unique())])[:8], 16)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(date_axis)-block+1, size=(cfg["bootstrap_replicates"], int(np.ceil(len(date_axis)/block))))
    indices = (starts[:, :, None] + np.arange(block)).reshape(cfg["bootstrap_replicates"], -1)[:, :len(date_axis)]
    den = counts[indices].sum(axis=1)
    lifts = ((sums-expected)[indices].sum(axis=1) / np.where(den > 0, den, np.nan))
    r.update(lift_ci95=[scalar(x) for x in np.nanquantile(lifts, [.025, .975])],
             ci_status="exploratory_uncorrected", block_sessions=block)
    return r


def make_definitions(cfg, taxonomy, etf_snapshot=None):
    strategies = [{"strategy_id": "all", "name": "全池参照", "family": "baseline", "definition_version": "1.0", "cadence": "weekly"}]
    groups = [{"group_id": "all:all", "strategy_id": "all",
               "name": "全部候选证券" if etf_snapshot else "全部候选股",
               "description": "当前证券名单回溯，含 FutuD ETF 组" if etf_snapshot else "当前101股名单回溯；不含基准ETF"}]
    names = {15: "3 周", 63: "3 月", 126: "6 月"}
    for window in cfg["trend_windows"]:
        sid = f"trend_{window}"
        strategies.append({"strategy_id": sid, "family": "trend", "name": f"{names.get(window, str(window))}趋势", "lookback_sessions": window, "definition_version": "1.0", "cadence": "weekly", "threshold": cfg["trend_z_threshold"]})
        for label, name in [("up", "上行"), ("range", "震荡"), ("down", "下行")]:
            groups.append({"group_id": f"{sid}:{label}", "strategy_id": sid, "name": f"{names.get(window, str(window))}·{name}", "description": f"{window}交易日，周前冻结，标准化位移阈值±{cfg['trend_z_threshold']}"})
    for family, label, cuts in [("volatility", "20 日波动", cfg["volatility_annual_cuts"]), ("liquidity", "20 日流动性", cfg["liquidity_dollars_cuts"])]:
        label = f"{cfg[f'{family}_window']} 日" + ("波动" if family == "volatility" else "流动性")
        strategies.append({"strategy_id": family, "family": family, "name": label, "lookback_sessions": cfg[f"{family}_window"], "cuts": cuts, "definition_version": "1.0", "cadence": "weekly"})
        for level, name in [("low", "低"), ("medium", "中"), ("high", "高")]:
            groups.append({"group_id": f"{family}:{level}", "strategy_id": family, "name": f"{label}·{name}", "description": f"固定分界：{cuts}"})
    strategies.append({"strategy_id": "industry", "family": "industry", "name": "行业与价值链", "definition_version": taxonomy["taxonomy_version"], "cadence": "weekly", "membership_basis": "current_snapshot_retrospective", "source_observed_at": taxonomy["generated_at"]})
    for g in taxonomy["groups"]:
        groups.append({**{k: v for k, v in g.items() if k != "symbols"}, "strategy_id": "industry"})
    if etf_snapshot is not None:
        strategies.append({"strategy_id": "watchlist_etf", "family": "watchlist", "name": "FutuD · ETF 分组",
                           "definition_version": etf_snapshot["observed_at"], "cadence": "weekly",
                           "membership_basis": "current_watchlist_snapshot_retrospective",
                           "source_observed_at": etf_snapshot["observed_at"]})
        groups.append({"group_id": etf_snapshot["group_id"], "strategy_id": "watchlist_etf",
                       "group_kind": "watchlist", "name": "ETF（FutuD 分组）",
                       "description": "FutuD 自定义 ETF 分组的当前成员快照；历史归属为回溯展示。",
                       "member_symbols": [m["symbol"] for m in etf_snapshot["members"]],
                       "source_observed_at": etf_snapshot["observed_at"]})
    return strategies, groups


def build(config_path, output):
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a fresh path: {output}")
    cfg = json.loads(config_path.read_text())
    etf_path = ROOT/cfg["etf_group_snapshot"] if cfg.get("etf_group_snapshot") else None
    etf_snapshot = json.loads(etf_path.read_text()) if etf_path else None
    taxonomy_path = ROOT / cfg["industry_tags"]
    taxonomy = json.loads(taxonomy_path.read_text())
    universe = json.loads((ROOT / cfg["universe"]).read_text())
    symbols = sorted(m["symbol"] for m in universe["members"] if m["role"] == "candidate")
    calendar = json.loads((ROOT / cfg["calendar"]).read_text())
    validate_inputs(cfg, taxonomy, universe, calendar, etf_snapshot)
    sessions = calendar["sessions"]
    grid = calendar_grid(sessions)
    all_dates = [s["session_date"] for s in sessions]
    dates = grid.date.to_numpy()
    starts = pd.DatetimeIndex(grid.start).as_unit("ns").asi8
    ends = pd.DatetimeIndex(grid.end).as_unit("ns").asi8
    asof = pd.Timestamp(cfg["evaluation_as_of"])
    asof_ns = asof.as_unit("ns").value
    generated = datetime.now(timezone.utc).isoformat()
    strategies, groups = make_definitions(cfg, taxonomy, etf_snapshot)
    weeks = {}
    sample_days = []
    for s in sessions:
        if not cfg["start_date"] <= s["session_date"] <= cfg["end_date"]:
            continue
        day = s["session_date"]
        iso = datetime.fromisoformat(day).isocalendar()
        week = f"{iso.year}-W{iso.week:02d}"
        # First trading session of this week, even when the requested interval starts midweek.
        first = next(x for x in sessions if datetime.fromisoformat(x["session_date"]).isocalendar()[:2] == iso[:2])
        weeks.setdefault(week, {"week_id": week, "effective_from": first["open_at"], "feature_cutoff_at": first["open_at"], "first_session": first["session_date"]})
        cutoff = pd.Timestamp(day + " " + cfg["decision_time_et"], tz=ET).tz_convert("UTC")
        if cutoff >= pd.Timestamp(s["close_at"]):
            continue
        earliest = cutoff + pd.Timedelta(seconds=cfg["generation_delay_seconds"]+cfg["manual_delay_seconds"])
        entry = earliest.ceil("5min")
        if entry >= pd.Timestamp(s["close_at"]) or entry-earliest > pd.Timedelta(minutes=cfg["entry_max_wait_minutes"]):
            continue
        pos = int(np.searchsorted(starts, entry.as_unit("ns").value))
        if pos >= len(starts) or starts[pos] != entry.as_unit("ns").value:
            raise ValueError("Entry does not align to official calendar")
        sample_days.append({"date": day, "week_id": week, "entry_at": entry.isoformat(), "decision_at": (cutoff+pd.Timedelta(seconds=cfg["generation_delay_seconds"])).isoformat(), "pos": pos})
    provenance = []
    for p in [config_path, taxonomy_path, ROOT/cfg["universe"], ROOT/cfg["calendar"]]+([etf_path] if etf_path else []):
        provenance.append({"path": str(p.relative_to(ROOT)), "sha256": file_hash(p)})
    memberships, outcomes, audits = [], [], []
    industry_for = {sym: [g["group_id"] for g in taxonomy["groups"] if sym in g["symbols"]] for sym in symbols}
    etf_symbols = {m["symbol"] for m in etf_snapshot["members"]} if etf_snapshot else set()
    unavailable_symbols = set()
    for si, sym in enumerate(symbols):
        files = sorted((ROOT/cfg["bars_dir"]/sym).glob("*.parquet"))
        action_path = ROOT/cfg["actions_dir"]/f"{sym}.parquet"
        data_ready = bool(files and action_path.exists())
        if not data_ready and sym not in etf_symbols:
            raise FileNotFoundError(f"Missing bars or corporate action metadata for {sym}")
        if data_ready:
            for p in files+[action_path]:
                provenance.append({"path": str(p.relative_to(ROOT)), "sha256": file_hash(p)})
            raw = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
            action_dates = corporate_dates(pd.read_parquet(action_path))
            aligned, arr, valid, entry_ok, dup, known_rows = prepare_bars(raw, grid, asof)
            daily = daily_history(grid, aligned, arr, valid)
            available_ns = pd.DatetimeIndex(aligned.available).as_unit("ns").asi8
            coverage = {"raw_rows": len(raw), "known_raw_rows": known_rows, "valid_regular_bars": int(valid.sum()),
                        "duplicate_rows": dup, "complete_days": int(daily.complete.sum()),
                        "capital_change_dates": action_dates, "market_data_status": "local_bars_and_actions_present"}
        else:
            unavailable_symbols.add(sym)
            action_dates = []
            daily = pd.DataFrame({"close": np.nan, "dollar_volume": np.nan, "complete": False,
                                  "available": pd.NaT}, index=all_dates)
            coverage = {"raw_rows": 0, "known_raw_rows": 0, "valid_regular_bars": 0, "duplicate_rows": 0,
                        "complete_days": 0, "capital_change_dates": [], "market_data_status": "5m_or_action_metadata_unavailable"}
        audits.append({"symbol": sym, **coverage})
        for week, meta in weeks.items():
            prior_dates = [d for d in all_dates if d < meta["first_session"]]
            cutoff = pd.Timestamp(meta["feature_cutoff_at"])
            for strategy in strategies:
                sid = strategy["strategy_id"]
                row = {"week_id": week, "strategy_id": sid, "symbol": sym, "issuer": "ALPHABET" if sym in ("GOOG", "GOOGL") else sym}
                if sid == "all":
                    gids, status, facts = ["all:all"], "classified", {}
                elif sid == "industry":
                    gids, status, facts = industry_for[sym], "classified" if industry_for[sym] else "unclassified", {}
                elif sid == "watchlist_etf":
                    gids, status, facts = ([etf_snapshot["group_id"]], "classified", {"snapshot_observed_at": etf_snapshot["observed_at"]}) if sym in etf_symbols else ([], "not_in_group", {})
                else:
                    level, status, facts = classify_history(daily, prior_dates, cutoff, strategy, action_dates)
                    gids = [f"{sid}:{level}"] if level else []
                row.update(group_ids=gids, status=status, facts=facts)
                memberships.append(row)
        for sd in sample_days:
            for expectation in cfg["expectations"]:
                out = (outcome_at(arr, valid, entry_ok, available_ns, starts, ends, dates, sd["pos"], expectation, asof_ns, action_dates, cfg["cost_bps_per_side"]/10000)
                       if data_ready else unavailable_outcome())
                outcomes.append({"sample_id": f"{sym}:{sd['date']}:{cfg['decision_time_et']}", "symbol": sym,
                                 "issuer": "ALPHABET" if sym in ("GOOG", "GOOGL") else sym,
                                 **{k: v for k, v in sd.items() if k != "pos"}, "expectation_id": expectation["expectation_id"], **out})
        if si % 20 == 0 or si == len(symbols)-1:
            print(f"Built {si+1}/{len(symbols)} stocks", flush=True)
    versions = []
    for strategy in strategies:
        sid = strategy["strategy_id"]
        for week, meta in weeks.items():
            rows = [m for m in memberships if m["strategy_id"] == sid and m["week_id"] == week]
            fingerprint = digest([strategy, meta, rows])[:12]
            vid = f"{sid}:{week}:{fingerprint}"
            for m in rows:
                m["version_id"] = vid
            versions.append({**meta, "version_id": vid, "strategy_id": sid, "definition_version": strategy["definition_version"],
                             "generated_at": generated, "classified_stocks": sum(bool(m["group_ids"]) for m in rows),
                             "membership_basis": ("current_snapshot_retrospective" if sid == "industry" else
                                                  "current_watchlist_snapshot_retrospective" if sid == "watchlist_etf" else "causal_weekly_reconstruction"),
                             "source_observed_at": taxonomy["generated_at"] if sid == "industry" else
                                                   etf_snapshot["observed_at"] if sid == "watchlist_etf" else None})
    for v in versions:
        next_week = next((w for w in weeks.values() if w["first_session"] > v["first_session"]), None)
        v["effective_to"] = next_week["effective_from"] if next_week else None
    odf = pd.DataFrame(outcomes)
    mdf = pd.DataFrame(memberships)
    exploded = mdf.loc[mdf.group_ids.map(bool), ["week_id", "symbol", "strategy_id", "version_id", "group_ids"]].explode("group_ids").rename(columns={"group_ids": "group_id"})
    joined = odf.merge(exploded, on=["week_id", "symbol"], how="inner", validate="many_to_many")
    # Per-strategy eligibility is deduplicated before creating a baseline for overlapping industry groups.
    eligible = odf.merge(exploded[["week_id", "symbol", "strategy_id"]].drop_duplicates(), on=["week_id", "symbol"])
    all_by = {eid: f for eid, f in odf.groupby("expectation_id")}
    eligible_by = {(sid, eid): f for (sid, eid), f in eligible.groupby(["strategy_id", "expectation_id"])}
    exps = {e["expectation_id"]: e for e in cfg["expectations"]}
    groups_by = {g["group_id"]: g for g in groups}
    cells, stock_cells, month_cells = [], [], []
    for (gid, eid), frame in joined.groupby(["group_id", "expectation_id"], sort=True):
        sid = groups_by[gid]["strategy_id"]
        cell = {"scope": "all", "group_id": gid, "expectation_id": eid, **describe(frame)}
        cell["market_comparison"] = paired_baseline(frame, all_by[eid], all_dates, exps[eid], cfg, True)
        cell["eligible_comparison"] = paired_baseline(frame, eligible_by[(sid, eid)], all_dates, exps[eid], cfg, True)
        cells.append(cell)
        for sym, sub in frame.groupby("symbol"):
            stock_cells.append({"group_id": gid, "expectation_id": eid, "symbol": sym, **describe(sub)})
        for month, sub in frame.groupby(frame.date.str[:7]):
            month_cells.append({"group_id": gid, "expectation_id": eid, "month": month, **describe(sub),
                                "market_comparison": paired_baseline(sub, all_by[eid], all_dates, exps[eid], cfg, False)})
        for week, sub in frame.groupby("week_id"):
            cells.append({"scope": week, "group_id": gid, "expectation_id": eid, **describe(sub),
                          "market_comparison": paired_baseline(sub, all_by[eid], all_dates, exps[eid], cfg, False),
                          "eligible_comparison": paired_baseline(sub, eligible_by[(sid, eid)], all_dates, exps[eid], cfg, False)})
    # Retain zero-member / zero-mature cells in every version scope.
    present = {(c["scope"], c["group_id"], c["expectation_id"]) for c in cells}
    for scope in ["all"]+list(weeks):
        for g in groups:
            for e in cfg["expectations"]:
                if (scope, g["group_id"], e["expectation_id"]) not in present:
                    cells.append({"scope": scope, "group_id": g["group_id"], "expectation_id": e["expectation_id"], **describe(odf.iloc[:0]), "market_comparison": {}, "eligible_comparison": {}})
    for item in provenance:
        if file_hash(ROOT/item["path"]) != item["sha256"]:
            raise RuntimeError(f"Source changed while building: {item['path']}; rerun into a fresh directory")
    output.mkdir(parents=True)
    coverage_by_symbol = {a["symbol"]: a["market_data_status"] for a in audits}
    instruments = [{"security_id": m["security_id"], "symbol": m["symbol"],
                    "name": m.get("name", m["symbol"]), "instrument_type": m.get("instrument_type", "stock"),
                    "roles": m.get("roles", [m["role"]]), "candidate": m["role"] == "candidate",
                    "watchlist_group_ids": m.get("watchlist_group_ids", []),
                    "market_data_status": coverage_by_symbol.get(m["symbol"], "not_in_matrix_candidate_pool")}
                   for m in universe["members"]]
    manifest = {"schema_version": "1.0", "protocol_version": cfg["protocol_version"], "generated_at": generated,
                "evaluation_as_of": cfg["evaluation_as_of"], "sample_start": cfg["start_date"], "sample_end": cfg["end_date"],
                "candidate_stocks": len(symbols), "issuer_count": len(set("ALPHABET" if s in ("GOOG", "GOOGL") else s for s in symbols)),
                "sampling": f"one_stock_day_at_{cfg['decision_time_et']}_ET", "outcome_rows": len(odf), "unique_stock_days": int(odf.sample_id.nunique()),
                "universe_mode": universe["universe_mode"], "historical_bar_availability": "assumed_end_plus_1_second",
                "price_basis": "NONE", "group_count": len(groups), "strategy_count": len(strategies),
                "week_count": len(weeks), "version_count": len(versions), "expectation_count": len(exps),
                "industry_classified_stocks": sum(bool(v) for v in industry_for.values()),
                "industry_group_count": len(taxonomy["groups"]),
                "industry_refinement_source_count": len(taxonomy.get("refinement", [])),
                "industry_parent_group_count": sum(g.get("group_kind") == "parent" for g in taxonomy["groups"]),
                "industry_subgroup_count": sum(g.get("group_kind") == "subgroup" for g in taxonomy["groups"]),
                "stock_candidate_count": sum(m["role"] == "candidate" and m.get("instrument_type", "stock") == "stock" for m in universe["members"]),
                "etf_candidate_count": len(etf_symbols),
                "etf_market_data_ready_count": len(etf_symbols - unavailable_symbols),
                "etf_market_data_unavailable_count": len(unavailable_symbols),
                "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                "config_sha256": file_hash(config_path), "source_files": provenance,
                "code_files": [{"path": str(p.relative_to(ROOT)), "sha256": file_hash(p)} for p in sorted(HERE.glob("*.py"))],
                "limitations": ["当前名单回溯，存在幸存者偏差", "历史每周版本为今日因果重放，非当年真实运行", "行业标签为当前业务快照回溯，未覆盖股票保持未分类", "多标签群重叠，GOOG/GOOGL为同一发行人", "触及率是历史描述，不是未来校准概率", "净评价是目标触及/期末估值与固定成本代理，不是组合实盘收益", "区间仅探索性，未做多重检验校正", "无中途止损；MAE是入场相对全窗口浮亏，非账户回撤", "6月趋势需127个完整收盘，早期与缺历史样本不可分类"] +
                               ([f"FutuD ETF分组为当前快照回溯；{len(unavailable_symbols)}只ETF缺5m行情或公司行动元数据，只能保留实体与缺失状态，不能当成失败样本", "全池现混合股票与ETF；ETF发行人字段使用基金代码代理；反向/杠杆ETF按自身价格路径观察，不能用标的指数收益替代"] if etf_snapshot else [])}
    bundle = {"manifest": {k:v for k,v in manifest.items() if k not in ("source_files", "code_files")},
              "config": cfg, "strategies": strategies, "groups": groups, "expectations": cfg["expectations"],
              "weeks": list(weeks.values()), "versions": versions, "memberships": memberships,
              "cells": cells, "stock_cells": stock_cells, "month_cells": month_cells,
              "instruments": instruments, "watchlist_snapshot": etf_snapshot,
              "industry_refinement": taxonomy.get("refinement", []),
              "group_relations": [{"parent_group_id": g["parent_group_id"], "child_group_id": g["group_id"], "relation_type": "subgroup_of"} for g in groups if g.get("parent_group_id")]}
    for name, obj in [("manifest", manifest), ("strategies", strategies), ("groups", groups), ("expectations", cfg["expectations"]),
                      ("versions", versions), ("memberships", memberships), ("matrix", cells), ("stock_metrics", stock_cells),
                      ("monthly_metrics", month_cells), ("coverage", audits), ("bundle", bundle),
                      ("instruments", instruments),
                      ("group_relations", bundle["group_relations"]),
                      ("matrix_all", {"manifest": bundle["manifest"], "groups": groups, "expectations": cfg["expectations"], "cells": [c for c in cells if c["scope"] == "all"], "group_relations": bundle["group_relations"]})]:
        write_json(output/f"{name}.json", obj)
    odf.to_parquet(output/"outcomes.parquet", compression="zstd", compression_level=7, index=False)
    write_json(output/"config.json", cfg)
    write_json(output/"industry_tags.json", taxonomy)
    if etf_snapshot:
        write_json(output/"watchlist_snapshot.json", etf_snapshot)
    (output/"PROTOCOL.md").write_text((HERE/"PROTOCOL.md").read_text())
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("source_files", "code_files", "limitations")}, ensure_ascii=False), flush=True)
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"config/default.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.config.resolve(), args.output.resolve())
