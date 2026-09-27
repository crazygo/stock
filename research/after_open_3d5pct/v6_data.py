"""Offline, source-frozen multiscale data for the exposed v6 development run."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SLOTS = 192  # ET 04:00--20:00; overnight is a real gap, never a zero-volume bar.
HOUR_BINS = [(a, min(a + 12, 66), 0) for a in range(0, 66, 12)] + [
    (a, min(a + 12, 144), 1) for a in range(66, 144, 12)] + [
    (a, a + 12, 2) for a in range(144, 192, 12)]
STRATEGIES = ("trend_15", "trend_63", "trend_126", "volatility", "liquidity")
PRICE_COLUMNS = ("open", "high", "low", "close", "volume", "turnover")
FEATURE_COLUMNS = ("bar_return", "high_from_open", "low_from_open", "cross_bar_return",
                   "log_dollar", "relative_dollar", "duration_fraction", "session_code",
                   "et_position", "gap_log_hours", "valid", "relative_dollar_valid",
                   "log_price", "log_share_volume")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _iso(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_convert("UTC")


def _calendar(config: dict) -> list[dict]:
    sessions = json.loads((ROOT / config["calendar"]).read_text())["sessions"]
    return [s for s in sessions if "2026-01-01" <= s["session_date"] <= "2026-09-24"]


def _issuer(symbol: str) -> str:
    return "GOOG" if symbol in ("GOOG", "GOOGL") else symbol


def _encode(raw: np.ndarray, durations: np.ndarray, types: np.ndarray,
            positions: np.ndarray, start_ns: np.ndarray, prior_scale: np.ndarray) -> np.ndarray:
    """Encode price/volume levels without fitting a full-sample scaler."""
    flat = raw.reshape(-1, 6)
    valid = np.isfinite(flat[:, :4]).all(axis=1) & (flat[:, 0] > 0) & (flat[:, 2] > 0)
    out = np.zeros((len(flat), len(FEATURE_COLUMNS)), np.float32)
    prev_close, prev_end = np.nan, np.nan
    for i in np.flatnonzero(valid):
        o, h, l, c, _, dollar = flat[i]
        out[i, 0:3] = np.log(np.maximum([c, h, l], 1e-8) / o)
        out[i, 4] = np.log1p(max(0, dollar)) / 20
        out[i, 12] = np.log(o) / 5
        out[i, 13] = np.log1p(max(0, flat[i, 4])) / 15
        scale = prior_scale.reshape(-1)[i]
        if np.isfinite(scale) and scale > 0:
            out[i, 5] = np.clip(np.log1p(max(0, dollar)) - np.log1p(scale), -5, 5)
            out[i, 11] = 1
        if np.isfinite(prev_close) and prev_close > 0:
            out[i, 3] = np.clip(np.log(o / prev_close), -1, 1)
        if np.isfinite(prev_end):
            out[i, 9] = np.log1p(max(0, (start_ns.reshape(-1)[i] - prev_end) / 3.6e12)) / 5
        prev_close = c
        prev_end = start_ns.reshape(-1)[i] + durations.reshape(-1)[i] * 60e9
    out[:, 6] = durations.reshape(-1) / 60
    out[:, 7] = types.reshape(-1)
    out[:, 8] = positions.reshape(-1)
    out[:, 10] = valid
    out[~valid, :6] = 0
    out[~valid, 9] = 0
    return out.reshape(*raw.shape[:-1], len(FEATURE_COLUMNS))


def _prior_median(dollar: np.ndarray, lookback: int = 20) -> np.ndarray:
    result = np.full(dollar.shape, np.nan, np.float32)
    for d in range(1, len(dollar)):
        past = dollar[max(0, d-lookback):d]
        with np.errstate(all="ignore"):
            result[d] = np.nanmedian(past, axis=0)
    return result


@dataclass
class SymbolData:
    raw5: np.ndarray
    effective_available: np.ndarray
    seq5: np.ndarray
    seq60: np.ndarray
    seqday: np.ndarray
    daily_full: np.ndarray
    cutoff_return: np.ndarray
    cutoff_rvol: np.ndarray
    source: str
    invalid: int


def load_symbol(symbol: str, sessions: list[dict], config: dict) -> SymbolData | None:
    path = ROOT / config["source_dir"] / symbol / "2026.parquet"
    if not path.exists():
        return None
    import pyarrow.parquet as pq
    columns = ["session_date", "start_at_et", "end_at", "available_at",
               "session_type", "price_basis", *PRICE_COLUMNS]
    if "received_at" in pq.read_schema(path).names:
        columns.append("received_at")
    frame = pd.read_parquet(path, columns=columns)
    if set(frame.price_basis.dropna()) != {"NONE"}:
        raise ValueError(f"{symbol}: mixed price basis")
    if frame.duplicated(["session_date", "start_at_et"]).any():
        raise ValueError(f"{symbol}: duplicate source bars")
    dates = [s["session_date"] for s in sessions]
    date_ix = {d: i for i, d in enumerate(dates)}
    raw = np.full((len(dates), SLOTS, 6), np.nan, np.float32)
    availability = np.full((len(dates), SLOTS), np.datetime64("NaT"), dtype="datetime64[ns]")
    invalid = 0
    for row in frame.itertuples(index=False):
        d = date_ix.get(row.session_date)
        if d is None:
            continue
        hour, minute = map(int, row.start_at_et[11:16].split(":"))
        slot = (hour - 4) * 12 + minute // 5
        if not 0 <= slot < SLOTS:
            continue
        actual_start = pd.Timestamp(row.start_at_et).tz_convert("UTC")
        actual_end = pd.Timestamp(row.end_at).tz_convert("UTC")
        expected = "pre_market" if slot < 66 else "regular" if slot < 144 else "post_market"
        values = np.array([getattr(row, c) for c in PRICE_COLUMNS], dtype=np.float32)
        if (row.session_type != expected or actual_end != actual_start + pd.Timedelta(minutes=5) or
            not np.isfinite(values).all() or
            values[0] <= 0 or values[2] <= 0 or values[1] < max(values[0], values[2], values[3]) or
            values[2] > min(values[0], values[1], values[3]) or values[4] < 0):
            invalid += 1
            continue
        raw[d, slot] = values
        observed = pd.Timestamp(row.available_at).tz_convert("UTC")
        if "received_at" in columns and pd.notna(row.received_at):
            observed = max(observed, pd.Timestamp(row.received_at).tz_convert("UTC"))
        availability[d, slot] = np.datetime64(observed.tz_localize(None))
    # Source data can contain after-hours slots on half days; official RTH close controls labels.
    starts = np.empty((len(dates), SLOTS), np.float64)
    for d, s in enumerate(sessions):
        base = pd.Timestamp(s["open_at_et"]).normalize() + pd.Timedelta(hours=4)
        starts[d] = pd.date_range(base, periods=SLOTS, freq="5min").tz_convert("UTC").asi8
    durations = np.full((len(dates), SLOTS), 5, np.float32)
    types = np.broadcast_to(np.r_[np.zeros(66), np.ones(78), np.full(48, 2)], (len(dates), SLOTS))
    positions = np.broadcast_to(np.arange(SLOTS) / SLOTS, (len(dates), SLOTS))
    seq5 = _encode(raw, durations, types, positions, starts, _prior_median(raw[:, :, 5]))
    # Each hour stays inside one ET session; short 09:00 and 15:30 bars retain duration.
    raw60 = np.full((len(dates), len(HOUR_BINS), 6), np.nan, np.float32)
    dur60 = np.zeros((len(dates), len(HOUR_BINS)), np.float32)
    type60 = np.zeros_like(dur60)
    pos60 = np.zeros_like(dur60)
    start60 = np.zeros_like(dur60, np.float64)
    for j, (a, b, kind) in enumerate(HOUR_BINS):
        block = raw[:, a:b]
        good = np.isfinite(block[:, :, :4]).all(axis=(1, 2))
        for d in np.flatnonzero(good):
            raw60[d, j] = [block[d, 0, 0], np.max(block[d, :, 1]), np.min(block[d, :, 2]),
                           block[d, -1, 3], np.sum(block[d, :, 4]), np.sum(block[d, :, 5])]
        dur60[:, j] = (b-a)*5
        type60[:, j] = kind
        pos60[:, j] = a / SLOTS
        start60[:, j] = starts[:, a]
    seq60 = _encode(raw60, dur60, type60, pos60, start60, _prior_median(raw60[:, :, 5]))
    daily = np.full((len(dates), 6), np.nan, np.float32)
    full = np.zeros(len(dates), bool)
    cutoff_return = np.full(len(dates), np.nan, np.float32)
    cutoff_rvol = np.full(len(dates), np.nan, np.float32)
    daily_vol = np.full(len(dates), np.nan, np.float32)
    prefix_vol = np.full(len(dates), np.nan, np.float32)
    for d, s in enumerate(sessions):
        n = int(s["duration_minutes"]) // 5
        reg = raw[d, 66:66+n]
        full[d] = len(reg) == n and np.isfinite(reg[:, :4]).all()
        if full[d]:
            daily[d] = [reg[0, 0], np.max(reg[:, 1]), np.min(reg[:, 2]), reg[-1, 3],
                        np.sum(reg[:, 4]), np.sum(reg[:, 5])]
            daily_vol[d] = daily[d, 5]
        pre = raw[d, 66:90]  # 09:30--11:30, only completed bars.
        if np.isfinite(pre[:, :4]).all():
            cutoff_return[d] = np.log(pre[-1, 3] / pre[0, 0])
            prefix_vol[d] = np.sum(pre[:, 5])
        past = prefix_vol[max(0, d-20):d]
        med = np.nanmedian(past) if np.isfinite(past).sum() >= 5 else np.nan
        if np.isfinite(med) and med > 0 and np.isfinite(prefix_vol[d]):
            cutoff_rvol[d] = prefix_vol[d] / med
    seqday = _encode(daily[:, None, :], np.array([[s["duration_minutes"] for s in sessions]], np.float32).T,
                     np.full((len(dates), 1), 1), np.zeros((len(dates), 1)),
                     np.array([[_iso(s["open_at"]).value] for s in sessions], np.float64),
                     _prior_median(daily[:, None, 5]))[:, 0]
    # The source clock is archived availability, not observed delivery.
    # A 126-day encoded candle can contain a relative-volume denominator
    # reaching a further 20 prior sessions back.
    daily_max_available = availability.view("int64").max(axis=1)
    for d, s in enumerate(sessions):
        cutoff = pd.Timestamp(s["open_at"]) + pd.Timedelta(hours=2)
        decision = cutoff + pd.Timedelta(seconds=config["decision_latency_seconds"])
        observed_past_max = daily_max_available[max(0, d-146):d].max(initial=np.iinfo("int64").min)
        observed_current_max = availability[d, :90].view("int64").max(initial=np.iinfo("int64").min)
        if (pd.Timestamp(s["close_at"]) <= cutoff or observed_past_max > decision.value or
            observed_current_max > decision.value):
            cutoff_return[d] = np.nan
            cutoff_rvol[d] = np.nan
    return SymbolData(raw, availability, seq5, seq60, seqday, full, cutoff_return, cutoff_rvol,
                      str(path.relative_to(ROOT)), invalid)


def _group_map(config: dict, dates: list[str], symbols: list[str]) -> tuple[dict, dict]:
    root = ROOT / config["group_run"]
    versions = json.loads((root / "versions.json").read_text())
    membership = json.loads((root / "memberships.json").read_text())
    valid = {(v["week_id"], v["strategy_id"]): v for v in versions
             if v["strategy_id"] in STRATEGIES and v["membership_basis"] == "causal_weekly_reconstruction"}
    own = {}
    peer = defaultdict(set)
    for m in membership:
        if m["symbol"] not in symbols or m["strategy_id"] not in STRATEGIES:
            continue
        v = valid.get((m["week_id"], m["strategy_id"]))
        if v is None or m["version_id"] != v["version_id"] or len(m["group_ids"]) != 1:
            continue
        key = (m["week_id"], m["strategy_id"], m["symbol"])
        own[key] = (m["group_ids"][0], m["facts"], v)
        peer[(m["week_id"], m["group_ids"][0])].add(m["symbol"])
    return own, peer


def _group_state(symbol: str, d: int, dates: list[str], sessions: list[dict],
                 views: dict[str, SymbolData], own: dict, peers: dict) -> np.ndarray:
    """At-date memberships and peer prices; same issuer appears at most once."""
    date = dates[d]
    iso = pd.Timestamp(date).isocalendar()
    week = f"{iso.year}-W{iso.week:02d}"
    group = np.zeros((6, 10), np.float32)
    group[5, 0] = 1
    qqq = views.get("QQQ")
    if qqq is not None and np.isfinite(qqq.cutoff_return[d]):
        group[5, 3] = qqq.cutoff_return[d]
        group[5, 4] = qqq.cutoff_rvol[d] if np.isfinite(qqq.cutoff_rvol[d]) else 0
        group[5, 9] = 1
    for k, strategy in enumerate(STRATEGIES):
        record = own.get((week, strategy, symbol))
        if record is None:
            continue
        gid, facts, version = record
        decision = pd.Timestamp(sessions[d]["open_at"]) + pd.Timedelta(hours=2, seconds=30)
        if not (pd.Timestamp(version["effective_from"]) <= decision < pd.Timestamp(version["effective_to"])):
            continue
        if pd.Timestamp(version["feature_cutoff_at"]) > decision:
            continue
        if facts.get("history_end", "0000") >= date:
            continue
        # GOOG/GOOGL share the issuer; duplicate labels cannot inflate evidence.
        unique = {}
        for p in sorted(peers[(week, gid)]):
            if _issuer(p) != _issuer(symbol) and p in views:
                unique.setdefault(_issuer(p), p)
        others = list(unique.values())
        returns = [views[p].cutoff_return[d] for p in others if np.isfinite(views[p].cutoff_return[d])]
        rvol = [views[p].cutoff_rvol[d] for p in others if np.isfinite(views[p].cutoff_rvol[d])]
        if not returns:
            continue
        category = gid.split(":", 1)[1]
        cats = {"up":0, "range":1, "down":2, "low":0, "medium":1, "high":2}
        if category in cats:
            group[k, cats[category]] = 1
        group[k, 3] = np.median(returns)
        group[k, 4] = np.percentile(returns, 75) - np.percentile(returns, 25)
        group[k, 5] = np.mean(np.array(returns) > 0)
        group[k, 6] = np.median(rvol) if rvol else 0
        group[k, 7] = np.log1p(len(returns)) / 5
        group[k, 8] = len(returns) / len(others)
        group[k, 9] = 1
    return group


def _outcomes(view: SymbolData, sessions: list[dict], day_ix: int, config: dict,
              split_days: set[str]) -> tuple[np.ndarray, np.ndarray, str, str, np.ndarray] | None:
    if not np.isfinite(view.raw5[day_ix, 91, 0]):  # 11:35 entry bar
        return None
    raw = []
    available = []
    bounds = []
    for d in range(day_ix, len(sessions)):
        n = int(sessions[d]["duration_minutes"]) // 5
        a = 91 if d == day_ix else 66
        if a >= 66+n:
            return None
        raw.append(view.raw5[d, a:66+n])
        available.append(view.effective_available[d, a:66+n])
        bounds.extend([d] * (66+n-a))
        if sum(len(x) for x in raw) >= 390:
            break
    path = np.concatenate(raw) if raw else np.empty((0, 6))
    path_available = np.concatenate(available)
    if len(path) < 390 or not np.isfinite(path[:390, :4]).all():
        return None
    if any(sessions[d]["session_date"] in split_days for d in set(bounds[:390])):
        return None
    entry = float(view.raw5[day_ix, 91, 0])
    hits = np.zeros((3, 3), np.float32)
    terminal = np.zeros(3, np.float32)
    for i, horizon in enumerate((78, 234, 390)):
        window = path[:horizon]
        hits[i] = (np.max(window[:, 1]) >= entry * (1 + np.array(config["thresholds"]))).astype(np.float32)
        terminal[i] = window[-1, 3] / entry - 1
    # Last source bar ends at calendar regular slot after 390 accumulated 5m bars.
    last_d = bounds[389]
    last_offset = 389 - next(i for i, d in enumerate(bounds) if d == last_d)
    last_a = 91 if last_d == day_ix else 66
    end = pd.Timestamp(sessions[last_d]["open_at"]) + pd.Timedelta(minutes=(last_a - 66 + last_offset + 1)*5)
    observed_max = pd.Timestamp(np.max(path_available[:390])).tz_localize("UTC")
    avail = max(end, observed_max)
    return hits, terminal, end.isoformat(), avail.isoformat(), np.array([entry], np.float32)


def _split_days(symbol: str) -> set[str]:
    path = ROOT / "market_data/corporate_actions" / f"{symbol}.parquet"
    if not path.exists():
        return set()
    f = pd.read_parquet(path)
    ratio = pd.to_numeric(f.get("split_ratio", pd.Series(np.nan, index=f.index)), errors="coerce")
    base = pd.to_numeric(f.get("split_base", pd.Series(np.nan, index=f.index)), errors="coerce")
    return set(f.loc[((ratio.notna()) & (ratio != 0) & (ratio != 1)) |
                     ((base.notna()) & (ratio.notna()) & (base != ratio)), "ex_div_date"].astype(str))


def inputs_available(view: SymbolData, session: dict, d: int, latency_seconds: int) -> bool:
    """The entire historical source window and today's prefix must exist by decision."""
    decision = (pd.Timestamp(session["open_at"]) +
                pd.Timedelta(hours=2, seconds=latency_seconds)).tz_localize(None)
    past = view.effective_available[max(0, d-146):d]
    today = view.effective_available[d, :90]
    return not (np.any(past > np.datetime64(decision)) or
                np.any(today > np.datetime64(decision)))


def build(config: dict, output: Path, *, limit_symbols: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sessions = _calendar(config)
    dates = [s["session_date"] for s in sessions]
    universe = json.loads((ROOT / config["universe"]).read_text())
    symbols = sorted(m["symbol"] for m in universe["members"] if m["role"] == "candidate")
    if limit_symbols is not None:
        symbols = symbols[:limit_symbols]
    views = {}
    source_hashes = {}
    for symbol in symbols + ["QQQ"]:
        v = load_symbol(symbol, sessions, config)
        if v is not None:
            views[symbol] = v
            source_hashes[v.source] = digest(ROOT / v.source)
    own, peers = _group_map(config, dates, symbols)
    all_x5, all_x60, all_xday, all_groups, all_group_seq, all_y = [], [], [], [], [], []
    rows, counts = [], Counter()
    for symbol in symbols:
        view = views.get(symbol)
        if view is None:
            counts["missing_source_symbol"] += 1
            continue
        split_days = _split_days(symbol)
        for d, date in enumerate(dates):
            if not config["sample_start"] <= date <= config["sample_end"]:
                continue
            counts["candidate_symbol_dates"] += 1
            if pd.Timestamp(sessions[d]["close_at"]) <= pd.Timestamp(sessions[d]["open_at"]) + pd.Timedelta(hours=2):
                counts["no_1130_decision"] += 1
                continue
            if not np.isfinite(view.cutoff_return[d]):
                counts["missing_or_late_prefix"] += 1
                continue
            # Reject late historical inputs because the precomputed rolling
            # denominators and three scales would otherwise include them.
            if not inputs_available(view, sessions[d], d, config["decision_latency_seconds"]):
                counts["late_input_history_or_current"] += 1
                continue
            if date in split_days or any(x in split_days for x in dates[max(0, d-126):d]):
                counts["split_in_lookback"] += 1
                continue
            outcome = _outcomes(view, sessions, d, config, split_days)
            if outcome is None:
                counts["unmature_or_incomplete_5day_label"] += 1
                continue
            iso = pd.Timestamp(date).isocalendar()
            week = f"{iso.year}-W{iso.week:02d}"
            group = _group_state(symbol, d, dates, sessions, views, own, peers)
            # Six ordered, once-per-week states (oldest -> current). Last
            # session of an earlier week is known before today's decision.
            representative = {week: d}
            for older in range(d-1, -1, -1):
                x = pd.Timestamp(dates[older]).isocalendar()
                wk = f"{x.year}-W{x.week:02d}"
                representative.setdefault(wk, older)
                if len(representative) >= 6:
                    break
            group_seq = np.zeros((6, 6, 10), np.float32)
            for j, older in enumerate(sorted(representative.values())[-6:]):
                group_seq[6-len(representative)+j] = _group_state(symbol, older, dates, sessions, views, own, peers)
            x5 = np.zeros((8, SLOTS, len(FEATURE_COLUMNS)), np.float32)
            lo = max(0, d-7)
            x5[7-(d-lo):7] = view.seq5[lo:d]
            x5[7, :90] = view.seq5[d, :90]
            x60 = np.zeros((31, len(HOUR_BINS), len(FEATURE_COLUMNS)), np.float32)
            lo = max(0, d-30)
            x60[30-(d-lo):30] = view.seq60[lo:d]
            # Today's completed hour bars only, including the 10:30/11:30 regular bars.
            for j, (_, b, _) in enumerate(HOUR_BINS):
                if b <= 90:
                    x60[30, j] = view.seq60[d, j]
            xday = np.zeros((126, len(FEATURE_COLUMNS)), np.float32)
            lo = max(0, d-126)
            xday[126-(d-lo):] = view.seqday[lo:d]
            # Incomplete day is explicitly missing, never a partial final day candle.
            all_x5.append(x5); all_x60.append(x60); all_xday.append(xday)
            all_groups.append(group); all_group_seq.append(group_seq); all_y.append(outcome[0])
            rows.append({"sample_id": f"{symbol}:{date}:11:30:v6", "symbol": symbol,
                         "session_date": date, "cutoff_at": (pd.Timestamp(sessions[d]["open_at"]) + pd.Timedelta(hours=2)).isoformat(),
                         "decision_at": (pd.Timestamp(sessions[d]["open_at"]) + pd.Timedelta(hours=2, seconds=30)).isoformat(),
                         "entry_at": (pd.Timestamp(sessions[d]["open_at"]) + pd.Timedelta(hours=2, minutes=5)).isoformat(),
                         "entry_price": float(outcome[4][0]), "label_end_at": outcome[2],
                         "label_available_at": outcome[3], "terminal_1d": float(outcome[1][0]),
                         "terminal_3d": float(outcome[1][1]), "terminal_5d": float(outcome[1][2]),
                         "full_126_prior_days": int(np.sum(view.daily_full[max(0, d-126):d]) == 126),
                         "available_daily_days": int(np.sum(view.daily_full[max(0, d-126):d])),
                         "group_valid_count": int(np.sum(group[:5, 9])), "group_week": week})
            counts["complete_rows"] += 1
    if not rows:
        raise ValueError(f"no complete rows: {counts}")
    table = pd.DataFrame(rows)
    table.to_parquet(output / "rows.parquet", index=False)
    np.savez_compressed(output / "features.npz", x5=np.stack(all_x5), x60=np.stack(all_x60),
                        xday=np.stack(all_xday), group=np.stack(all_groups),
                        group_seq=np.stack(all_group_seq), y=np.stack(all_y))
    source_hashes[config["calendar"]] = digest(ROOT / config["calendar"])
    source_hashes[config["universe"]] = digest(ROOT / config["universe"])
    for name in ("versions.json", "memberships.json", "groups.json"):
        path = ROOT / config["group_run"] / name
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
    for symbol in views:
        action = ROOT / "market_data/corporate_actions" / f"{symbol}.parquet"
        if action.exists():
            source_hashes[str(action.relative_to(ROOT))] = digest(action)
    manifest = {"protocol": config["protocol"], "source_hashes": source_hashes,
                "builder_sha256": digest(Path(__file__)), "availability_quality": "assumed_archived_end_plus_1s_no_received_at",
                "universe_quality": "current_snapshot_retrospective",
                "group_quality": "causal_weekly_reconstruction_on_current_universe_not_original_PIT",
                "industry_branch": "excluded_current_snapshot_retrospective",
                "counts": dict(counts), "rows": len(rows), "feature_columns": FEATURE_COLUMNS,
                "groups": STRATEGIES, "input_shapes": {"x5": list(all_x5[0].shape),
                "x60": list(all_x60[0].shape), "xday": list(all_xday[0].shape),
                "group_seq": list(all_group_seq[0].shape)}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
