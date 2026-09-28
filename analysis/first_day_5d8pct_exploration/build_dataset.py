"""Offline first-day feature panel, using frozen v5 group and target labels."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.group_expectation_matrix.build import calendar_grid, corporate_dates, prepare_bars

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "research/group_expectation_matrix/outputs/20260925_v5"
CHECKPOINTS = ["11:30", "12:05", "12:35", "13:35", "15:35", "16:00"]
ET = "America/New_York"

# key: user-facing definition, threshold units/family
FEATURES = {
    "session_return": ("当日开盘至观察时涨幅", "pct", "momentum"),
    "open_gap": ("相对前收盘的开盘跳空", "pct", "gap"),
    "open30_return": ("开盘前30分钟涨幅", "pct", "opening"),
    "open60_return": ("开盘前60分钟涨幅", "pct", "opening"),
    "open30_range": ("开盘前30分钟振幅", "pct", "range"),
    "session_range": ("当日截至观察时振幅", "pct", "range"),
    "close_location": ("现价在当日高低区间的位置", "ratio", "position"),
    "drawdown_high": ("现价距离当日最高价的回落", "pct", "position"),
    "path_efficiency": ("日内净涨幅/逐5分钟绝对变动之和", "ratio", "persistence"),
    "up_bar_fraction": ("当日上涨5分钟K线占比", "ratio", "persistence"),
    "last30_return": ("最后30分钟涨幅", "pct", "late_momentum"),
    "acceleration": ("末30分钟涨幅减去此前30分钟涨幅", "pct", "late_momentum"),
    "realized_vol": ("当日5分钟实现波动", "pct", "volatility"),
    "vwap_distance": ("现价相对当日VWAP涨幅", "pct", "vwap"),
    "above_vwap_fraction": ("收盘价高于当时累计VWAP的5分钟占比", "ratio", "vwap"),
    "relative_volume": ("累计量/过去20日同时间成交量中位数", "multiple", "volume"),
    "range_expansion": ("累计振幅/过去20日同时间振幅中位数", "multiple", "range"),
    "late_volume_ratio": ("末30分钟量/此前30分钟量", "multiple", "volume"),
    "prior5_return": ("前5交易日收盘收益", "pct", "history"),
    "prior20_return": ("前20交易日收盘收益", "pct", "history"),
    "prior20_vol": ("过去20日收盘收益日波动率", "pct", "volatility"),
    "progress": ("现价相对11:35起始价涨幅", "pct", "progress"),
    "max_progress": ("11:35以来曾达到的最大涨幅", "pct", "progress"),
    "post_entry_dip": ("11:35以来最大向下偏离起始价", "pct", "dip"),
    "relative_qqq": ("当日涨幅减QQQ当日涨幅", "pct", "relative"),
    "qqq_return": ("QQQ当日涨幅", "pct", "market"),
    "market_breadth": ("101只股票中当时高于当日开盘价的比例", "ratio", "market"),
}


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(prefix, history, past_closes, entry, post, allow_history=True):
    """Only accepts past/current data; no future path or target passed here."""
    o, h, l, c, v, t = prefix.T
    n = len(prefix)
    width = h.max() - l.min()
    changes = np.diff(np.r_[o[0], c])
    abs_path = np.abs(changes).sum()
    result = {k: np.nan for k in FEATURES}
    result.update(
        session_return=c[-1] / o[0] - 1,
        open30_return=c[5] / o[0] - 1,
        open60_return=c[11] / o[0] - 1,
        open30_range=(h[:6].max() - l[:6].min()) / o[0],
        session_range=width / o[0],
        close_location=(c[-1] - l.min()) / width if width > 0 else .5,
        drawdown_high=1 - c[-1] / h.max(),
        path_efficiency=(c[-1] - o[0]) / abs_path if abs_path > 0 else 0.,
        up_bar_fraction=float((c > o).mean()),
        last30_return=c[-1] / o[-6] - 1,
        acceleration=(c[-1] / o[-6] - 1) - (c[-7] / o[-12] - 1),
        realized_vol=np.std(np.diff(np.log(np.r_[o[0], c])), ddof=1) * np.sqrt(n),
        late_volume_ratio=v[-6:].sum() / v[-12:-6].sum() if v[-12:-6].sum() > 0 else np.nan,
    )
    good_volume = np.isfinite(v).all() and (v >= 0).all() and v.sum() > 0
    if good_volume:
        positive = v > 0
        each_vwap = np.divide(t, v, out=np.full(n, np.nan), where=positive)
        good_vwap = (np.isfinite(t).all() and (t >= 0).all()
                     and ((each_vwap[positive] >= l[positive] - .01)
                          & (each_vwap[positive] <= h[positive] + .01)).all()
                     and (t[~positive] == 0).all())
        if good_vwap:
            cumulative = np.divide(np.cumsum(t), np.cumsum(v), out=np.full(n, np.nan), where=np.cumsum(v) > 0)
            result["vwap_distance"] = c[-1] / cumulative[-1] - 1
            result["above_vwap_fraction"] = float(np.mean(c[np.isfinite(cumulative)] > cumulative[np.isfinite(cumulative)]))
    if allow_history:
        if len(past_closes) and np.isfinite(past_closes[-1]):
            result["open_gap"] = o[0] / past_closes[-1] - 1
        for days in (5, 20):
            recent = np.array(past_closes[-days - 1:])
            if len(recent) == days + 1 and np.isfinite(recent).all():
                result[f"prior{days}_return"] = recent[-1] / recent[0] - 1
                if days == 20:
                    result["prior20_vol"] = np.std(np.diff(np.log(recent)), ddof=1)
        prior_prefixes = [x[:n] for x in history[-20:] if x is not None and len(x) >= n]
        if len(prior_prefixes) >= 10:
            volumes = [x[:, 4].sum() for x in prior_prefixes if np.isfinite(x[:, 4]).all() and (x[:, 4] >= 0).all()]
            ranges = [(x[:, 1].max() - x[:, 2].min()) / x[0, 0] for x in prior_prefixes]
            if len(volumes) >= 10 and np.median(volumes) > 0 and good_volume:
                result["relative_volume"] = v.sum() / np.median(volumes)
            if np.median(ranges) > 0:
                result["range_expansion"] = result["session_range"] / np.median(ranges)
    if len(post):
        result.update(progress=c[-1] / entry - 1,
                      max_progress=max(0., post[:, 1].max() / entry - 1),
                      post_entry_dip=min(0., post[:, 2].min() / entry - 1))
    return result


def main():
    cfg = json.loads((SOURCE / "config.json").read_text())
    calendar = json.loads((ROOT / cfg["calendar"]).read_text())
    grid = calendar_grid(calendar["sessions"])
    starts = pd.DatetimeIndex(grid.start).as_unit("ns").asi8
    ends = pd.DatetimeIndex(grid.end).as_unit("ns").asi8
    days = grid.groupby("date", sort=True).indices
    all_days = list(days)
    labels = pd.read_parquet(SOURCE / "outcomes.parquet")
    labels = labels[labels.expectation_id.eq("d5_r5") & labels.status.eq("mature")]
    assert not labels.sample_id.duplicated().any()
    instruments = json.loads((SOURCE / "instruments.json").read_text())
    members = json.loads((SOURCE / "memberships.json").read_text())
    member_map = {}
    for m in members:
        if m["status"] == "classified":
            member_map.setdefault((m["symbol"], m["week_id"]), []).extend(m["group_ids"])
    group_versions = {(v["strategy_id"], v["week_id"]): v for v in json.loads((SOURCE / "versions.json").read_text())}
    for row in labels.itertuples():
        for gid in member_map.get((row.symbol, row.week_id), []):
            strategy = "watchlist_etf" if gid == "watchlist:ETF" else gid.split(":")[0]
            assert pd.Timestamp(group_versions[strategy, row.week_id]["feature_cutoff_at"]) < pd.Timestamp(row.entry_at)

    records, context_records, audits, sources = [], [], [], []
    asof = pd.Timestamp(cfg["evaluation_as_of"])
    for instrument in instruments:
        symbol = instrument["symbol"]
        path = ROOT / cfg["bars_dir"] / symbol / "2026.parquet"
        action_path = ROOT / cfg["actions_dir"] / (symbol + ".parquet")
        if not path.exists() or not action_path.exists():
            audits.append({"symbol": symbol, "status": "no_data", "rows": 0})
            continue
        source_labels = labels[labels.symbol.eq(symbol)].set_index("date")
        raw = pd.read_parquet(path)
        actions = corporate_dates(pd.read_parquet(action_path))
        aligned, _, valid, _, duplicates, _ = prepare_bars(raw, grid, asof)
        arr = aligned[["open", "high", "low", "close", "volume", "turnover"]].to_numpy(float)
        available = pd.DatetimeIndex(aligned.available).as_unit("ns").asi8
        history, closes, rows, checks = [], [], 0, 0
        for day_index, day in enumerate(all_days):
            ids = days[day]
            daily = arr[ids]
            complete = bool(valid[ids].all())
            # Market context must not depend on a security's future label quality.
            if "2026-01-02" <= day <= "2026-09-24":
                for checkpoint in CHECKPOINTS:
                    cutoff = pd.Timestamp(f"{day} {checkpoint}", tz=ET).tz_convert("UTC").value
                    n_context = int(np.sum(ends[ids] <= cutoff))
                    if (n_context >= 12 and valid[ids[:n_context]].all()
                            and (available[ids[:n_context]] <= cutoff + 1_000_000_000).all()
                            and (checkpoint == "16:00" or cutoff <= ends[ids[-1]])):
                        context_records.append({"symbol": symbol, "instrument_type": instrument["instrument_type"],
                                                "date": day, "checkpoint": checkpoint,
                                                "session_return": daily[n_context - 1, 3] / daily[0, 0] - 1})
            if day in source_labels.index:
                label = source_labels.loc[day]
                pos = int(np.searchsorted(starts, pd.Timestamp(label.entry_at).value))
                entry = arr[pos, 0]
                full = arr[pos:pos + 390]
                assert len(full) == 390 and valid[pos:pos + 390].all()
                assert np.isclose(entry, label.entry_price, rtol=0, atol=1e-10)
                hit = bool((full[:, 1] >= entry * 1.08).any())
                assert hit == bool(label.mfe >= .08)
                checks += 1
                allow_history = not any(all_days[max(0, day_index - 21)] < a <= day for a in actions)
                for checkpoint in CHECKPOINTS:
                    cutoff = pd.Timestamp(f"{day} {checkpoint}", tz=ET).tz_convert("UTC")
                    cutoff_ns = cutoff.value
                    n = int(np.sum(ends[ids] <= cutoff_ns))
                    if n < 12 or not valid[ids[:n]].all() or (available[ids[:n]] > cutoff_ns + 1_000_000_000).any():
                        continue
                    if checkpoint != "16:00" and cutoff_ns > ends[ids[-1]]:
                        continue
                    used = ids[:n]
                    post_ids = used[used >= pos]
                    post = arr[post_ids]
                    already = bool(len(post) and (post[:, 1] >= entry * 1.08).any())
                    observation_pos = max(pos, int(used[-1]) + 1)
                    remaining = arr[observation_pos:pos + 390]
                    decision_price = daily[n - 1, 3]
                    first_hits = np.flatnonzero(remaining[:, 1] >= entry * 1.08)
                    feats = features(daily[:n], history, closes, entry, post, allow_history)
                    row = {"sample_id": label.sample_id, "symbol": symbol, "instrument_type": instrument["instrument_type"],
                           "date": day, "week_id": label.week_id, "checkpoint": checkpoint,
                           "feature_available_at": (cutoff + pd.Timedelta(seconds=1)).isoformat(),
                           "entry_at": label.entry_at, "entry_price": entry, "label_end_at": label.label_end_at,
                           "label_available_at": label.label_available_at, "target": int(hit),
                           "already_hit": already, "group_ids": member_map.get((symbol, label.week_id), []),
                           "remaining_upside": entry * 1.08 / decision_price - 1,
                           "future_mae_from_observation": min(0., remaining[:, 2].min() / decision_price - 1),
                           "terminal_from_observation": remaining[-1, 3] / decision_price - 1,
                           "minutes_to_hit_after_observation": int((first_hits[0] + 1) * 5) if len(first_hits) else np.nan,
                           **feats}
                    records.append(row)
                    rows += 1
            history.append(daily if complete else None)
            closes.append(daily[-1, 3] if complete else np.nan)
        audits.append({"symbol": symbol, "status": "ready", "rows": rows, "labels_checked": checks, "duplicates": duplicates})
        sources += [{"path": str(p.relative_to(ROOT)), "sha256": hash_file(p)} for p in (path, action_path)]
        if len(audits) % 20 == 0:
            print(f"Prepared {len(audits)}/{len(instruments)} securities", flush=True)
    panel = pd.DataFrame(records)
    context = pd.DataFrame(context_records)
    qqq = context[context.symbol.eq("QQQ")][["date", "checkpoint", "session_return"]].rename(columns={"session_return": "qqq_value"})
    panel = panel.merge(qqq, on=["date", "checkpoint"], how="left", validate="many_to_one")
    panel["qqq_return"] = panel.qqq_value
    panel["relative_qqq"] = panel.session_return - panel.qqq_value
    stock_rows = context[context.instrument_type.eq("stock")].copy()
    stock_rows["up"] = stock_rows.session_return.gt(0)
    breadth = stock_rows.groupby(["date", "checkpoint"]).agg(breadth=("up", "mean"), n=("symbol", "nunique"))
    breadth.loc[breadth.n < 90, "breadth"] = np.nan
    panel = panel.merge(breadth[["breadth"]], left_on=["date", "checkpoint"], right_index=True, how="left")
    panel["market_breadth"] = panel.breadth
    panel.drop(columns=["qqq_value", "breadth"], inplace=True)
    available = pd.to_datetime(panel.label_available_at, utc=True)
    panel["split"] = "purged"
    panel.loc[panel.date.le("2026-05-29") & available.lt(pd.Timestamp("2026-06-01 09:30", tz=ET)), "split"] = "discovery"
    panel.loc[panel.date.between("2026-06-01", "2026-07-17") & available.lt(pd.Timestamp("2026-07-20 09:30", tz=ET)), "split"] = "selection"
    panel.loc[panel.date.between("2026-07-20", "2026-09-17"), "split"] = "confirmation"
    assert not panel.duplicated(["sample_id", "checkpoint"]).any()
    assert (panel.loc[panel.already_hit, "target"] == 1).all()
    panel.to_parquet(HERE / "features.parquet", index=False, compression="zstd", compression_level=7)
    (HERE / "feature_definitions.json").write_text(json.dumps(FEATURES, ensure_ascii=False, indent=2))
    sources += [{"path": str((SOURCE / n).relative_to(ROOT)), "sha256": hash_file(SOURCE / n)}
                for n in ("outcomes.parquet", "instruments.json", "memberships.json", "versions.json", "groups.json", "strategies.json")]
    result = {"rows": len(panel), "unique_samples": panel.sample_id.nunique(), "securities": panel.symbol.nunique(),
              "split_rows": panel.groupby("split").size().to_dict(), "feature_nonmissing": panel[list(FEATURES)].notna().mean().to_dict(),
              "coverage": audits, "sources": sources, "source_files_verified_against_v5": None,
              "protocol_sha256": hash_file(HERE / "PROTOCOL.md"), "script_sha256": hash_file(Path(__file__))}
    original = {x["path"]: x["sha256"] for x in json.loads((SOURCE / "manifest.json").read_text())["source_files"]}
    mismatches = [x["path"] for x in sources if x["path"] in original and x["sha256"] != original[x["path"]]]
    result["source_files_verified_against_v5"] = {"mismatches": mismatches, "matched_count": sum(x["path"] in original for x in sources) - len(mismatches)}
    assert not mismatches, mismatches
    (HERE / "dataset_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k: result[k] for k in ("rows", "unique_samples", "securities", "split_rows", "source_files_verified_against_v5")}, indent=2))


if __name__ == "__main__":
    main()
