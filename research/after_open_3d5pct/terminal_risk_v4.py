"""Full-window touch and terminal-valuation research contract (no interim stop)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


UTC = timezone.utc
REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def net_return(gross: float | np.ndarray, buy_cost: float, sell_cost: float):
    """Liquidation *valuation* after a symmetric execution-cost scenario."""
    return (1 + np.asarray(gross)) * (1 - sell_cost) / (1 + buy_cost) - 1


def full_sample_es_loss(returns: np.ndarray, tail_fraction: float = 0.05) -> float:
    """Worst-tail mean loss using every evaluated opportunity and fractional boundary."""
    x = np.asarray(returns, dtype=float)
    if not len(x) or not np.isfinite(x).all() or not 0 < tail_fraction <= 1:
        raise ValueError("ES needs finite full-sample returns and a valid tail fraction")
    x = np.sort(x)
    mass = len(x) * tail_fraction
    whole = int(math.floor(mass))
    weighted = float(x[:whole].sum())
    if whole < len(x):
        weighted += (mass - whole) * float(x[whole])
    return -weighted / mass


def evaluate_complete_path(path: pd.DataFrame, *, entry_at: pd.Timestamp,
                           entry_price: float, label_end_at: pd.Timestamp,
                           expected_starts: pd.DatetimeIndex, as_of: pd.Timestamp,
                           target_return: float = 0.05,
                           buy_cost: float = 0.0006,
                           sell_cost: float = 0.0006) -> dict:
    """Score only a unique, valid, available 234-bar RTH path in exact order.

    The input may contain later rows/versions. It is filtered by event and
    availability time before revisions are chosen, so future data cannot
    change a fixed historical label. Entry price comes from the existing proxy.
    """
    base = {"status": "insufficient_data", "reason": None, "target": None,
            "terminal_close": None, "gross_evaluation_return": None,
            "net_evaluation_return": None, "first_hit_at_upper": None}
    if len(expected_starts) != 234 or expected_starts[0] != entry_at or expected_starts[-1] + pd.Timedelta(minutes=5) != label_end_at:
        return {**base, "reason": "invalid_calendar_or_window_boundary"}
    if as_of < label_end_at:
        return {**base, "status": "pending", "reason": "full_window_not_elapsed"}
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {**base, "reason": "invalid_entry_price"}
    f = path.copy()
    for col in ("start_at", "end_at", "available_at"):
        f[col] = pd.to_datetime(f[col], utc=True)
    if "received_at" in f:
        f["received_at"] = pd.to_datetime(f["received_at"], utc=True)
    f = f[(f.start_at >= entry_at) & (f.end_at <= label_end_at) &
          (f.available_at <= as_of)]
    if "received_at" in f:
        f = f[f.received_at.isna() | (f.received_at <= as_of)]
    if f.empty:
        return {**base, "reason": "window_not_available"}
    if "price_basis" not in f or (f.price_basis != "NONE").any():
        return {**base, "reason": "missing_or_mixed_price_basis"}
    version_cols = ["start_at", "received_at", "available_at"] if "received_at" in f else ["start_at", "available_at"]
    repeated = f.duplicated(version_cols, keep=False)
    if repeated.any():
        for _, versions in f.loc[repeated].groupby(version_cols, dropna=False):
            if len(versions[["end_at", "open", "high", "low", "close", "volume", "price_basis"]].drop_duplicates()) > 1:
                return {**base, "reason": "conflicting_duplicate_version"}
    sort = ["start_at", "received_at", "available_at"] if "received_at" in f else ["start_at", "available_at"]
    f = f.sort_values(sort).drop_duplicates("start_at", keep="last")
    f = f[f.session_type == "regular"].set_index("start_at").reindex(expected_starts)
    if len(f) != 234 or f["open"].isna().any() or f["end_at"].isna().any():
        return {**base, "reason": "missing_regular_bar"}
    values = f[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        return {**base, "reason": "invalid_ohlcv"}
    if (values[:, 1] < values[:, [0, 2, 3]].max(axis=1)).any() or (values[:, 2] > values[:, [0, 1, 3]].min(axis=1)).any():
        return {**base, "reason": "invalid_ohlc_order"}
    if not (pd.to_datetime(f.end_at, utc=True).to_numpy() == (expected_starts + pd.Timedelta(minutes=5)).to_numpy()).all():
        return {**base, "reason": "bar_duration_or_alignment_mismatch"}
    if pd.Timestamp(f.available_at.max()) > as_of:
        return {**base, "reason": "window_not_available"}
    hit_positions = np.flatnonzero(values[:, 1] >= entry_price * (1 + target_return))
    hit = bool(len(hit_positions))
    # No interim drawdown, later +20%, or post-touch move enters valuation.
    gross = target_return if hit else values[-1, 3] / entry_price - 1
    return {"status": "mature", "reason": None, "target": int(hit),
            "terminal_close": float(values[-1, 3]),
            "gross_evaluation_return": float(gross),
            "net_evaluation_return": float(net_return(gross, buy_cost, sell_cost)),
            "first_hit_at_upper": (expected_starts[hit_positions[0]] + pd.Timedelta(minutes=5)).isoformat() if hit else None,
            "label_available_at": pd.Timestamp(f.available_at.max()).isoformat(),
            "terminal_price_kind": "final_regular_5m_close_mark_to_liquidate_proxy",
            "touch_exit_kind": "fixed_plus_5pct_high_touch_proxy" if hit else None}


def derive_terminal_outcomes(source_run: Path, config: dict,
                             *, as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Read and verify the frozen v3 files; never modify their contents."""
    as_of = pd.Timestamp(as_of if as_of is not None else datetime.now(UTC))
    old = json.loads((source_run / "manifest.json").read_text())
    if old["feature_schema"] != config["feature_schema"]:
        raise ValueError("frozen feature schema mismatch")
    for filename in ("features.parquet", "outcomes.parquet", "sequences.npz"):
        if sha256(source_run / filename) != old["result_artifacts_sha256"][filename]:
            raise ValueError(f"frozen v3 artifact hash mismatch: {filename}")
    features = pd.read_parquet(source_run / "features.parquet")
    prior = pd.read_parquet(source_run / "outcomes.parquet")
    rows = features[["sample_id", "symbol", "session_date", "cutoff_et"]].merge(prior, on="sample_id", validate="one_to_one")
    calendar = json.loads((REPO / config["calendar"]).read_text())
    starts = pd.DatetimeIndex(np.concatenate([
        pd.date_range(pd.Timestamp(s["open_at"]), pd.Timestamp(s["close_at"]), freq="5min", inclusive="left").to_numpy()
        for s in calendar["sessions"]]), tz="UTC")
    out = []
    source_hashes = {}
    for symbol, group in rows.groupby("symbol", sort=True):
        paths = sorted((REPO / config["source_dir"] / symbol).glob("*.parquet"))
        frames = []
        for path in paths:
            rel = str(path.relative_to(REPO))
            expected = old["sources_sha256"].get(rel)
            if expected is None or sha256(path) != expected:
                raise ValueError(f"v3 frozen source hash mismatch: {rel}")
            source_hashes[rel] = expected
            frames.append(pd.read_parquet(path))
        if not frames:
            raise ValueError(f"frozen source absent: {symbol}")
        source = pd.concat(frames, ignore_index=True)
        source["start_at"] = pd.to_datetime(source.start_at, utc=True)
        source["end_at"] = pd.to_datetime(source.end_at, utc=True)
        source["available_at"] = pd.to_datetime(source.available_at, utc=True)
        regular = source[source.session_type == "regular"].sort_values("start_at")
        if regular.start_at.duplicated().any():
            raise ValueError(f"frozen historical regular source has duplicate starts: {symbol}")
        if regular.empty or regular.price_basis.isna().any() or (regular.price_basis != "NONE").any():
            raise ValueError(f"frozen historical price basis invalid: {symbol}")
        aligned = regular.set_index("start_at").reindex(starts)
        prices = aligned[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        end_ns = pd.to_datetime(aligned.end_at, utc=True).astype("datetime64[ns, UTC]").array.asi8
        available_ns = pd.to_datetime(aligned.available_at, utc=True).astype("datetime64[ns, UTC]").array.asi8
        expected_end_ns = (starts + pd.Timedelta(minutes=5)).astype("datetime64[ns, UTC]").asi8
        if len(prices) != len(starts):
            raise AssertionError("calendar source alignment failed")
        for row in group.itertuples():
            entry_at = pd.Timestamp(row.entry_at)
            label_end = pd.Timestamp(row.label_end_at)
            i = starts.searchsorted(entry_at)
            j = i + config["horizon_regular_bars"]
            base = {"status": "insufficient_data", "reason": None, "target": None,
                    "terminal_close": None, "gross_evaluation_return": None,
                    "net_evaluation_return": None, "first_hit_at_upper": None,
                    "label_available_at": None, "terminal_price_kind": None,
                    "touch_exit_kind": None}
            if j > len(starts) or starts[i] != entry_at or starts[j - 1] + pd.Timedelta(minutes=5) != label_end:
                result = {**base, "reason": "invalid_calendar_or_window_boundary"}
            elif as_of < label_end:
                result = {**base, "status": "pending", "reason": "full_window_not_elapsed"}
            else:
                segment = prices[i:j]
                valid = (len(segment) == config["horizon_regular_bars"] and
                         np.isfinite(segment).all() and
                         (segment[:, :4] > 0).all() and (segment[:, 4] >= 0).all() and
                         (segment[:, 1] >= segment[:, [0, 2, 3]].max(axis=1)).all() and
                         (segment[:, 2] <= segment[:, [0, 1, 3]].min(axis=1)).all() and
                         np.array_equal(end_ns[i:j], expected_end_ns[i:j]) and
                         (available_ns[i:j] > 0).all() and
                         (available_ns[i:j] <= as_of.value).all())
                if not valid:
                    result = {**base, "reason": "missing_invalid_or_unavailable_regular_bar"}
                else:
                    hit_positions = np.flatnonzero(segment[:, 1] >= float(row.entry_price) * (1 + config["touch_gross_return"]))
                    hit = len(hit_positions) > 0
                    gross = config["touch_gross_return"] if hit else segment[-1, 3] / float(row.entry_price) - 1
                    ready = pd.Timestamp(available_ns[i:j].max(), tz="UTC")
                    result = {"status": "mature", "reason": None, "target": int(hit),
                              "terminal_close": float(segment[-1, 3]),
                              "gross_evaluation_return": float(gross),
                              "net_evaluation_return": float(net_return(gross, config["buy_cost_rate"], config["sell_cost_rate"])),
                              "first_hit_at_upper": (starts[i + hit_positions[0]] + pd.Timedelta(minutes=5)).isoformat() if hit else None,
                              "label_available_at": ready.isoformat(),
                              "terminal_price_kind": "final_regular_5m_close_mark_to_liquidate_proxy",
                              "touch_exit_kind": "fixed_plus_5pct_high_touch_proxy" if hit else None}
                    if hit != bool(row.target) or not np.isclose(segment[0, 0], row.entry_price, rtol=0, atol=1e-6):
                        raise ValueError(f"rederived touch/entry disagrees with frozen v3 label: {row.sample_id}")
                    if ready != pd.Timestamp(row.label_available_at):
                        raise ValueError(f"rederived availability disagrees with frozen v3 label: {row.sample_id}")
            out.append({"sample_id": row.sample_id, "symbol": symbol, "session_date": row.session_date,
                        "cutoff_et": row.cutoff_et, "entry_at": row.entry_at,
                        "entry_price": row.entry_price, "label_end_at": row.label_end_at,
                        **result})
    result = pd.DataFrame(out)
    audit = {"source_run": str(source_run), "source_manifest_sha256": sha256(source_run / "manifest.json"),
             "source_hashes_verified": source_hashes, "rows": len(result),
             "statuses": result.status.value_counts().to_dict(),
             "rederived_target_mismatches": 0}
    return result, audit


def mixed_expected_net(p_touch: np.ndarray, failure_mean_gross: np.ndarray,
                       *, touch_gross: float, buy_cost: float, sell_cost: float) -> np.ndarray:
    p = np.asarray(p_touch, dtype=float)
    failure_net = net_return(np.asarray(failure_mean_gross, dtype=float), buy_cost, sell_cost)
    touch_net = net_return(touch_gross, buy_cost, sell_cost)
    return p * touch_net + (1 - p) * failure_net


def pinball(y: np.ndarray, prediction: np.ndarray, q: float) -> float:
    error = np.asarray(y) - np.asarray(prediction)
    return float(np.mean(np.maximum(q * error, (q - 1) * error)))
