"""Causal, decision-time stock quality from complete historical label days.

The original label ledger is left intact. A label version is filtered by the
query's decision time *before* choosing revisions or aggregating a stock-day.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd


HOURS = ("10:30", "11:30", "12:30", "13:30", "14:30", "15:30")
FEATURE_COLUMNS = ("h_quality_rate_63", "h_quality_mature_days_63",
                   "h_quality_recent_rate_21", "h_quality_recent_days_21",
                   "h_quality_prior_rate_42", "h_quality_prior_days_42",
                   "h_quality_delta_recent_prior", "h_quality_rate_available_mask",
                   "h_quality_delta_available_mask")


def _ns(values: pd.Series, *, allow_null: bool = False) -> np.ndarray:
    dt = pd.to_datetime(values, utc=True, errors="raise")
    if dt.isna().any() and not allow_null:
        raise ValueError("null timing in quality label or query")
    return dt.astype("datetime64[ns, UTC]").astype("int64").to_numpy()


def _prepared(queries: pd.DataFrame, labels: pd.DataFrame,
              sessions: Sequence[str], config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required_q = {"sample_id", "symbol", "session_date", "decision_at"}
    required_l = {"symbol", "session_date", "cutoff_et", "target", "status",
                  "label_end_at", "label_available_at"}
    if not required_q.issubset(queries) or not required_l.issubset(labels):
        raise ValueError("quality input missing required columns")
    if queries.sample_id.duplicated().any():
        raise ValueError("duplicate quality query sample_id")
    if list(config["expected_distinct_cutoffs_et"]) != list(HOURS):
        raise ValueError("unsupported quality cutoff schema")
    session_ids = list(sessions)
    if len(set(session_ids)) != len(session_ids) or session_ids != sorted(session_ids):
        raise ValueError("official session dates are not unique and ordered")
    ordinal = {day: i for i, day in enumerate(session_ids)}
    q = queries[["sample_id", "symbol", "session_date", "decision_at"]].copy()
    q["_ordinal"] = q.session_date.map(ordinal)
    if q._ordinal.isna().any():
        raise ValueError("query date absent from official calendar")
    q["_decision_ns"] = _ns(q.decision_at)
    q["_row"] = np.arange(len(q))
    # An unrelated/unsupported hour is not one of this protocol's six daily
    # observations. In particular, a later appended 16:30 row cannot turn an
    # already complete six-hour day into a seven-row incomplete day.
    l = labels.loc[labels.cutoff_et.isin(HOURS)].copy()
    l["_ordinal"] = l.session_date.map(ordinal)
    if l._ordinal.isna().any():
        raise ValueError("label date absent from official calendar")
    l["_time_valid"] = l.label_end_at.notna() & l.label_available_at.notna()
    l["_end_ns"] = _ns(l.label_end_at, allow_null=True)
    l["_available_ns"] = _ns(l.label_available_at, allow_null=True)
    l["_known_ns"] = np.maximum(l._end_ns, l._available_ns)
    l.loc[~l._time_valid, "_known_ns"] = np.iinfo(np.int64).max
    receipt_col = "label_received_at" if "label_received_at" in l else (
        "received_at" if "received_at" in l else None)
    if receipt_col:
        receipt = pd.to_datetime(l[receipt_col], utc=True, errors="raise")
        l["_receipt_observed"] = receipt.notna().astype(np.int8)
        l["_receipt_ns"] = receipt.astype("datetime64[ns, UTC]").astype("int64")
        l.loc[receipt.notna(), "_known_ns"] = np.maximum(
            l.loc[receipt.notna(), "_known_ns"].to_numpy(np.int64),
            l.loc[receipt.notna(), "_receipt_ns"].to_numpy(np.int64))
    else:
        l["_receipt_observed"] = np.int8(0)
        l["_receipt_ns"] = np.int64(-1)
    return q, l, ordinal


def _daily_unique(labels: pd.DataFrame, n_sessions: int) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray]:
    """Fast path when the frozen ledger has at most one row per stock/date/hour."""
    rate = np.full(n_sessions, np.nan)
    known = np.full(n_sessions, np.iinfo(np.int64).max, dtype=np.int64)
    complete = np.zeros(n_sessions, dtype=bool)
    incomplete = np.zeros(n_sessions, dtype=bool)
    for ordinal, day in labels.groupby("_ordinal", sort=False):
        i = int(ordinal)
        if (len(day) == len(HOURS) and set(day.cutoff_et) == set(HOURS) and
                day.status.eq("mature").all() and day._time_valid.all() and
                day.target.isin([0, 1]).all()):
            rate[i] = float(day.target.mean())
            known[i] = int(day._known_ns.max())
            complete[i] = True
        else:
            incomplete[i] = True
    return rate, known, complete, incomplete


def _choose_visible_versions(labels: pd.DataFrame, decision_ns: int) -> pd.DataFrame:
    """Slow revision path: visibility precedes per-hour version selection."""
    visible = labels.loc[labels._known_ns < decision_ns].copy()
    if visible.empty:
        return visible
    visible["_version_ns"] = np.where(visible._receipt_observed.to_numpy(bool),
                                      visible._receipt_ns.to_numpy(np.int64),
                                      visible._available_ns.to_numpy(np.int64))
    visible.sort_values(["_ordinal", "cutoff_et", "_receipt_observed", "_version_ns"],
                        kind="stable", inplace=True)
    conflicts = visible.duplicated(["_ordinal", "cutoff_et", "_receipt_observed", "_version_ns"],
                                   keep=False)
    for _, group in visible.loc[conflicts].groupby(
            ["_ordinal", "cutoff_et", "_receipt_observed", "_version_ns"]):
        if len(group[["status", "target", "_end_ns", "_available_ns"]].drop_duplicates()) != 1:
            raise ValueError("conflicting same-time quality label versions")
    return visible.drop_duplicates(["_ordinal", "cutoff_et"], keep="last")


def _features_for_query(qord: int, decision_ns: int, rate: np.ndarray,
                        known: np.ndarray, complete: np.ndarray,
                        config: dict) -> dict:
    lookback = int(config["lookback_official_prior_sessions"])
    recent, prior = map(int, config["recent_prior_windows_sessions"])
    if recent + prior != lookback:
        raise ValueError("recent/prior windows do not sum to lookback")
    lo = max(0, qord - lookback)
    dates = np.arange(lo, qord)
    visible = complete[dates] & (known[dates] < decision_ns)

    def period(a: int, b: int) -> tuple[int, float, int]:
        chosen = visible & (dates >= a) & (dates < b)
        values = rate[dates[chosen]]
        hits = int(np.rint(values * len(HOURS)).sum())
        return len(values), hits / (len(HOURS) * len(values)) if len(values) else np.nan, hits

    count, all_rate, hits = period(qord - lookback, qord)
    recent_count, recent_rate, _ = period(qord - recent, qord)
    prior_count, prior_rate, _ = period(qord - lookback, qord - recent)
    rate_ready = count >= int(config["min_complete_mature_stock_dates"])
    recent_min, prior_min = map(int, config["recent_prior_min_complete_dates"])
    delta_ready = recent_count >= recent_min and prior_count >= prior_min
    # 0.60 is exactly 3/5. Compare integer hit/date counts so floating-point
    # summation cannot admit an exactly-60% stock by accident.
    threshold = float(config["strict_historical_rate_gt"])
    eligible = rate_ready and (5 * hits > 3 * len(HOURS) * count if threshold == .6
                               else all_rate > threshold)
    history_dates = dates[visible]
    history_hits = np.rint(rate[history_dates] * len(HOURS)).astype(np.int16)
    digest = hashlib.sha256()
    for day_ordinal, day_hits in zip(history_dates, history_hits):
        digest.update(f"{int(day_ordinal)}:{int(day_hits)}\n".encode())
    latest_known = (pd.Timestamp(int(known[history_dates].max()), unit="ns", tz="UTC").isoformat()
                    if len(history_dates) else None)
    return {"h_quality_rate_63": all_rate if rate_ready else np.nan,
            "h_quality_mature_days_63": count,
            "h_quality_recent_rate_21": recent_rate if recent_count >= recent_min else np.nan,
            "h_quality_recent_days_21": recent_count,
            "h_quality_prior_rate_42": prior_rate if prior_count >= prior_min else np.nan,
            "h_quality_prior_days_42": prior_count,
            "h_quality_delta_recent_prior": recent_rate - prior_rate if delta_ready else np.nan,
            "h_quality_rate_available_mask": int(rate_ready),
            "h_quality_delta_available_mask": int(delta_ready),
            "quality_eligible": bool(eligible),
            "quality_history_days_sha256": digest.hexdigest(),
            "quality_history_latest_known_at": latest_known,
            "quality_status": "quality_gt60" if eligible
                              else "below_or_equal_threshold" if rate_ready else "insufficient_history"}


def build_quality_features(queries: pd.DataFrame, labels: pd.DataFrame,
                           sessions: Sequence[str], config: dict) -> pd.DataFrame:
    """Return one traceable quality row per query, without using its own outcome.

    Frozen unique labels use preaggregated day arrays. Sources containing
    revisions use a query-specific visibility-first path so an appended late
    version of an old label cannot erase a previously complete history day.
    """
    q, l, _ = _prepared(queries, labels, sessions, config)
    n_sessions = len(sessions)
    output = [None] * len(q)
    for symbol, group in q.groupby("symbol", sort=False):
        stock = l.loc[l.symbol == symbol]
        duplicated = stock.duplicated(["_ordinal", "cutoff_et"]).any()
        if not duplicated:
            rate, known, complete, _ = _daily_unique(stock, n_sessions)
            for _, _, _, _, ordinal, decision_ns, row_number in group.itertuples(index=False, name=None):
                item = _features_for_query(int(ordinal), int(decision_ns),
                                           rate, known, complete, config)
                output[int(row_number)] = item
        else:
            for _, _, _, _, ordinal, decision_ns, row_number in group.itertuples(index=False, name=None):
                qord, decision_ns = int(ordinal), int(decision_ns)
                lo = qord - int(config["lookback_official_prior_sessions"])
                window = stock[(stock._ordinal >= lo) & (stock._ordinal < qord)]
                visible = _choose_visible_versions(window, decision_ns)
                rate, known, complete, _ = _daily_unique(visible, n_sessions)
                item = _features_for_query(qord, decision_ns, rate, known, complete, config)
                output[int(row_number)] = item
    result = pd.DataFrame(output)
    result.insert(0, "sample_id", q.sample_id.to_numpy())
    return result


def feature_lineage_hash(frame: pd.DataFrame, source_hashes: dict[str, str]) -> str:
    """Compact version marker for a materialized feature table and its sources."""
    digest = hashlib.sha256()
    for path, value in sorted(source_hashes.items()):
        digest.update(f"{path}:{value}\n".encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()
