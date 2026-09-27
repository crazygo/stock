"""Frozen inner/outer metrics for causal quality-pool model comparisons."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .terminal_risk_v4 import full_sample_es_loss


def binary_scores(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    if not len(y):
        return {"n": 0, "brier": None, "logloss": None, "touches": 0,
                "touch_rate": None, "prediction_mean": None}
    return {"n": int(len(y)), "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, p, labels=[0, 1])),
            "touches": int(y.sum()), "touch_rate": float(y.mean()),
            "prediction_mean": float(p.mean())}


def outcome_summary(frame: pd.DataFrame, tail: float) -> dict:
    if frame.empty:
        return {"n": 0, "stock_dates": 0, "ET_dates": 0, "symbols": 0,
                "touches": 0, "touch_rate": None, "mean_net": None,
                "es95_loss_full": None}
    shares = frame.symbol.value_counts(normalize=True)
    return {"n": int(len(frame)),
            "stock_dates": int(frame[["symbol", "session_date"]].drop_duplicates().shape[0]),
            "ET_dates": int(frame.session_date.nunique()),
            "symbols": int(frame.symbol.nunique()),
            "max_single_symbol_share": float(shares.max()),
            "touches": int(frame.target.sum()), "touch_rate": float(frame.target.mean()),
            "mean_net": float(frame.realized_net.mean()),
            "es95_loss_full": full_sample_es_loss(frame.realized_net.to_numpy(float), tail)}


def choose_quality_recommendations(frame: pd.DataFrame, scenario: dict) -> np.ndarray:
    """Only same-time quality rows compete; the 5% cap never requires fill."""
    chosen = np.zeros(len(frame), dtype=bool)
    if frame.empty:
        return chosen
    for _, group in frame.groupby(["session_date", "cutoff_et"], sort=False):
        q = group[group.quality_eligible]
        k = math.ceil(len(q) * scenario["opportunity_budget_fraction_max_each_quality_date_hour"])
        eligible = q[(q.p_touch >= scenario["candidate_calibrated_p_touch_min"]) &
                     (q.expected_net > scenario["predicted_expected_net_min"]) &
                     (q.failure_q10_net >= scenario["predicted_failure_q10_net_min"])]
        top = eligible.sort_values(["p_touch", "symbol"], ascending=[False, True]).head(k)
        chosen[top.index.to_numpy()] = True
    return chosen


def matched_history_rows(frame: pd.DataFrame, chosen: np.ndarray) -> pd.DataFrame:
    """Causally choose exactly the model's count in each date/hour quality pool."""
    matches = []
    for _, group in frame.groupby(["session_date", "cutoff_et"], sort=False):
        count = int(chosen[group.index.to_numpy()].sum())
        if count:
            q = group[group.quality_eligible]
            matches.append(q.sort_values(["h_quality_rate_63", "symbol"],
                                         ascending=[False, True]).head(count))
    return pd.concat(matches) if matches else frame.iloc[:0].copy()


def _block_ci(frame: pd.DataFrame, value_column: str, date_axis: list[str],
              width: int, reps: int, seed: int) -> dict:
    # The axis is the full official phase calendar, including sessions with no
    # recommendation or no quality row. Never collapse gaps between signals.
    days = list(date_axis)
    if len(days) < 2 * width:
        return {"status": "insufficient_independent_calendar_blocks", "ET_dates": len(days),
                "block_sessions": width, "lower": None, "upper": None}
    values = frame[value_column].to_numpy(float)
    if not set(frame.session_date).issubset(days):
        raise ValueError("block CI observations absent from full phase date axis")
    by_date = {day: np.flatnonzero(frame.session_date.to_numpy() == day) for day in days}
    starts = np.arange(len(days) - width + 1)
    rng = np.random.default_rng(seed)
    sampled = []
    empty_draws = 0
    for _ in range(reps):
        draw = []
        while len(draw) < len(days):
            j = int(rng.choice(starts))
            draw.extend(days[j:j + width])
        ids = np.concatenate([by_date[d] for d in draw[:len(days)]])
        if len(ids):
            sampled.append(float(values[ids].mean()))
        else:
            empty_draws += 1
    if empty_draws:
        return {"status": "bootstrap_empty_selected_draw", "ET_dates": len(days),
                "block_sessions": width, "empty_draws": empty_draws,
                "replicates": reps, "lower": None, "upper": None}
    lower, upper = np.quantile(sampled, [.025, .975])
    return {"status": "exposed_development_block_bootstrap", "ET_dates": len(days),
            "block_sessions": width, "replicates": reps,
            "lower": float(lower), "upper": float(upper)}


def probability_comparison(frame: pd.DataFrame, config: dict,
                           date_axis: list[str] | None = None) -> dict:
    date_axis = sorted(frame.session_date.unique()) if date_axis is None else date_axis
    q = frame[frame.quality_eligible].reset_index(drop=True).copy()
    if q.empty:
        return {"status": "no_quality_rows", "quality_n": 0}
    y = q.target.to_numpy(int)
    result = {"status": "scored", "quality_n": len(q),
              "model": binary_scores(y, q.p_touch),
              "model_raw": binary_scores(y, q.p_raw),
              "history_raw": binary_scores(y, q.h_quality_rate_63),
              "history_calibrated": binary_scores(y, q.history_calibrated_p)}
    q["brier_improvement"] = ((y - q.history_calibrated_p.to_numpy(float)) ** 2 -
                              (y - q.p_touch.to_numpy(float)) ** 2)
    result["brier_improvement_over_calibrated_history"] = float(q.brier_improvement.mean())
    b = config["bootstrap"]
    result["brier_improvement_date_block_ci"] = {
        str(width): _block_ci(q, "brier_improvement", date_axis, width, b["replicates"], b["seed"])
        for width in b["block_sessions"]}
    return result


def action_comparison(frame: pd.DataFrame, chosen: np.ndarray, config: dict,
                      date_axis: list[str] | None = None) -> dict:
    date_axis = sorted(frame.session_date.unique()) if date_axis is None else date_axis
    scenario = config["development_scenario"]
    selected = frame.loc[chosen].reset_index(drop=True)
    summary = outcome_summary(selected, config["tail_fraction"])
    if selected.empty:
        return {"status": "no_recommendation", "selected": summary,
                "history_matched": outcome_summary(selected, config["tail_fraction"]),
                "touch_lift": None, "mean_net_lift": None,
                "touch_lift_date_block_ci": None}
    matched = matched_history_rows(frame, chosen).reset_index(drop=True)
    history = outcome_summary(matched, config["tail_fraction"])
    if len(matched) != len(selected):
        raise ValueError("history baseline has different recommendation coverage")
    # Pair within each time cross-section; no future outcome chooses a match.
    counts = selected.groupby(["session_date", "cutoff_et"]).size()
    control_counts = matched.groupby(["session_date", "cutoff_et"]).size()
    if not counts.equals(control_counts):
        raise ValueError("history baseline differs in date/hour opportunity counts")
    paired = []
    for key, chosen_group in selected.groupby(["session_date", "cutoff_et"]):
        hist_group = matched[(matched.session_date == key[0]) & (matched.cutoff_et == key[1])]
        # A cross-section usually has one slot; equal-count group means are
        # enough for a date-clustered paired lift when k > 1.
        paired.append({"session_date": key[0], "touch_lift": float(chosen_group.target.mean() - hist_group.target.mean()),
                       "selected_n": len(chosen_group)})
    pair = pd.DataFrame(paired)
    # Weight cross-sections by selected opportunity count, preserving the
    # exact observation-level same-coverage lift.
    b = config["bootstrap"]
    expanded = pair.loc[pair.index.repeat(pair.selected_n)].reset_index(drop=True)
    ci = {str(width): _block_ci(expanded, "touch_lift", date_axis, width, b["replicates"], b["seed"] + 1)
          for width in b["block_sessions"]}
    return {"status": "scored", "selected": summary, "history_matched": history,
            "touch_lift": float(summary["touch_rate"] - history["touch_rate"]),
            "mean_net_lift": float(summary["mean_net"] - history["mean_net"]),
            "touch_lift_date_block_ci": ci}


def selection_status(probability: dict, action: dict, config: dict) -> dict:
    scenario = config["development_scenario"]
    p_reasons = []
    if probability.get("status") != "scored":
        p_reasons.append("no_quality_probability_rows")
    else:
        if (probability["brier_improvement_over_calibrated_history"] <
                scenario["quality_probability_brier_improvement_over_calibrated_causal_history_min"]):
            p_reasons.append("Brier_gain_below_calibrated_history_target")
        ci = probability["brier_improvement_date_block_ci"]["5"]
        if ci["lower"] is None or ci["lower"] <= scenario["quality_probability_brier_5day_block_ci_lower_min"]:
            p_reasons.append("Brier_gain_date_block_CI_not_positive_or_unavailable")
    probability_pass = not p_reasons
    a_reasons = []
    selected = action["selected"]
    if not selected["n"]:
        a_reasons.append("no_recommendation")
    if selected["n"] < scenario["min_recommended_rows"]:
        a_reasons.append("recommended_rows_below_evidence_minimum")
    if selected["stock_dates"] < scenario["min_recommended_stock_dates"]:
        a_reasons.append("recommended_stock_dates_below_evidence_minimum")
    if selected["ET_dates"] < scenario["min_recommended_ET_dates"]:
        a_reasons.append("recommended_ET_dates_below_evidence_minimum")
    insufficient = any("evidence_minimum" in reason for reason in a_reasons) or not selected["n"]
    if selected["n"]:
        if selected["touch_rate"] < scenario["observed_recommended_touch_rate_min"]:
            a_reasons.append("observed_touch_below_development_target")
        if action["touch_lift"] < scenario["matched_historical_rate_touch_lift_min"]:
            a_reasons.append("touch_lift_below_history_only_target")
        ci = action["touch_lift_date_block_ci"]["5"]
        if ci["lower"] is None or ci["lower"] <= scenario["matched_touch_lift_5day_block_ci_lower_min"]:
            a_reasons.append("touch_lift_date_block_CI_not_positive_or_unavailable")
        if selected["mean_net"] <= scenario["observed_mean_net_min"]:
            a_reasons.append("mean_net_not_positive")
        if selected["es95_loss_full"] > scenario["observed_full_sample_es95_loss_max"]:
            a_reasons.append("full_sample_ES95_above_development_scenario")
    action_status = "no_recommendation" if not selected["n"] else (
        "insufficient_action_evidence" if insufficient else (
            "passed_action_development_scenario" if not a_reasons else "failed_action_development_scenario"))
    return {"probability_status": "passed_probability_development_target" if probability_pass else
                                  "not_met_probability_development_target",
            "probability_reasons": p_reasons,
            "action_status": action_status, "action_reasons": a_reasons,
            "formal_action_enabled": False}
