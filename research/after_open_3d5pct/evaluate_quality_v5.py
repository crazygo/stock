"""Hash-verified exposed-development evaluation of the frozen v5 experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .quality_eval_v5 import (action_comparison, binary_scores, outcome_summary,
                              probability_comparison)
from .terminal_risk_v4 import REPO, sha256
from .train_quality_v5 import PROJECT, _date_axis


def _verify(run: Path) -> tuple[dict, dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    registered = PROJECT / "configs/quality_training_v5.json"
    if sha256(registered) != manifest["config_sha256"]:
        raise ValueError("v5 registered config hash mismatch")
    config = json.loads((run / "config.json").read_text())
    if config != json.loads(registered.read_text()):
        raise ValueError("v5 run config differs from registration")
    if sha256(run / "provenance_at_launch.json") != manifest["provenance_at_launch_sha256"]:
        raise ValueError("v5 launch provenance hash mismatch")
    provenance = json.loads((run / "provenance_at_launch.json").read_text())
    for relative, expected in provenance["files_sha256"].items():
        if sha256(REPO / relative) != expected:
            raise ValueError(f"v5 launch source or code changed: {relative}")
    for name, expected in manifest["result_artifacts_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"v5 result hash mismatch: {name}")
    for name, expected in manifest["model_artifacts_sha256"].items():
        if sha256(run / "models" / name) != expected:
            raise ValueError(f"v5 model hash mismatch: {name}")
    return config, manifest


def _bins(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {}
    f = frame.copy()
    f["probability_bin"] = pd.cut(f.p_touch, [0, .2, .4, .6, .8, 1],
                                  include_lowest=True).astype(str)
    return {str(key): {"n": len(group), "touches": int(group.target.sum()),
                       "observed_rate": float(group.target.mean()),
                       "mean_predicted": float(group.p_touch.mean())}
            for key, group in f.groupby("probability_bin")}


def _history_bin(value: float) -> str:
    if not np.isfinite(value):
        return "insufficient_history"
    if value <= .4:
        return "at_most_40pct"
    if value <= .6:
        return "above_40_to_60pct"
    if value <= .7:
        return "above_60_to_70pct"
    if value <= .8:
        return "above_70_to_80pct"
    return "above_80pct"


def _weighted_es_loss(returns: np.ndarray, weights: np.ndarray, tail_fraction: float) -> float:
    """Worst weighted tail of *all* opportunity returns, including fractional boundary."""
    x, w = np.asarray(returns, float), np.asarray(weights, float)
    if not len(x) or len(x) != len(w) or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("weighted ES requires finite aligned returns and weights")
    if (w <= 0).any() or not 0 < tail_fraction <= 1:
        raise ValueError("weighted ES requires positive weights and valid tail fraction")
    order = np.argsort(x)
    remaining = float(w.sum() * tail_fraction)
    mass = remaining
    total = 0.0
    for index in order:
        taken = min(float(w[index]), remaining)
        total += taken * float(x[index])
        remaining -= taken
        if remaining <= 1e-12:
            break
    return -total / mass


def _same_time_all_pool(phase: pd.DataFrame, tail_fraction: float) -> dict:
    quality = phase[phase.quality_eligible]
    if quality.empty:
        return {"status": "no_quality_rows", "quality_n": 0}
    weights = quality.groupby(["session_date", "cutoff_et"]).size().rename("quality_n")
    pool = phase.join(weights, on=["session_date", "cutoff_et"])
    pool = pool[pool.quality_n.notna()].copy()
    sizes = pool.groupby(["session_date", "cutoff_et"]).size().rename("pool_n")
    pool = pool.join(sizes, on=["session_date", "cutoff_et"])
    w = (pool.quality_n / pool.pool_n).to_numpy(float)
    pool_touch = float(np.average(pool.target, weights=w))
    pool_net = float(np.average(pool.realized_net, weights=w))
    pool_es = _weighted_es_loss(pool.realized_net.to_numpy(float), w, tail_fraction)
    quality_es = outcome_summary(quality, tail_fraction)["es95_loss_full"]
    return {"status": "descriptive_same_date_hour_weighted_all_pool",
            "quality_n": len(quality), "matching_quality_cross_sections": len(weights),
            "all_pool_touch_rate_same_time_weighted": pool_touch,
            "all_pool_mean_net_same_time_weighted": pool_net,
            "all_pool_ES95_loss_same_time_weighted": pool_es,
            "quality_touch_rate": float(quality.target.mean()),
            "quality_mean_net": float(quality.realized_net.mean()),
            "quality_ES95_loss_full": quality_es,
            "touch_difference_vs_time_weighted_pool": float(quality.target.mean() - pool_touch),
            "net_difference_vs_time_weighted_pool": float(quality.realized_net.mean() - pool_net),
            "ES95_loss_difference_vs_time_weighted_pool": float(quality_es - pool_es)}


def _group_summaries(frame: pd.DataFrame, config: dict,
                     group: str, expected_keys: list[str] | None = None) -> dict:
    result = {}
    grouped = dict(tuple(frame.groupby(group)))
    for key in expected_keys if expected_keys is not None else sorted(grouped):
        x = grouped.get(key, frame.iloc[:0])
        result[str(key)] = {"quality_pool": outcome_summary(x[x.quality_eligible], config["tail_fraction"]),
                            "recommended": outcome_summary(x[x.selected_quality_scenario], config["tail_fraction"]),
                            "all_pool": outcome_summary(x, config["tail_fraction"])}
    return result


def _candidate_metrics(frame: pd.DataFrame, config: dict,
                       date_axis: list[str]) -> dict:
    quality = frame[frame.quality_eligible]
    result = {"all_pool": outcome_summary(frame, config["tail_fraction"]),
              "quality_pool": outcome_summary(quality, config["tail_fraction"]),
              "probability": probability_comparison(frame, config, date_axis),
              "action": action_comparison(frame, frame.selected_quality_scenario.to_numpy(bool),
                                          config, date_axis),
              "calibration_bins_quality": _bins(quality),
              "by_hour": _group_summaries(frame, config, "cutoff_et",
                                          config["quality"]["expected_distinct_cutoffs_et"]),
              "by_date": _group_summaries(frame, config, "session_date", date_axis),
              "by_symbol": _group_summaries(frame, config, "symbol")}
    binned = frame.copy()
    binned["history_bin"] = binned.h_quality_rate_63.map(_history_bin)
    result["by_historical_rate_bin"] = _group_summaries(
        binned, config, "history_bin",
        ["insufficient_history", "at_most_40pct", "above_40_to_60pct",
         "above_60_to_70pct", "above_70_to_80pct", "above_80pct"])
    return result


def _quality_source_audit(run: Path, config: dict) -> dict:
    v4 = REPO / config["source_v4_run"]
    ledger = pd.read_parquet(v4 / "terminal_outcomes.parquet")
    q = pd.read_parquet(run / "quality_features.parquet")
    counts = ledger.groupby(["symbol", "session_date"]).agg(
        rows=("cutoff_et", "size"), distinct_cutoffs=("cutoff_et", "nunique"),
        mature=("status", lambda s: bool(s.eq("mature").all())))
    complete = ((counts.rows == 6) & (counts.distinct_cutoffs == 6) & counts.mature)
    examples = counts.loc[~complete].head(20).reset_index().to_dict("records")
    return {"source_rows": len(ledger), "source_stock_dates": len(counts),
            "source_complete_six_cutoff_stock_dates": int(complete.sum()),
            "source_incomplete_or_not_mature_stock_dates": int((~complete).sum()),
            "incomplete_examples": examples,
            "quality_feature_rows": len(q),
            "quality_eligible_rows": int(q.quality_eligible.sum()),
            "quality_insufficient_history_rows": int((q.quality_status == "insufficient_history").sum()),
            "quality_leq_60_rows": int((q.quality_status == "below_or_equal_threshold").sum()),
            "source_duplicate_sample_ids": int(ledger.sample_id.duplicated().sum()),
            "source_duplicate_symbol_date_cutoffs": int(
                ledger.duplicated(["symbol", "session_date", "cutoff_et"]).sum())}


def evaluate(run: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config, manifest = _verify(run)
    frames = pd.read_parquet(run / "predictions.parquet")
    if frames.duplicated(["sample_id", "fold", "phase", "candidate"]).any():
        raise ValueError("duplicate v5 prediction keys")
    trials = [json.loads(line) for line in (run / "trials.jsonl").read_text().splitlines()]
    failed = {(t["fold"], t["candidate"]) for t in trials if t["status"] == "failed"}
    for row in frames[["fold", "candidate"]].drop_duplicates().itertuples():
        if (row.fold, row.candidate) in failed:
            raise ValueError("failed trial leaked into persisted predictions")
    calendar = json.loads((REPO / config["calendar"]).read_text())["sessions"]
    result = {"protocol": config["protocol_version"],
              "scope": "exposed_development_outer_not_independent",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "training_manifest_sha256": sha256(run / "manifest.json"),
              "evaluation_code_sha256": sha256(Path(__file__)),
              "quality_source_audit": _quality_source_audit(run, config),
              "candidate_trials_completed": manifest["completed_trials"],
              "candidate_trials_failed": manifest["failed_trials"],
              "formal_action_enabled": False, "folds": {}}
    for fold in config["folds"]:
        name = fold["name"]
        axis = _date_axis(calendar, fold, "outer")
        phase = frames[(frames.fold == name) & (frames.phase == "outer")]
        reference = phase[phase.candidate == "v4_lgbm_HAB"]
        if reference.empty:
            raise ValueError("frozen v4 same-sample all-pool reference absent")
        metrics = {}
        for candidate, group in phase.groupby("candidate"):
            metrics[candidate] = _candidate_metrics(group.reset_index(drop=True), config, axis)
        for candidate, value in metrics.items():
            if candidate.startswith("v4_") or candidate.endswith("pooled_control"):
                value["same_family_pooled_control_brier_improvement"] = None
                continue
            parts = candidate.split("_")
            control_key = f"{parts[0]}_{parts[1]}_pooled_control"
            control = metrics.get(control_key)
            value["same_family_pooled_control"] = control_key
            value["same_family_pooled_control_brier_improvement"] = (
                control["probability"]["model"]["brier"] - value["probability"]["model"]["brier"]
                if control and control["probability"].get("status") == "scored" and
                value["probability"].get("status") == "scored" else None)
        selected_prob = manifest["folds"][name]["best_inner_probability_candidate"]
        selected_action = manifest["folds"][name]["best_inner_action_candidate"]
        quality_reference = reference[reference.quality_eligible]
        result["folds"][name] = {
            "selection_frozen_before_outer": {
                "best_inner_probability_candidate": selected_prob,
                "best_inner_probability_status": manifest["folds"][name]["best_inner_probability_status"],
                "best_inner_action_candidate": selected_action},
            "all_pool": outcome_summary(reference, config["tail_fraction"]),
            "quality_pool": outcome_summary(quality_reference, config["tail_fraction"]),
            "quality_pool_vs_same_time_all_pool": _same_time_all_pool(reference, config["tail_fraction"]),
            "history_probability_baselines": {
                "raw": binary_scores(quality_reference.target, quality_reference.h_quality_rate_63),
                "calibrated": binary_scores(quality_reference.target, quality_reference.history_calibrated_p)},
            "outer_result_of_inner_probability_candidate": metrics.get(selected_prob),
            "outer_result_of_inner_action_candidate": metrics.get(selected_action),
            "candidates": metrics, "selection_uses_outer": False}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluated = evaluate(args.run.resolve(), args.output.resolve())
    for name, fold in evaluated["folds"].items():
        print(json.dumps({"fold": name, "quality_n": fold["quality_pool"]["n"],
                          "quality_touch_rate": fold["quality_pool"]["touch_rate"],
                          "inner_probability_choice": fold["selection_frozen_before_outer"]["best_inner_probability_candidate"],
                          "inner_action_choice": fold["selection_frozen_before_outer"]["best_inner_action_candidate"]},
                         ensure_ascii=False))
