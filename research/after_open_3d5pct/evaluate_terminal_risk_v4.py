"""Expose proper scores, causal coverage, matched controls and date-block uncertainty."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .terminal_risk_v4 import full_sample_es_loss, net_return, pinball, sha256
from .train_terminal_risk_v4 import _score_binary, fold_indices, select_cross_section

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]


def _summary(frame: pd.DataFrame, selected: pd.DataFrame, tail: float) -> dict:
    n = len(selected)
    failures = selected[selected.target == 0] if n else selected
    failure_gross = failures.failure_actual_gross.to_numpy(dtype=float) if n else np.array([])
    return {"pool_n": len(frame), "selected_n": n,
            "coverage": n / len(frame) if len(frame) else None,
            "selected_dates": int(selected.session_date.nunique()) if n else 0,
            "touches": int(selected.target.sum()) if n else 0,
            "touch_rate": float(selected.target.mean()) if n else None,
            "mean_net": float(selected.realized_net.mean()) if n else None,
            "predicted_mean_net": float(selected.expected_net.mean()) if n else None,
            "realized_terminal_loss_rate": float((selected.realized_net < 0).mean()) if n else None,
            "predicted_terminal_loss_probability_mean": float(selected.terminal_loss_probability.mean()) if n else None,
            "es95_loss_full_selected": full_sample_es_loss(selected.realized_net.to_numpy(), tail) if n else None,
            "failure_condition_n": len(failures),
            "failure_terminal_gross_mean": float(np.mean(failure_gross)) if len(failures) else None,
            "failure_terminal_gross_quantiles_10_50_90": np.quantile(failure_gross, [.1, .5, .9]).tolist() if len(failures) else None,
            "failure_terminal_net_mean": float(failures.realized_net.mean()) if len(failures) else None}


def _block_ci(frame: pd.DataFrame, control: pd.DataFrame, config: dict, seed: int) -> dict:
    """Sample contiguous ET date blocks, retaining all symbols/hours in each date."""
    days = sorted(frame.session_date.unique())
    if len(days) < 2 or len(frame) != len(control):
        return {"status": "insufficient_paired_dates"}
    date_indices = {d: np.flatnonzero(frame.session_date.to_numpy() == d) for d in days}
    rng = np.random.default_rng(seed)
    result = {}
    for width in config["bootstrap"]["block_sessions"]:
        if len(days) < 2 * width:
            result[f"{width}_session_blocks"] = {"status": "insufficient_independent_calendar_blocks",
                                                     "calendar_dates": len(days), "minimum_dates": 2 * width,
                                                     "touch_lift_ci95": None, "mean_net_lift_ci95": None}
            continue
        starts = np.arange(max(1, len(days) - width + 1))
        values = []
        for _ in range(config["bootstrap"]["replicates"]):
            selected_days = []
            while len(selected_days) < len(days):
                begin = int(rng.choice(starts))
                selected_days.extend(days[begin:begin + width])
            ids = np.concatenate([date_indices[d] for d in selected_days[:len(days)]])
            a, b = frame.iloc[ids], control.iloc[ids]
            values.append((float(a.target.mean() - b.target.mean()),
                           float(a.realized_net.mean() - b.realized_net.mean())))
        values = np.asarray(values)
        result[f"{width}_session_blocks"] = {
            "status": "development_block_bootstrap_sensitivity",
            "touch_lift_ci95": np.quantile(values[:, 0], [.025, .975]).tolist(),
            "mean_net_lift_ci95": np.quantile(values[:, 1], [.025, .975]).tolist(),
            "calendar_dates": len(days), "replicates": len(values)}
    return result


def _matched_controls(frame: pd.DataFrame, selected: pd.DataFrame, train_features: pd.DataFrame,
                      config: dict) -> tuple[dict, pd.DataFrame]:
    if selected.empty:
        return {"status": "no_selected_rows"}, selected.copy()
    bins = config["control"]["quantile_bins"]
    edges = {}
    for col in ("b_range_actual", "b_dollar_actual_log"):
        q = np.quantile(train_features[col].dropna(), np.linspace(0, 1, bins + 1))
        q[0], q[-1] = -np.inf, np.inf
        edges[col] = q
    pool = frame.copy()
    for col in edges:
        pool[f"{col}_bin"] = np.digitize(pool[col], edges[col][1:-1], right=True)
    selected_pool = pool.loc[pool.sample_id.isin(selected.sample_id)].copy()
    nonselected = pool.loc[~pool.sample_id.isin(selected.sample_id)].reset_index(drop=True)
    rng = np.random.default_rng(config["control"]["seed"])
    repetitions = config["control"]["repetitions"]
    sampled = []
    strict_groups = nonselected.groupby(["session_date", "cutoff_et", "b_range_actual_bin",
                                          "b_dollar_actual_log_bin"]).indices
    time_groups = nonselected.groupby(["session_date", "cutoff_et"]).indices
    choices = []
    fallback_rows = 0
    for row in selected_pool.itertuples():
        strict_key = (row.session_date, row.cutoff_et,
                      row.b_range_actual_bin, row.b_dollar_actual_log_bin)
        match = strict_groups.get(strict_key)
        if match is None or not len(match):
            match = time_groups.get((row.session_date, row.cutoff_et))
            fallback_rows += 1
        if match is None or not len(match):
            return {"status": "unmatched_date_hour"}, selected.iloc[0:0]
        choices.append(match)
    for _ in range(repetitions):
        positions = [int(rng.choice(match)) for match in choices]
        sampled.append(nonselected.iloc[positions].reset_index(drop=True))
    fallback = fallback_rows * repetitions
    touch = np.array([f.target.mean() for f in sampled])
    net = np.array([f.realized_net.mean() for f in sampled])
    es = np.array([full_sample_es_loss(f.realized_net.to_numpy(), config["tail_fraction"]) for f in sampled])
    average = selected_pool[["sample_id", "session_date"]].reset_index(drop=True).copy()
    average["target"] = np.mean(np.stack([f.target.to_numpy() for f in sampled]), axis=0)
    average["realized_net"] = np.mean(np.stack([f.realized_net.to_numpy() for f in sampled]), axis=0)
    comparison = {"status": "strict_matched_development_control" if fallback == 0 else
                             "matched_with_same_date_hour_fallback_diagnostic",
                  "repetitions": repetitions,
                  "matched_n_each_draw": len(selected),
                  "same_date_hour_and_train_fitted_vol_liquidity_bins": fallback == 0,
                  "fallback_to_same_date_hour_draws": fallback,
                  "fallback_fraction": fallback / (repetitions * len(selected)),
                  "matched_touch_rate_mean": float(touch.mean()),
                  "matched_mean_net_mean": float(net.mean()),
                  "matched_full_sample_es95_loss_mean": float(es.mean()),
                  "touch_lift": float(selected.target.mean() - touch.mean()),
                  "mean_net_lift": float(selected.realized_net.mean() - net.mean())}
    comparison["date_block_ci"] = _block_ci(selected.reset_index(drop=True), average, config,
                                            config["bootstrap"]["seed"])
    return comparison, average


def _risk_scores(frame: pd.DataFrame) -> dict:
    failure = frame[frame.target == 0]
    if failure.empty:
        return {"failure_n": 0, "status": "unavailable"}
    y = failure.realized_net.to_numpy()
    # Risk model predicts gross failure return. Convert actual net back with
    # the run's explicitly recorded cost outside this function for scoring.
    gross = failure.failure_actual_gross.to_numpy()
    mean = failure.failure_mean_gross.to_numpy()
    loss = (y < 0).astype(int)
    return {"failure_n": len(failure),
            "mean_mse": float(np.mean((gross - mean) ** 2)),
            "mean_rmse": float(np.sqrt(np.mean((gross - mean) ** 2))),
            "mean_mae_auxiliary": float(np.mean(np.abs(gross - mean))),
            "pinball_q10": pinball(gross, failure.failure_q10_gross, .1),
            "pinball_q50": pinball(gross, failure.failure_q50_gross, .5),
            "pinball_q90": pinball(gross, failure.failure_q90_gross, .9),
            "q10_empirical_below": float((gross <= failure.failure_q10_gross).mean()),
            "q90_empirical_below": float((gross <= failure.failure_q90_gross).mean()),
            "q10_q90_interval_coverage": float(((gross >= failure.failure_q10_gross) &
                                                  (gross <= failure.failure_q90_gross)).mean()),
            "conditional_loss_brier": float(np.mean((loss - failure.failure_loss_probability.to_numpy()) ** 2)),
            "conditional_loss_rate": float(loss.mean()),
            "predicted_conditional_loss_mean": float(failure.failure_loss_probability.mean())}


def evaluate(run: Path, output: Path | None = None) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    registered_config = PROJECT / "configs/terminal_risk_v4.json"
    if sha256(registered_config) != manifest["config_sha256"]:
        raise ValueError("v4 preregistered source config hash mismatch")
    if json.loads((run / "config.json").read_text()) != json.loads(registered_config.read_text()):
        raise ValueError("v4 run config content differs from preregistered source")
    for name, expected in manifest["result_artifacts_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"v4 result artifact hash mismatch: {name}")
    for name, expected in manifest["model_artifacts_sha256"].items():
        if sha256(run / "models" / name) != expected:
            raise ValueError(f"v4 model artifact hash mismatch: {name}")
    if sha256(run / "provenance_at_launch.json") != manifest["provenance_at_launch_sha256"]:
        raise ValueError("v4 launch provenance hash mismatch")
    config = json.loads((run / "config.json").read_text())
    source = REPO / config["source_run"]
    rows = pd.read_parquet(source / "features.parquet").merge(
        pd.read_parquet(source / "outcomes.parquet")[["sample_id", "label_end_at", "label_available_at"]],
        on="sample_id", validate="one_to_one")
    outcome = pd.read_parquet(run / "terminal_outcomes.parquet")
    pred = pd.read_parquet(run / "predictions.parquet")
    pred = pred.merge(outcome[["sample_id", "gross_evaluation_return"]], on="sample_id", validate="many_to_one")
    pred["failure_actual_gross"] = pred.gross_evaluation_return.where(pred.target == 0)
    result = {"protocol": config["protocol_version"],
              "scope": "exposed_development_outer_not_independent",
              "training_manifest_sha256": sha256(run / "manifest.json"),
              "full_window_statuses": outcome.status.value_counts().to_dict(),
              "predicted_es95_loss": "unavailable_three_quantiles_insufficient_for_mixture_tail_integral",
              "folds": {}}
    for fold in config["folds"]:
        name = fold["name"]
        indices, _ = fold_indices(rows, fold)
        train_features = rows.iloc[indices["train"]]
        chosen_model = manifest["folds"][name]["selected_candidate"]
        fold_result = {"selection_frozen_before_outer": chosen_model,
                       "candidates": {}, "risk_scores_outer_failure_only": None,
                       "primary_scenario": config["development_scenario"],
                       "formal_action_enabled": False}
        for key in [f"{f}_{i}" for i in config["candidate_input_sets"] for f in config["candidate_families"]]:
            phase = pred[(pred.fold == name) & (pred.phase == "outer") & (pred.candidate == key)]
            if phase.empty:
                fold_result["candidates"][key] = {"status": "trial_failed_or_no_outer_predictions"}
                continue
            selected = phase[phase.selected_primary_scenario]
            outcome_metrics = _summary(phase, selected, config["tail_fraction"])
            touch_scores = {"raw": _score_binary(phase.target.to_numpy(), phase.p_raw.to_numpy()),
                            "calibrated": _score_binary(phase.target.to_numpy(), phase.p_touch.to_numpy())}
            hours = {}
            for hour, group in phase.groupby("cutoff_et"):
                hours[hour] = _summary(group, group[group.selected_primary_scenario], config["tail_fraction"])
            days = {}
            for day, group in phase.groupby("session_date"):
                days[day] = _summary(group, group[group.selected_primary_scenario], config["tail_fraction"])
            matched, _ = _matched_controls(phase.reset_index(drop=True), selected.reset_index(drop=True),
                                           train_features, config)
            fold_result["candidates"][key] = {"status": "outer_evaluation_only",
                                               "is_inner_selected": key == chosen_model,
                                               "touch_scores": touch_scores,
                                               "all_pool": _summary(phase, phase, config["tail_fraction"]),
                                               "selected": outcome_metrics,
                                               "by_hour": hours,
                                               "by_date": days,
                                               "matched_control": matched,
                                               "whole_pipeline_information_range": "HAB"}
            if fold_result["risk_scores_outer_failure_only"] is None:
                fold_result["risk_scores_outer_failure_only"] = _risk_scores(phase)
        if chosen_model != "no_candidate_passed":
            winner = pred[(pred.fold == name) & (pred.phase == "outer") & (pred.candidate == chosen_model)]
            sensitivities = []
            for budget in config["sensitivity"]["budgets"]:
                for q10 in config["sensitivity"]["failure_q10_net_min"]:
                    mask = select_cross_section(winner.reset_index(drop=True), config["development_scenario"],
                                                budget=budget, q10_min=q10)
                    s = winner.reset_index(drop=True).loc[mask]
                    metrics = _summary(winner, s, config["tail_fraction"])
                    metrics.update({"budget": budget, "failure_q10_net_min": q10,
                                    "es95_loss_max_scenarios": {str(limit):
                                                                metrics["es95_loss_full_selected"] is not None and
                                                                metrics["es95_loss_full_selected"] <= limit
                                                                for limit in config["sensitivity"]["es95_loss_max"]},
                                    "selection_rule_frozen": True,
                                    "status": "development_sensitivity_not_formal_action"})
                    sensitivities.append(metrics)
            fold_result["sensitivity"] = sensitivities
        result["folds"][name] = fold_result
    target = output if output is not None else run / "evaluation.json"
    if target.exists():
        raise FileExistsError(f"evaluation output already exists: {target}")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new immutable evaluation JSON path")
    args = parser.parse_args()
    result = evaluate(args.run.resolve(), args.output.resolve() if args.output else None)
    for fold, value in result["folds"].items():
        chosen = value["selection_frozen_before_outer"]
        print(f"{fold}: inner selected {chosen}")
        if chosen in value["candidates"]:
            print(json.dumps(value["candidates"][chosen]["selected"], ensure_ascii=False))
