"""Offline nested development experiment for touch and terminal valuation."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import platform

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error

from .hourly_v3_data import Bundle
from .terminal_risk_v4 import (REPO, derive_terminal_outcomes, full_sample_es_loss,
                               mixed_expected_net, net_return, pinball, sha256)
from .train_hourly_v3 import (_fit_calibrator, _fit_tcn, _static, _tabular,
                              apply_calibrator)

PROJECT = Path(__file__).resolve().parent
UTC = timezone.utc


def fold_indices(rows: pd.DataFrame, fold: dict) -> tuple[dict[str, np.ndarray], dict]:
    decision = pd.to_datetime(rows.decision_at, utc=True)
    end = pd.to_datetime(rows.label_end_at, utc=True)
    available = pd.to_datetime(rows.label_available_at, utc=True)
    boundaries = {k: pd.Timestamp(fold[k], tz="America/New_York").tz_convert("UTC")
                  for k in ("early_start", "calibration_start", "selection_start", "outer_start", "outer_end_exclusive")}
    specs = {
        "train": (None, "early_start"),
        "early": ("early_start", "calibration_start"),
        "calibration": ("calibration_start", "selection_start"),
        "selection": ("selection_start", "outer_start"),
        "outer": ("outer_start", "outer_end_exclusive"),
    }
    indices, audit = {}, {}
    for name, (start_key, stop_key) in specs.items():
        stop = boundaries[stop_key]
        range_mask = decision < stop
        if start_key:
            range_mask &= decision >= boundaries[start_key]
        mature_mask = (end < stop) & (available < stop) if name != "outer" else pd.Series(True, index=rows.index)
        selected = (range_mask & mature_mask).to_numpy()
        indices[name] = np.flatnonzero(selected)
        audit[name] = {"count": int(selected.sum()),
                       "dates": int(rows.loc[selected, "session_date"].nunique()),
                       "purged_by_window_or_availability": int((range_mask & ~mature_mask).sum())}
    if min(len(v) for v in indices.values()) == 0:
        raise ValueError(f"empty nested fold: {audit}")
    return indices, audit


def _score_binary(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return {"brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, p, labels=[0, 1])),
            "observed_rate": float(np.mean(y)), "mean_prediction": float(np.mean(p))}


def _brier_metric(y_true, y_pred):
    return "brier", float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)), False


def _risk_prediction(models: dict, X: pd.DataFrame, adjustment: dict | None = None) -> dict[str, np.ndarray]:
    mean = models["mean"].predict(X)
    qs = np.column_stack([models[f"q{int(q*100)}"].predict(X) for q in (.1, .5, .9)])
    loss_raw = models["loss"].predict_proba(X)[:, 1]
    if adjustment:
        mean = mean + adjustment["mean_offset"]
        qs = qs + np.asarray(adjustment["quantile_offsets"])
        loss_p = apply_calibrator(loss_raw, adjustment["loss_calibrator"])
    else:
        loss_p = loss_raw
    qs = np.sort(qs, axis=1)
    return {"failure_mean_gross": np.asarray(mean, dtype=float),
            "failure_q10_gross": qs[:, 0], "failure_q50_gross": qs[:, 1],
            "failure_q90_gross": qs[:, 2], "failure_loss_probability": loss_p,
            "failure_loss_probability_raw": loss_raw}


def _fit_risk_head(bundle: Bundle, outcomes: pd.DataFrame, indices: dict,
                   config: dict, output: Path, fold_name: str) -> tuple[dict, dict, dict]:
    x = _tabular(bundle, "HAB")
    y = outcomes.gross_evaluation_return.to_numpy(float)
    touch = outcomes.target.to_numpy(int)
    train = indices["train"][touch[indices["train"]] == 0]
    early = indices["early"][touch[indices["early"]] == 0]
    calibration = indices["calibration"][touch[indices["calibration"]] == 0]
    if min(len(train), len(early), len(calibration)) < 100:
        raise ValueError("insufficient mature no-touch observations for risk head")
    print(f"{fold_name} risk failure n train/early/cal={len(train)}/{len(early)}/{len(calibration)}", flush=True)
    models, log = {}, {}
    common = {"n_estimators": 160, "learning_rate": 0.04, "num_leaves": 9,
              "max_depth": 4, "min_child_samples": 180, "reg_lambda": 5.0,
              "verbosity": -1, "n_jobs": 4, "random_state": config["seed"]}
    for key, objective, alpha in (("mean", "regression", None),
                                  ("q10", "quantile", .1), ("q50", "quantile", .5),
                                  ("q90", "quantile", .9)):
        params = dict(common, objective=objective)
        if alpha is not None:
            params["alpha"] = alpha
        model = lgb.LGBMRegressor(**params)
        curve = {}
        started = time.monotonic()
        model.fit(x.iloc[train], y[train], eval_set=[(x.iloc[early], y[early])],
                  callbacks=[lgb.early_stopping(15, verbose=False), lgb.record_evaluation(curve)])
        path = output / f"{fold_name}_risk_{key}.txt"
        model.booster_.save_model(str(path))
        models[key] = model
        log[key] = {"seconds": time.monotonic() - started, "best_iteration": model.best_iteration_,
                    "early_curve": curve, "file": path.name}
    loss_target = (net_return(y, config["buy_cost_rate"], config["sell_cost_rate"]) < 0).astype(int)
    if len(np.unique(loss_target[train])) < 2 or len(np.unique(loss_target[calibration])) < 2:
        raise ValueError("failure loss classifier or calibration has one class")
    loss_model = lgb.LGBMClassifier(**dict(common, objective="binary"))
    curve = {}
    started = time.monotonic()
    loss_model.fit(x.iloc[train], loss_target[train], eval_set=[(x.iloc[early], loss_target[early])],
                   eval_metric="binary_logloss",
                   callbacks=[lgb.early_stopping(15, verbose=False), lgb.record_evaluation(curve)])
    path = output / f"{fold_name}_risk_loss.txt"
    loss_model.booster_.save_model(str(path))
    models["loss"] = loss_model
    log["loss"] = {"seconds": time.monotonic() - started, "best_iteration": loss_model.best_iteration_,
                   "early_curve": curve, "file": path.name}
    raw_cal = _risk_prediction(models, x.iloc[calibration])
    offsets = [float(np.quantile(y[calibration] - raw_cal[f"failure_q{int(q*100)}_gross"], q))
               for q in (.1, .5, .9)]
    adjustment = {"mean_offset": float(np.mean(y[calibration] - raw_cal["failure_mean_gross"])),
                  "quantile_offsets": offsets,
                  "loss_calibrator": _fit_calibrator(loss_target[calibration],
                                                     raw_cal["failure_loss_probability_raw"]),
                  "calibration_n": int(len(calibration)),
                  "failure_only": True, "feature_columns": x.columns.tolist(),
                  "risk_head_input_set": "HAB"}
    path = output / f"{fold_name}_risk.calibration.json"
    path.write_text(json.dumps(adjustment, ensure_ascii=False, indent=2) + "\n")
    log["calibration_file"] = path.name
    predictions = {name: _risk_prediction(models, x.iloc[ix], adjustment)
                   for name, ix in indices.items() if name in ("selection", "outer")}
    return predictions, log, adjustment


def select_cross_section(frame: pd.DataFrame, scenario: dict, budget: float | None = None,
                         q10_min: float | None = None) -> np.ndarray:
    """Make every choice from only that date/hour's cross-section."""
    budget = scenario["opportunity_budget_fraction"] if budget is None else budget
    q10_min = scenario["predicted_failure_q10_net_min"] if q10_min is None else q10_min
    chosen = np.zeros(len(frame), dtype=bool)
    for _, group in frame.groupby(["session_date", "cutoff_et"], sort=False):
        k = math.ceil(len(group) * budget)
        eligible = group[(group.expected_net > scenario["predicted_expected_net_min"]) &
                         (group.failure_q10_net >= q10_min)]
        top = eligible.sort_values(["p_touch", "symbol"], ascending=[False, True]).head(k)
        chosen[top.index.to_numpy()] = True
    return chosen


def selection_metrics(frame: pd.DataFrame, chosen: np.ndarray, scenario: dict,
                      *, tail_fraction: float) -> dict:
    k_budget = sum(math.ceil(len(g) * scenario["opportunity_budget_fraction"])
                   for _, g in frame.groupby(["session_date", "cutoff_et"]))
    selected = frame.loc[chosen]
    n = len(selected)
    metrics = {"pool_n": len(frame), "budget_n": k_budget, "selected_n": n,
               "coverage": n / len(frame), "budget_fill": n / k_budget,
               "selected_dates": int(selected.session_date.nunique()) if n else 0,
               "touches": int(selected.target.sum()) if n else 0,
               "touch_rate": float(selected.target.mean()) if n else None,
               "mean_net": float(selected.realized_net.mean()) if n else None,
               "es95_loss_full_selected": full_sample_es_loss(selected.realized_net.to_numpy(), tail_fraction) if n else None}
    reasons = []
    if n < scenario["min_selected_count"]:
        reasons.append("selected_count_below_minimum")
    if metrics["selected_dates"] < scenario["min_selection_dates"]:
        reasons.append("selection_dates_below_minimum")
    if n < k_budget:
        reasons.append("per_cross_section_budget_not_filled")
    if n and metrics["mean_net"] <= scenario["observed_selected_mean_net_min"]:
        reasons.append("realized_mean_net_not_positive")
    if n and metrics["es95_loss_full_selected"] > scenario["observed_selected_es95_loss_max"]:
        reasons.append("realized_full_sample_es95_exceeds_development_scenario")
    metrics["selection_status"] = "passed_development_scenario" if not reasons else "failed_development_scenario"
    metrics["failure_reasons"] = reasons
    return metrics


def _candidate_frame(rows: pd.DataFrame, outcomes: pd.DataFrame, ix: np.ndarray,
                     raw: np.ndarray, calibrated: np.ndarray, risk: dict,
                     config: dict) -> pd.DataFrame:
    source = rows.iloc[ix].reset_index(drop=True)
    actual = outcomes.iloc[ix].reset_index(drop=True)
    f = source[["sample_id", "symbol", "session_date", "cutoff_et", "b_range_actual", "b_dollar_actual_log"]].copy()
    f["target"] = actual.target.to_numpy(int)
    f["realized_net"] = actual.net_evaluation_return.to_numpy(float)
    f["p_raw"] = raw
    f["p_touch"] = calibrated
    for key, values in risk.items():
        f[key] = values
    f["expected_net"] = mixed_expected_net(calibrated, f.failure_mean_gross.to_numpy(),
                                             touch_gross=config["touch_gross_return"],
                                             buy_cost=config["buy_cost_rate"], sell_cost=config["sell_cost_rate"])
    f["failure_q10_net"] = net_return(f.failure_q10_gross.to_numpy(),
                                       config["buy_cost_rate"], config["sell_cost_rate"])
    f["terminal_loss_probability"] = (1 - calibrated) * f.failure_loss_probability.to_numpy()
    f["predicted_es95_loss"] = np.nan  # three quantiles cannot identify mixture ES95
    return f


def run(config_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    if config["protocol_version"] != "three_day_touch_terminal_valuation_v4":
        raise ValueError("unsupported terminal valuation protocol")
    source_run = REPO / config["source_run"]
    v3_config_path = source_run / "config.json"
    if sha256(v3_config_path) != sha256(REPO / config["source_config"]):
        raise ValueError("v3 actual training config differs from registered source_config")
    started = time.monotonic()
    output.mkdir(parents=True)
    model_dir = output / "models"
    model_dir.mkdir()
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    source_manifest = json.loads((source_run / "manifest.json").read_text())
    provenance_paths = [config_path, PROJECT / "docs/17_terminal_risk_v4_registration.md",
                        PROJECT / "terminal_risk_v4.py", PROJECT / "train_terminal_risk_v4.py",
                        PROJECT / "train_hourly_v3.py", PROJECT / "hourly_v3_data.py",
                        v3_config_path, REPO / config["source_config"],
                        source_run / "manifest.json", REPO / config["calendar"],
                        REPO / json.loads(v3_config_path.read_text())["universe"]]
    provenance = {"captured_at_utc": datetime.now(UTC).isoformat(),
                  "timing": "before_outcome_derivation_and_before_any_model_fit",
                  "files_sha256": {str(p.relative_to(REPO)): sha256(p) for p in provenance_paths},
                  "versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "lightgbm": lgb.__version__,
                               "sklearn": sklearn.__version__, "torch": torch.__version__},
                  "v3_actual_config_sha256": sha256(v3_config_path),
                  "v3_registered_source_config_sha256": sha256(REPO / config["source_config"])}
    (output / "provenance_at_launch.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print("deriving complete-window terminal outcomes", flush=True)
    outcomes, source_audit = derive_terminal_outcomes(source_run, config)
    outcomes.to_parquet(output / "terminal_outcomes.parquet", index=False)
    rows = pd.read_parquet(source_run / "features.parquet").merge(
        pd.read_parquet(source_run / "outcomes.parquet")[["sample_id", "label_end_at", "label_available_at", "target"]],
        on="sample_id", validate="one_to_one")
    if not (rows.sample_id.to_numpy() == outcomes.sample_id.to_numpy()).all():
        raise ValueError("frozen feature and terminal row order mismatch")
    if (outcomes.status != "mature").any():
        raise ValueError(f"frozen v3 rows are no longer all mature: {outcomes.status.value_counts().to_dict()}")
    with np.load(source_run / "sequences.npz") as seq:
        bundle = Bundle(rows=rows, h_seq=seq["h"], post_seq=seq["post"],
                        pre_seq=seq["pre"], b_seq=seq["b"],
                        exclusions={}, coverage={}, source_paths=[])
    print(f"verified {len(rows)} mature outcomes in {time.monotonic()-started:.1f}s", flush=True)
    config_v3 = json.loads((source_run / "config.json").read_text())
    y = rows.target.to_numpy(int)
    trials, predictions, fold_reports = [], [], {}
    for fold in config["folds"]:
        name = fold["name"]
        indices, split_audit = fold_indices(rows, fold)
        print(f"{name} nested split {split_audit}", flush=True)
        risk_predictions, risk_log, risk_adjustment = _fit_risk_head(bundle, outcomes, indices, config,
                                                                     model_dir, name)
        fold_trials = []
        for input_set in config["candidate_input_sets"]:
            for family in config["candidate_families"]:
                key = f"{family}_{input_set}"
                trial_started = time.monotonic()
                try:
                    model_path = model_dir / f"{name}_{key}.{'txt' if family == 'lgbm' else 'pt'}"
                    cal_ix, select_ix, outer_ix = (indices[k] for k in ("calibration", "selection", "outer"))
                    if family == "lgbm":
                        x = _tabular(bundle, input_set)
                        model = lgb.LGBMClassifier(**dict(config_v3["lightgbm"], metric="None"),
                                                   random_state=config["seed"])
                        curve = {}
                        model.fit(x.iloc[indices["train"]], y[indices["train"]],
                                  eval_set=[(x.iloc[indices["early"]], y[indices["early"]])],
                                  eval_metric=_brier_metric,
                                  callbacks=[lgb.early_stopping(20, first_metric_only=True, verbose=False),
                                             lgb.record_evaluation(curve)])
                        raw_cal = model.predict_proba(x.iloc[cal_ix])[:, 1]
                        raw_select = model.predict_proba(x.iloc[select_ix])[:, 1]
                        raw_outer = model.predict_proba(x.iloc[outer_ix])[:, 1]
                        model.booster_.save_model(str(model_path))
                        details = {"best_iteration": int(model.best_iteration_),
                                   "feature_columns": x.columns.tolist(), "early_curve": curve}
                    else:
                        together = np.concatenate((select_ix, outer_ix))
                        raw_both, raw_cal, details = _fit_tcn(bundle, input_set,
                                                               indices["train"], indices["early"],
                                                               cal_ix, together, config_v3, model_path)
                        raw_select, raw_outer = raw_both[:len(select_ix)], raw_both[len(select_ix):]
                    calibrator = _fit_calibrator(y[cal_ix], raw_cal)
                    cal_path = model_dir / f"{name}_{key}.calibration.json"
                    cal_path.write_text(json.dumps({"calibrator": calibrator,
                                                    "calibration_n": len(cal_ix),
                                                    "source": "nested_calibration_only"},
                                                   ensure_ascii=False, indent=2) + "\n")
                    frame_select = _candidate_frame(rows, outcomes, select_ix, raw_select,
                                                    apply_calibrator(raw_select, calibrator),
                                                    risk_predictions["selection"], config)
                    frame_outer = _candidate_frame(rows, outcomes, outer_ix, raw_outer,
                                                   apply_calibrator(raw_outer, calibrator),
                                                   risk_predictions["outer"], config)
                    chosen_select = select_cross_section(frame_select, config["development_scenario"])
                    selected = selection_metrics(frame_select, chosen_select, config["development_scenario"],
                                                 tail_fraction=config["tail_fraction"])
                    selected["candidate"] = key
                    selected["touch_scores"] = {"raw": _score_binary(y[select_ix], raw_select),
                                                 "calibrated": _score_binary(y[select_ix], frame_select.p_touch)}
                    elapsed = time.monotonic() - trial_started
                    trial = {"fold": name, "candidate": key, "status": "completed",
                             "seconds": elapsed, "training": details, "selection": selected,
                             "calibration": calibrator, "checkpoint": model_path.name,
                             "calibration_file": cal_path.name,
                             "touch_information_range": input_set,
                             "whole_pipeline_information_range": "HAB"}
                    for phase, f, chosen in (("selection", frame_select, chosen_select),
                                              ("outer", frame_outer, select_cross_section(frame_outer, config["development_scenario"]))):
                        f["fold"] = name
                        f["phase"] = phase
                        f["candidate"] = key
                        f["selected_primary_scenario"] = chosen
                        predictions.append(f)
                    print(f"{name} {key} selection={selected['selection_status']} n={selected['selected_n']}/{selected['budget_n']} hit={selected['touch_rate']:.3f} mean_net={selected['mean_net']:.4f} ES={selected['es95_loss_full_selected']:.4f} seconds={elapsed:.1f}", flush=True)
                except Exception as exc:
                    trial = {"fold": name, "candidate": key, "status": "failed",
                             "error": str(exc), "traceback": traceback.format_exc(),
                             "seconds": time.monotonic() - trial_started}
                    print(f"{name} {key} FAILED {exc}", flush=True)
                fold_trials.append(trial)
                trials.append(trial)
                (output / "trials.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trials))
        passing = [t for t in fold_trials if t["status"] == "completed" and
                   t["selection"]["selection_status"] == "passed_development_scenario"]
        tolerance = config["development_scenario"]["tie_precision_tolerance"]
        if passing:
            top_hit = max(t["selection"]["touch_rate"] for t in passing)
            close = [t for t in passing if t["selection"]["touch_rate"] >= top_hit - tolerance]
            winner = sorted(close, key=lambda t: (-t["selection"]["mean_net"],
                                                   t["selection"]["touch_scores"]["calibrated"]["brier"],
                                                   t["seconds"], 0 if t["candidate"].startswith("lgbm") else 1))[0]
            winner_id = winner["candidate"]
        else:
            winner_id = None
        fold_reports[name] = {"split": split_audit, "risk_training": risk_log,
                              "risk_calibration": risk_adjustment,
                              "selected_candidate": winner_id or "no_candidate_passed",
                              "selection_reasons": {t["candidate"]: t.get("selection", {}).get("failure_reasons", [t.get("error")]) for t in fold_trials},
                              "selection_uses_outer": False}
        print(f"{name} frozen inner winner={fold_reports[name]['selected_candidate']}", flush=True)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_parquet(output / "predictions.parquet", index=False)
    (output / "selection.json").write_text(json.dumps(fold_reports, ensure_ascii=False, indent=2) + "\n")
    verified_market_hashes = source_audit.pop("source_hashes_verified")
    source_audit["source_hashes_verified_count"] = len(verified_market_hashes)
    for rel in (config["calendar"], json.loads(v3_config_path.read_text())["universe"]):
        expected = source_manifest["sources_sha256"].get(rel)
        if not expected or sha256(REPO / rel) != expected:
            raise ValueError(f"v3 frozen metadata source hash mismatch: {rel}")
    source_audit["metadata_hashes_verified"] = {
        rel: sha256(REPO / rel) for rel in (config["calendar"],
                                           json.loads(v3_config_path.read_text())["universe"])}
    manifest = {"protocol": config["protocol_version"], "created_at": datetime.now(UTC).isoformat(),
                "scope": "exposed_development_not_independent", "config_sha256": provenance["files_sha256"][str(config_path.relative_to(REPO))],
                "registration_sha256": provenance["files_sha256"][str((PROJECT / "docs/17_terminal_risk_v4_registration.md").relative_to(REPO))],
                "provenance_at_launch_sha256": sha256(output / "provenance_at_launch.json"),
                "source": source_audit, "folds": fold_reports,
                "model_artifacts_sha256": {p.name: sha256(p) for p in sorted(model_dir.iterdir())},
                "result_artifacts_sha256": {p.name: sha256(p) for p in
                                            (output / "terminal_outcomes.parquet", output / "predictions.parquet",
                                             output / "trials.jsonl", output / "selection.json")},
                "completed_trials": sum(t["status"] == "completed" for t in trials),
                "failed_trials": sum(t["status"] == "failed" for t in trials),
                "seconds": time.monotonic() - started,
                "risk_budget_user_approved": False, "formal_action_enabled": False}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "seconds": manifest["seconds"],
                      "completed": manifest["completed_trials"], "failed": manifest["failed_trials"],
                      "winners": {k: v["selected_candidate"] for k, v in fold_reports.items()}},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/terminal_risk_v4.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())
