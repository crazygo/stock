"""Offline preregistered quality-cohort LightGBM/TCN development training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn

from .hourly_v3_data import Bundle
from .quality_eval_v5 import (action_comparison, choose_quality_recommendations,
                              probability_comparison, selection_status)
from .quality_pool_v5 import FEATURE_COLUMNS, build_quality_features, feature_lineage_hash
from .terminal_risk_v4 import REPO, mixed_expected_net, net_return, sha256
from .train_hourly_v3 import (MultiBranchTCN, _fit_calibrator, _static, _tabular,
                              apply_calibrator)
from .train_terminal_risk_v4 import _brier_metric, fold_indices


PROJECT = Path(__file__).resolve().parent
UTC = timezone.utc
RISK_COLUMNS = ("failure_mean_gross", "failure_q10_gross", "failure_q50_gross",
                "failure_q90_gross", "failure_loss_probability")


def _base_static_names(input_set: str) -> list[str]:
    cols = ["h_range_actual", "h_range_relative", "h_dollar_actual_log", "h_dollar_relative"]
    if "A" in input_set:
        cols += ["a_post_range", "a_post_dollar_actual_log", "a_post_dollar_relative",
                 "a_post_coverage", "a_pre_range", "a_pre_dollar_actual_log",
                 "a_pre_dollar_relative", "a_pre_coverage"]
    if "B" in input_set:
        cols += ["b_range_actual", "b_range_relative", "b_dollar_actual_log", "b_dollar_relative"]
    return cols + ["cutoff_index"]


def _static_v5(bundle: Bundle, quality: pd.DataFrame, input_set: str,
               include_quality: bool) -> tuple[np.ndarray, list[str]]:
    base = _static(bundle, input_set)
    cols = _base_static_names(input_set)
    if base.shape[1] != len(cols):
        raise ValueError("v3 static schema does not match declared v5 base columns")
    if not include_quality:
        return base, cols
    extra = quality.loc[:, FEATURE_COLUMNS].to_numpy(np.float32, copy=True)
    extra = np.nan_to_num(extra, nan=0, posinf=10, neginf=-10)
    # Counts retain their meaning in the saved schema while entering the TCN
    # at comparable scale to probabilities. Missing values are paired with the
    # explicit availability masks in the same ordered static vector.
    extra[:, 1] /= 63
    extra[:, 3] /= 21
    extra[:, 5] /= 42
    return np.column_stack([base, extra]).astype(np.float32), cols + list(FEATURE_COLUMNS)


def _tabular_v5(bundle: Bundle, quality: pd.DataFrame, input_set: str,
                include_quality: bool) -> pd.DataFrame:
    base = _tabular(bundle, input_set)
    if any(c.startswith("h_quality_") for c in base):
        raise ValueError("quality columns leaked into pooled control base features")
    if include_quality:
        for col in FEATURE_COLUMNS:
            if col in base:
                raise ValueError("duplicate quality model feature")
            base[col] = quality[col].to_numpy()
    if include_quality != any(c in base for c in FEATURE_COLUMNS):
        raise ValueError("v5 quality feature schema mismatch")
    return base


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, dtype=float) - np.asarray(p, dtype=float)) ** 2))


def _training_indices(indices: dict[str, np.ndarray], quality_mask: np.ndarray,
                      scope: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_quality = indices["train"][quality_mask[indices["train"]]]
    early_quality = indices["early"][quality_mask[indices["early"]]]
    if scope == "pooled":
        train = indices["train"]
    elif scope == "decision_time_quality_rows_only":
        train = train_quality
    else:
        raise ValueError(f"unknown v5 training scope: {scope}")
    return train, train_quality, early_quality


def _fit_lgbm(bundle: Bundle, quality: pd.DataFrame, y: np.ndarray,
              indices: dict[str, np.ndarray], candidate: dict,
              v3_config: dict, config: dict, model_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    x = _tabular_v5(bundle, quality, candidate["input_set"], candidate["quality_features"])
    qmask = quality.quality_eligible.to_numpy(bool)
    train, train_quality, early_quality = _training_indices(indices, qmask, candidate["training_scope"])
    if min(len(train), len(train_quality), len(early_quality)) < 100:
        raise ValueError("insufficient quality training or early-stopping rows")
    params = dict(v3_config["lightgbm"], metric="None", random_state=config["seed"])
    if candidate.get("regularized"):
        params.update(config["round2_if_inner_diagnostic"]["lgbm_changes"])
    model = lgb.LGBMClassifier(**params)
    curve = {}
    started = time.monotonic()
    model.fit(x.iloc[train], y[train], eval_set=[(x.iloc[early_quality], y[early_quality])],
              eval_metric=_brier_metric,
              callbacks=[lgb.early_stopping(20, first_metric_only=True, verbose=False),
                         lgb.record_evaluation(curve)])
    model.booster_.save_model(str(model_path))
    if x.columns.tolist() != model.booster_.feature_name():
        raise ValueError("saved LightGBM feature order mismatch")
    prediction = {name: model.predict_proba(x.iloc[ix])[:, 1]
                  for name, ix in indices.items() if name in ("calibration", "selection", "outer")}
    details = {"seconds": time.monotonic() - started, "train_rows": len(train),
               "train_quality_rows": len(train_quality), "early_quality_rows": len(early_quality),
               "best_iteration": int(model.best_iteration_), "early_curve": curve,
               "train_quality_brier": _brier(y[train_quality], model.predict_proba(x.iloc[train_quality])[:, 1]),
               "early_quality_brier": _brier(y[early_quality], model.predict_proba(x.iloc[early_quality])[:, 1]),
               "feature_columns": x.columns.tolist(), "quality_features_enabled": candidate["quality_features"],
               "training_scope": candidate["training_scope"]}
    return prediction, details


def _fit_tcn(bundle: Bundle, quality: pd.DataFrame, y: np.ndarray,
             indices: dict[str, np.ndarray], candidate: dict,
             v3_config: dict, config: dict, model_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    qmask = quality.quality_eligible.to_numpy(bool)
    train, train_quality, early_quality = _training_indices(indices, qmask, candidate["training_scope"])
    if min(len(train), len(train_quality), len(early_quality)) < 100:
        raise ValueError("insufficient quality training or early-stopping rows")
    static, columns = _static_v5(bundle, quality, candidate["input_set"], candidate["quality_features"])
    model_cfg = dict(v3_config["tcn"])
    if candidate.get("regularized"):
        changes = config["round2_if_inner_diagnostic"]["tcn_changes"]
        model_cfg["dropout"] = changes["dropout"]
        model_cfg["weight_decay"] *= changes["weight_decay_multiplier"]
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiBranchTCN(model_cfg, candidate["input_set"], static.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg["learning_rate"],
                                  weight_decay=model_cfg["weight_decay"])
    arrays = [bundle.h_seq, bundle.post_seq, bundle.pre_seq, bundle.b_seq]

    def batch(ix: np.ndarray):
        tensors = [torch.from_numpy(np.clip(a[ix], -50, 50)).to(device) for a in arrays]
        return (*tensors, torch.from_numpy(static[ix]).to(device))

    def predict(ix: np.ndarray) -> np.ndarray:
        model.eval()
        chunks = np.array_split(ix, max(1, math.ceil(len(ix) / 512)))
        with torch.no_grad():
            parts = [torch.sigmoid(model(*batch(part))).cpu().numpy() for part in chunks if len(part)]
        return np.concatenate(parts)

    best, best_epoch, best_state, bad = float("inf"), None, None, 0
    epochs = []
    rng = np.random.default_rng(config["seed"])
    started = time.monotonic()
    for epoch in range(model_cfg["max_epochs"]):
        epoch_start = time.monotonic()
        model.train()
        loss_values = []
        shuffled = rng.permutation(train)
        for ix in np.array_split(shuffled, max(1, math.ceil(len(shuffled) / model_cfg["batch_size"]))):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch(ix))
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.from_numpy(y[ix].astype(np.float32)).to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_values.append(float(loss.item()))
        early_brier = _brier(y[early_quality], predict(early_quality))
        train_quality_brier = _brier(y[train_quality], predict(train_quality))
        epochs.append({"epoch": epoch + 1, "train_BCE": float(np.mean(loss_values)),
                       "train_quality_brier": train_quality_brier,
                       "early_quality_brier": early_brier,
                       "epoch_seconds": time.monotonic() - epoch_start,
                       "cumulative_seconds": time.monotonic() - started})
        if early_brier < best - 1e-5:
            best, best_epoch, best_state, bad = early_brier, epoch + 1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= model_cfg["patience"]:
                break
    if best_state is None:
        raise ValueError("TCN has no usable quality early-stopping checkpoint")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "input_set": candidate["input_set"],
                "static_columns": columns, "static_dim": static.shape[1],
                "quality_features_enabled": candidate["quality_features"],
                "quality_count_scaling": config["tcn_quality_count_scaling"] if candidate["quality_features"] else {},
                "config": model_cfg}, model_path)
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    if saved["static_columns"] != columns or saved["static_dim"] != static.shape[1]:
        raise ValueError("saved TCN feature schema mismatch")
    prediction = {name: predict(ix) for name, ix in indices.items()
                  if name in ("calibration", "selection", "outer")}
    best_log = epochs[best_epoch - 1]
    details = {"seconds": time.monotonic() - started, "train_rows": len(train),
               "train_quality_rows": len(train_quality), "early_quality_rows": len(early_quality),
               "best_epoch": best_epoch, "epochs": epochs,
               "train_quality_brier": best_log["train_quality_brier"],
               "early_quality_brier": best_log["early_quality_brier"],
               "static_columns": columns, "static_dim": static.shape[1],
               "quality_features_enabled": candidate["quality_features"],
               "training_scope": candidate["training_scope"],
               "parameters": sum(p.numel() for p in model.parameters()), "device": str(device)}
    return prediction, details


def _risk_for_phase(v4_predictions: pd.DataFrame, fold: str, phase: str,
                    sample_ids: pd.Series) -> pd.DataFrame:
    source = v4_predictions[(v4_predictions.fold == fold) &
                            (v4_predictions.phase == phase) &
                            (v4_predictions.candidate == "lgbm_HAB")]
    if source.sample_id.duplicated().any() or len(source) != len(sample_ids):
        raise ValueError("v4 fold/phase shared risk rows missing or duplicated")
    risk = source.set_index("sample_id").reindex(sample_ids)
    if risk[list(RISK_COLUMNS)].isna().any().any():
        raise ValueError("v4 shared risk head missing prediction")
    return risk.reset_index(drop=True)


def _prediction_frame(rows: pd.DataFrame, outcome: pd.DataFrame,
                      quality: pd.DataFrame, indices: np.ndarray,
                      raw: np.ndarray, calibrated: np.ndarray,
                      history_calibrated: np.ndarray, risk: pd.DataFrame,
                      config: dict, fold: str, phase: str, candidate: str,
                      kind: str) -> pd.DataFrame:
    f = rows.iloc[indices][["sample_id", "symbol", "session_date", "cutoff_et",
                            "b_range_actual", "b_dollar_actual_log"]].reset_index(drop=True).copy()
    result = outcome.iloc[indices]
    f["target"] = result.target.to_numpy(int)
    f["realized_net"] = result.net_evaluation_return.to_numpy(float)
    f["p_raw"] = raw
    f["p_touch"] = calibrated
    f["history_calibrated_p"] = history_calibrated
    for col in (*FEATURE_COLUMNS, "quality_eligible", "quality_status"):
        f[col] = quality.iloc[indices][col].to_numpy()
    for col in RISK_COLUMNS:
        f[col] = risk[col].to_numpy(float)
    f["expected_net"] = mixed_expected_net(f.p_touch.to_numpy(float),
                                             f.failure_mean_gross.to_numpy(float),
                                             touch_gross=config["touch_gross_return"],
                                             buy_cost=config["buy_cost_rate"],
                                             sell_cost=config["sell_cost_rate"])
    f["failure_q10_net"] = net_return(f.failure_q10_gross.to_numpy(float),
                                       config["buy_cost_rate"], config["sell_cost_rate"])
    f["fold"] = fold
    f["phase"] = phase
    f["candidate"] = candidate
    f["candidate_kind"] = kind
    f["selected_quality_scenario"] = choose_quality_recommendations(f, config["development_scenario"])
    return f


def _score_inner(frame: pd.DataFrame, config: dict, date_axis: list[str]) -> dict:
    prob = probability_comparison(frame, config, date_axis)
    action = action_comparison(frame, frame.selected_quality_scenario.to_numpy(bool), config, date_axis)
    return {"probability": prob, "action": action,
            "status": selection_status(prob, action, config)}


def _date_axis(calendar: list[dict], fold: dict, phase: str) -> list[str]:
    start = fold["selection_start"] if phase == "selection" else fold["outer_start"]
    stop = fold["outer_start"] if phase == "selection" else fold["outer_end_exclusive"]
    return [s["session_date"] for s in calendar if start <= s["session_date"] < stop]


def _round2_candidate(family: str) -> dict:
    return {"name": f"{family}_HAB_quality_only_regularized", "family": family,
            "input_set": "HAB", "training_scope": "decision_time_quality_rows_only",
            "quality_features": True, "regularized": True}


def _verify_sources(config: dict) -> tuple[Path, Path, dict, dict]:
    v3 = REPO / config["source_v3_run"]
    v4 = REPO / config["source_v4_run"]
    m4 = json.loads((v4 / "manifest.json").read_text())
    registered_v4_config = PROJECT / "configs/terminal_risk_v4.json"
    if (sha256(registered_v4_config) != m4["config_sha256"] or
            json.loads((v4 / "config.json").read_text()) != json.loads(registered_v4_config.read_text())):
        raise ValueError("frozen v4 config hash mismatch")
    p4 = json.loads((v4 / "provenance_at_launch.json").read_text())
    if sha256(v4 / "provenance_at_launch.json") != m4["provenance_at_launch_sha256"]:
        raise ValueError("frozen v4 launch provenance hash mismatch")
    if sha256(v3 / "config.json") != p4["v3_actual_config_sha256"]:
        raise ValueError("frozen v3 actual config hash mismatch")
    if sha256(v3 / "manifest.json") != m4["source"]["source_manifest_sha256"]:
        raise ValueError("frozen v3 source manifest hash mismatch")
    expected_calendar = m4["source"]["metadata_hashes_verified"][config["calendar"]]
    if sha256(REPO / config["calendar"]) != expected_calendar:
        raise ValueError("frozen calendar hash mismatch")
    for name, expected in m4["result_artifacts_sha256"].items():
        if sha256(v4 / name) != expected:
            raise ValueError(f"frozen v4 result changed: {name}")
    for name, expected in m4["model_artifacts_sha256"].items():
        if sha256(v4 / "models" / name) != expected:
            raise ValueError(f"frozen v4 model changed: {name}")
    c3 = json.loads((v3 / "config.json").read_text())
    c4 = json.loads((v4 / "config.json").read_text())
    if c4["source_run"] != config["source_v3_run"] or c4["calendar"] != config["calendar"]:
        raise ValueError("v5 registered source lineage differs from frozen v4")
    m3 = json.loads((v3 / "manifest.json").read_text())
    for name in ("features.parquet", "outcomes.parquet", "sequences.npz"):
        expected = m3["result_artifacts_sha256"][name]
        if sha256(v3 / name) != expected:
            raise ValueError(f"frozen v3 source changed: {name}")
    return v3, v4, c3, m4


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    if config["protocol_version"] != "causal_quality_pool_touch_v5":
        raise ValueError("unsupported v5 quality protocol")
    if tuple(config["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("v5 registered quality feature order mismatch")
    if config["tcn_quality_count_scaling"] != {
            "h_quality_mature_days_63": 63,
            "h_quality_recent_days_21": 21,
            "h_quality_prior_days_42": 42}:
        raise ValueError("v5 TCN quality scaling differs from registered code")
    v3, v4, v3_config, v4_manifest = _verify_sources(config)
    calendar_path = REPO / config["calendar"]
    calendar = json.loads(calendar_path.read_text())["sessions"]
    started = time.monotonic()
    output.mkdir(parents=True)
    models = output / "models"
    models.mkdir()
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    provenance_files = [config_path, PROJECT / "docs/19_quality_training_v5_registration.md",
                        PROJECT / "quality_pool_v5.py", PROJECT / "quality_eval_v5.py",
                        PROJECT / "train_quality_v5.py", PROJECT / "train_hourly_v3.py",
                        PROJECT / "train_terminal_risk_v4.py", calendar_path,
                        v3 / "features.parquet", v3 / "outcomes.parquet", v3 / "sequences.npz",
                        v3 / "config.json", v3 / "manifest.json",
                        v4 / "terminal_outcomes.parquet", v4 / "predictions.parquet",
                        v4 / "config.json", v4 / "manifest.json"]
    source_hashes = {str(p.relative_to(REPO)): sha256(p) for p in provenance_files}
    provenance = {"captured_at_utc": datetime.now(UTC).isoformat(),
                  "timing": "before_quality_materialization_and_before_any_v5_model_fit",
                  "files_sha256": source_hashes,
                  "versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "lightgbm": lgb.__version__,
                               "sklearn": sklearn.__version__, "torch": torch.__version__}}
    (output / "provenance_at_launch.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    features = pd.read_parquet(v3 / "features.parquet")
    old_outcome = pd.read_parquet(v3 / "outcomes.parquet")
    terminal = pd.read_parquet(v4 / "terminal_outcomes.parquet")
    if features.sample_id.duplicated().any() or terminal.sample_id.duplicated().any():
        raise ValueError("frozen source sample IDs are not unique")
    if not (features.sample_id.to_numpy() == terminal.sample_id.to_numpy()).all():
        raise ValueError("frozen feature/terminal rows differ")
    if (terminal.status != "mature").any():
        raise ValueError("frozen v4 terminal source includes nonmature sample")
    rows = features.merge(old_outcome[["sample_id", "label_end_at", "label_available_at", "target"]],
                          on="sample_id", validate="one_to_one")
    if not (rows.target.to_numpy(int) == terminal.target.to_numpy(int)).all():
        raise ValueError("v3/v4 touch label mismatch")
    quality = build_quality_features(rows[["sample_id", "symbol", "session_date", "decision_at"]],
                                     terminal, [s["session_date"] for s in calendar], config["quality"])
    if not (quality.sample_id.to_numpy() == rows.sample_id.to_numpy()).all():
        raise ValueError("quality feature row order mismatch")
    quality.to_parquet(output / "quality_features.parquet", index=False)
    quality_source_hashes = {k: v for k, v in source_hashes.items()
                             if k.endswith(("terminal_outcomes.parquet", "features.parquet")) or k == config["calendar"]}
    quality_lineage = feature_lineage_hash(quality, quality_source_hashes)
    print(f"quality features {len(quality)} rows, eligible {int(quality.quality_eligible.sum())}, lineage {quality_lineage}", flush=True)
    with np.load(v3 / "sequences.npz") as seq:
        bundle = Bundle(rows=rows, h_seq=seq["h"], post_seq=seq["post"],
                        pre_seq=seq["pre"], b_seq=seq["b"], exclusions={}, coverage={}, source_paths=[])
    v4_pred = pd.read_parquet(v4 / "predictions.parquet")
    y = rows.target.to_numpy(int)
    trials, all_frames, fold_reports = [], [], {}
    for fold in config["folds"]:
        fold_name = fold["name"]
        indices, split_audit = fold_indices(rows, fold)
        cal_quality = indices["calibration"][quality.quality_eligible.to_numpy(bool)[indices["calibration"]]]
        if len(cal_quality) < 100:
            raise ValueError(f"{fold_name} insufficient quality calibration rows")
        history_calibrator = _fit_calibrator(y[cal_quality],
                                              quality.iloc[cal_quality].h_quality_rate_63.to_numpy(float))
        history_cal_path = models / f"{fold_name}_history_only.calibration.json"
        history_cal_path.write_text(json.dumps({"calibrator": history_calibrator,
                                                "quality_calibration_n": len(cal_quality)}, indent=2) + "\n")
        history_p = {phase: apply_calibrator(
            quality.iloc[indices[phase]].h_quality_rate_63.fillna(.5).to_numpy(float), history_calibrator)
            for phase in ("selection", "outer")}
        risk = {phase: _risk_for_phase(v4_pred, fold_name, phase,
                                      rows.iloc[indices[phase]].sample_id)
                for phase in ("selection", "outer")}
        phase_axis = {phase: _date_axis(calendar, fold, phase) for phase in ("selection", "outer")}
        fold_trials = {}

        def complete_trial(candidate: dict, round_number: int):
            key = candidate["name"]
            trial_start = time.monotonic()
            try:
                path = models / f"{fold_name}_{key}.{'txt' if candidate['family'] == 'lgbm' else 'pt'}"
                if candidate["family"] == "lgbm":
                    raw, training = _fit_lgbm(bundle, quality, y, indices, candidate,
                                               v3_config, config, path)
                else:
                    raw, training = _fit_tcn(bundle, quality, y, indices, candidate,
                                              v3_config, config, path)
                cal = _fit_calibrator(y[cal_quality], raw["calibration"][
                    quality.quality_eligible.to_numpy(bool)[indices["calibration"]]])
                cal_path = models / f"{fold_name}_{key}.calibration.json"
                cal_path.write_text(json.dumps({"calibrator": cal, "calibration_n": len(cal_quality),
                                                "quality_only": True}, indent=2) + "\n")
                phase_frames = {}
                for phase in ("selection", "outer"):
                    ix = indices[phase]
                    frame = _prediction_frame(rows, terminal, quality, ix, raw[phase],
                                              apply_calibrator(raw[phase], cal), history_p[phase],
                                              risk[phase], config, fold_name, phase, key, "new_v5")
                    phase_frames[phase] = frame
                selection = _score_inner(phase_frames["selection"], config, phase_axis["selection"])
                all_frames.extend(phase_frames.values())
                trial = {"fold": fold_name, "candidate": key, "round": round_number,
                         "status": "completed", "seconds": time.monotonic() - trial_start,
                         "training": training, "calibration": cal,
                         "selection": selection,
                         "checkpoint": path.name, "calibration_file": cal_path.name,
                         "whole_pipeline_information_range": "HAB_v4_shared_risk_head"}
                print(f"{fold_name} {key}: early_quality_brier={training['early_quality_brier']:.4f} "
                      f"selection_quality_brier={selection['probability']['model']['brier']:.4f} "
                      f"recommendations={selection['action']['selected']['n']} "
                      f"prob={selection['status']['probability_status']} "
                      f"action={selection['status']['action_status']}", flush=True)
            except Exception as exc:
                trial = {"fold": fold_name, "candidate": key, "round": round_number,
                         "status": "failed", "seconds": time.monotonic() - trial_start,
                         "error": str(exc), "traceback": traceback.format_exc()}
                print(f"{fold_name} {key}: FAILED {exc}", flush=True)
            fold_trials[key] = trial
            trials.append(trial)
            (output / "trials.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trials))

        for candidate in config["round1_new_candidates"]:
            complete_trial(candidate, 1)
        round2 = {}
        for family in config["families"]:
            only_key = f"{family}_HAB_quality_only"
            pooled_key = f"{family}_HAB_quality_aware"
            only, pooled = fold_trials[only_key], fold_trials[pooled_key]
            if only["status"] != "completed" or pooled["status"] != "completed":
                round2[family] = {"trigger": False, "reason": "comparison_trial_failed"}
                continue
            early_gap = only["training"]["early_quality_brier"] - pooled["training"]["early_quality_brier"]
            train_gap = only["training"]["early_quality_brier"] - only["training"]["train_quality_brier"]
            trigger = early_gap >= .02 or train_gap > .05
            round2[family] = {"trigger": trigger, "same_quality_early_brier_gap_vs_pooled": early_gap,
                              "same_quality_train_to_early_gap": train_gap,
                              "reason": "pre_registered_inner_overfit_diagnostic" if trigger else "not_triggered"}
            if trigger:
                complete_trial(_round2_candidate(family), 2)
        reference = {}
        for ref in config["frozen_v4_references"]:
            ref_frames = {}
            for phase in ("selection", "outer"):
                ix = indices[phase]
                source = v4_pred[(v4_pred.fold == fold_name) & (v4_pred.phase == phase) &
                                 (v4_pred.candidate == ref)].set_index("sample_id").reindex(rows.iloc[ix].sample_id)
                if source.p_touch.isna().any():
                    raise ValueError(f"v4 frozen reference {ref} lacks same-sample score")
                f = _prediction_frame(rows, terminal, quality, ix, source.p_raw.to_numpy(float),
                                      source.p_touch.to_numpy(float), history_p[phase], risk[phase],
                                      config, fold_name, phase, f"v4_{ref}", "frozen_v4_reference")
                ref_frames[phase] = f
            reference[ref] = _score_inner(ref_frames["selection"], config, phase_axis["selection"])
            all_frames.extend(ref_frames.values())
        completed = [t for t in fold_trials.values() if t["status"] == "completed"]
        best_probability = min(completed, key=lambda t: (t["selection"]["probability"]["model"]["brier"],
                                                         t["seconds"], t["candidate"])) if completed else None
        action_pass = [t for t in completed if
                       t["selection"]["status"]["probability_status"] == "passed_probability_development_target" and
                       t["selection"]["status"]["action_status"] == "passed_action_development_scenario"]
        best_action = min(action_pass, key=lambda t: (t["selection"]["probability"]["model"]["brier"],
                                                       -t["selection"]["action"]["touch_lift"],
                                                       -t["selection"]["action"]["selected"]["mean_net"],
                                                       t["seconds"], t["candidate"])) if action_pass else None
        fold_reports[fold_name] = {"split": split_audit, "quality_counts": {
            phase: {"rows": int(quality.iloc[ix].quality_eligible.sum()),
                    "stock_dates": int(rows.iloc[ix][quality.iloc[ix].quality_eligible.to_numpy()][
                        ["symbol", "session_date"]].drop_duplicates().shape[0]),
                    "ET_dates": int(rows.iloc[ix][quality.iloc[ix].quality_eligible.to_numpy()].session_date.nunique())}
            for phase, ix in indices.items()},
            "history_only_calibrator": history_calibrator,
            "round2": round2, "frozen_v4_reference_selection": reference,
            "best_inner_probability_candidate": best_probability["candidate"] if best_probability else None,
            "best_inner_probability_status": best_probability["selection"]["status"]["probability_status"] if best_probability else "no_completed_trial",
            "best_inner_action_candidate": best_action["candidate"] if best_action else None,
            "formal_action_enabled": False, "outer_used_for_selection": False}
        print(f"{fold_name}: probability={fold_reports[fold_name]['best_inner_probability_candidate']} "
              f"action={fold_reports[fold_name]['best_inner_action_candidate']} round2={round2}", flush=True)
    pd.concat(all_frames, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    (output / "selection.json").write_text(json.dumps(fold_reports, ensure_ascii=False, indent=2) + "\n")
    manifest = {"protocol": config["protocol_version"], "created_at": datetime.now(UTC).isoformat(),
                "scope": "exposed_development_not_independent", "config_sha256": sha256(config_path),
                "registration_sha256": source_hashes[str((PROJECT / "docs/19_quality_training_v5_registration.md").relative_to(REPO))],
                "provenance_at_launch_sha256": sha256(output / "provenance_at_launch.json"),
                "v4_manifest_sha256": sha256(v4 / "manifest.json"),
                "quality_lineage_sha256": quality_lineage,
                "quality_definition": config["quality"], "folds": fold_reports,
                "model_artifacts_sha256": {p.name: sha256(p) for p in sorted(models.iterdir())},
                "result_artifacts_sha256": {p.name: sha256(p) for p in
                                            (output / "quality_features.parquet", output / "predictions.parquet",
                                             output / "selection.json", output / "trials.jsonl")},
                "completed_trials": sum(t["status"] == "completed" for t in trials),
                "failed_trials": sum(t["status"] == "failed" for t in trials),
                "seconds": time.monotonic() - started,
                "risk_budget_user_approved": False, "formal_action_enabled": False}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "completed_trials": manifest["completed_trials"],
                      "failed_trials": manifest["failed_trials"], "seconds": manifest["seconds"],
                      "folds": {k: {"probability": v["best_inner_probability_candidate"],
                                     "action": v["best_inner_action_candidate"]}
                                for k, v in fold_reports.items()}}, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/quality_training_v5.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())
