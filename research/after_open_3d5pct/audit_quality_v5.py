"""Independent artifact, schema, as-of feature and selection audit for v5."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from .evaluate_quality_v5 import _verify
from .quality_pool_v5 import FEATURE_COLUMNS, build_quality_features
from .terminal_risk_v4 import REPO, sha256
from .train_terminal_risk_v4 import fold_indices


def audit(run: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config, manifest = _verify(run)
    v3, v4 = REPO / config["source_v3_run"], REPO / config["source_v4_run"]
    rows = pd.read_parquet(v3 / "features.parquet").merge(
        pd.read_parquet(v3 / "outcomes.parquet")[["sample_id", "label_end_at", "label_available_at", "target"]],
        on="sample_id", validate="one_to_one")
    terminal = pd.read_parquet(v4 / "terminal_outcomes.parquet")
    sessions = [s["session_date"] for s in
                json.loads((REPO / config["calendar"]).read_text())["sessions"]]
    fresh = build_quality_features(rows[["sample_id", "symbol", "session_date", "decision_at"]],
                                   terminal, sessions, config["quality"])
    saved = pd.read_parquet(run / "quality_features.parquet")
    pd.testing.assert_frame_equal(fresh, saved, check_exact=True)
    trials = [json.loads(line) for line in (run / "trials.jsonl").read_text().splitlines()]
    predictions = pd.read_parquet(run / "predictions.parquet")
    if predictions.duplicated(["sample_id", "fold", "phase", "candidate"]).any():
        raise ValueError("duplicate v5 prediction keys")
    checked = 0
    by_fold = {}
    for fold in config["folds"]:
        name = fold["name"]
        ix, counts = fold_indices(rows, fold)
        quality_mask = saved.quality_eligible.to_numpy(bool)
        relevant = [t for t in trials if t["fold"] == name]
        by_name = {t["candidate"]: t for t in relevant}
        if len(by_name) != len(relevant):
            raise ValueError("duplicate v5 trial names")
        for t in relevant:
            if t["status"] != "completed":
                if len(predictions[(predictions.fold == name) &
                                   (predictions.candidate == t["candidate"])]):
                    raise ValueError("failed trial leaked into prediction table")
                continue
            checked += 1
            key, train = t["candidate"], t["training"]
            checkpoint = run / "models" / t["checkpoint"]
            enabled = train["quality_features_enabled"]
            if key.startswith("lgbm"):
                model = lgb.Booster(model_file=str(checkpoint))
                columns = train["feature_columns"]
                if model.feature_name() != columns:
                    raise ValueError("LightGBM checkpoint column order mismatch")
            else:
                saved_model = torch.load(checkpoint, map_location="cpu", weights_only=False)
                columns = saved_model["static_columns"]
                if columns != train["static_columns"] or saved_model["static_dim"] != len(columns):
                    raise ValueError("TCN checkpoint static schema mismatch")
                if saved_model["quality_features_enabled"] != enabled:
                    raise ValueError("TCN checkpoint quality flag mismatch")
                if enabled and saved_model["quality_count_scaling"] != config["tcn_quality_count_scaling"]:
                    raise ValueError("TCN checkpoint scaling mismatch")
                branches = 1 + (2 if "A" in saved_model["input_set"] else 0) + (
                    1 if "B" in saved_model["input_set"] else 0)
                expected_input = branches * saved_model["config"]["channels"] * 2 + len(columns)
                actual_input = saved_model["state_dict"]["output.0.weight"].shape[1]
                if actual_input != expected_input:
                    raise ValueError("TCN checkpoint fusion input dimension mismatch")
            present = [c for c in columns if c in FEATURE_COLUMNS]
            if present != (list(FEATURE_COLUMNS) if enabled else []):
                raise ValueError("v5 quality feature allowlist or order mismatch")
            if train["early_quality_rows"] != int(quality_mask[ix["early"]].sum()):
                raise ValueError("different early-stop cohort between trials")
            expected_train = int(quality_mask[ix["train"]].sum()) if (
                train["training_scope"] == "decision_time_quality_rows_only") else len(ix["train"])
            if train["train_rows"] != expected_train:
                raise ValueError("quality-only training did not use each row's historical identity")
            for phase in ("selection", "outer"):
                f = predictions[(predictions.fold == name) & (predictions.phase == phase) &
                                (predictions.candidate == key)]
                if len(f) != len(ix[phase]) or not np.array_equal(
                        f.sample_id.to_numpy(), rows.iloc[ix[phase]].sample_id.to_numpy()):
                    raise ValueError("completed trial lacks exact same-sample phase predictions")
                if f.loc[f.selected_quality_scenario, "quality_eligible"].eq(False).any():
                    raise ValueError("nonquality row recommended")
        for family in config["families"]:
            only = by_name[f"{family}_HAB_quality_only"]
            pooled = by_name[f"{family}_HAB_quality_aware"]
            observed = manifest["folds"][name]["round2"][family]
            if only["status"] == pooled["status"] == "completed":
                early_gap = only["training"]["early_quality_brier"] - pooled["training"]["early_quality_brier"]
                train_gap = only["training"]["early_quality_brier"] - only["training"]["train_quality_brier"]
                if observed["trigger"] != (early_gap >= .02 or train_gap > .05):
                    raise ValueError("round2 trigger differs from preregistered inner diagnostic")
            regularized = by_name.get(f"{family}_HAB_quality_only_regularized")
            if bool(regularized) != observed["trigger"]:
                raise ValueError("round2 trial does not match trigger")
        by_fold[name] = {"split": counts, "quality_counts": manifest["folds"][name]["quality_counts"],
                         "round2": manifest["folds"][name]["round2"],
                         "best_inner_probability_candidate": manifest["folds"][name]["best_inner_probability_candidate"],
                         "best_inner_action_candidate": manifest["folds"][name]["best_inner_action_candidate"]}
    result = {"status": "integrity_passed_not_effectiveness_release",
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "run": str(run), "manifest_sha256": sha256(run / "manifest.json"),
              "quality_rows_exactly_rederived": len(saved),
              "quality_eligible_rows": int(saved.quality_eligible.sum()),
              "completed_checkpoint_schemas_verified": checked,
              "model_artifacts_hash_verified": len(manifest["model_artifacts_sha256"]),
              "result_artifacts_hash_verified": len(manifest["result_artifacts_sha256"]),
              "failed_trials": manifest["failed_trials"],
              "folds": by_fold}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run.resolve(), args.output.resolve())
    print(json.dumps({k: result[k] for k in ("status", "quality_rows_exactly_rederived",
                                              "completed_checkpoint_schemas_verified")},
                     ensure_ascii=False))
