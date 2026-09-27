"""Joint final scoring and exact model replay after all three B-group rounds."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .experiment import BASE, DATASET, HERE, ROOT, TARGETS, digest, load, make_x, metrics, project_monotone, reference_predictions


def reliability(frame: pd.DataFrame, key: str):
    y, p = frame[f"y_{key}"].to_numpy(), frame[f"p_{key}"].to_numpy()
    which = np.minimum((p * 10).astype(int), 9)
    buckets = []
    for b in range(10):
        ix = which == b
        buckets.append({"bin": b, "n": int(ix.sum()), "mean_p": float(p[ix].mean()) if ix.any() else None,
                        "actual": float(y[ix].mean()) if ix.any() else None})
    ece = sum(v["n"] * abs(v["mean_p"] - v["actual"]) for v in buckets if v["n"]) / len(p)
    return {"ece10": float(ece), "buckets": buckets,
            "p_quantiles": [float(v) for v in np.quantile(p, [0, .1, .25, .5, .75, .9, 1])]}


def date_block_ci(frame, base, key="1_1", reps=2000, seed=3566):
    dates = sorted(frame.session_date.unique())
    idx = {day: np.flatnonzero(frame.session_date.to_numpy() == day) for day in dates}
    y = frame[f"y_{key}"].to_numpy()
    p = frame[f"p_{key}"].to_numpy()
    q = base[f"p_{key}"].to_numpy()
    daily = np.array([np.mean((y[idx[d]] - p[idx[d]])**2 - (y[idx[d]] - q[idx[d]])**2) for d in dates])
    rng = np.random.default_rng(seed)
    sample = []
    for _ in range(reps):
        picked = []
        while len(picked) < len(dates):
            start = int(rng.integers(len(dates)))
            picked.extend(dates[start:min(len(dates), start + 5)])
        chosen = np.concatenate([idx[d] for d in picked[:len(dates)]])
        sample.append(np.mean((y[chosen] - p[chosen])**2 - (y[chosen] - q[chosen])**2))
    return {"delta": float(np.mean((y-p)**2 - (y-q)**2)),
            "ci95_5date": [float(v) for v in np.quantile(sample, [.025, .975])],
            "bootstrap_replicates": reps,
            "positive_day_count": int((daily > 0).sum()), "dates": len(dates),
            "daily_delta": {str(d): float(v) for d, v in zip(dates, daily)}}


def score(frame):
    return {key: metrics(frame[f"y_{key}"], frame[f"p_{key}"]) for key in TARGETS}


def replay(run, variant, rows, data, folds):
    x = make_x(data, variant)
    expected = json.loads((run / "feature_schema.json").read_text())
    assert x.columns.tolist() == expected
    settings = json.loads((run / "settings.json").read_text())
    assert settings["dataset_sha256"] == digest(DATASET / "manifest.json")
    assert settings["code_sha256"] == digest(run / "experiment_source.py")
    p_raw = np.column_stack([lgb.Booster(model_file=str(run / "models" / f"target_{j}.txt")).predict(x, num_threads=1)
                             for j in range(9)])
    p = project_monotone(p_raw).reshape(-1, 9)
    out = rows[["sample_id", "symbol", "session_date"]].copy()
    for j, key in enumerate(TARGETS):
        out[f"p_{key}"] = p[:, j]
    out.to_parquet(run / "replay_all.parquet", index=False)
    errors = {}
    for split, ids in folds.items():
        saved = pd.read_parquet(run / f"predictions_{split}.parquet")
        assert saved.sample_id.tolist() == out.iloc[ids].sample_id.tolist()
        errors[split] = float(max(np.max(np.abs(saved[f"p_{key}"].to_numpy() - p[ids, j]))
                                  for j, key in enumerate(TARGETS)))
    assert max(errors.values()) < 1e-9
    return errors


def main():
    rows, data, folds = load()
    r0 = pd.read_parquet(BASE / "predictions_B_lgbm_group.parquet")
    frames = {"R0": r0}
    runs = {"R1": ("R1_fixed", "curve"), "R2": ("R2_fixed", "curve_group"),
            "R3": ("R3", "curve")}
    replay_errors = {}
    hashes = {}
    for name, (suffix, variant) in runs.items():
        run = ROOT / f"research/after_open_3d5pct/runs/v7_b_group_{suffix}"
        replay_errors[name] = replay(run, variant, rows, data, folds)
        frames[name] = pd.read_parquet(run / "predictions_outer.parquet")
        hashes[name] = {"settings": digest(run / "settings.json"),
                        "schema": digest(run / "feature_schema.json"),
                        "models": {key: digest(run / "models" / f"target_{j}.txt")
                                   for j, key in enumerate(TARGETS)}}
    expected_ids = r0.sample_id.tolist()
    for name, frame in frames.items():
        assert frame.sample_id.tolist() == expected_ids, name
        for key in TARGETS:
            assert np.array_equal(frame[f"y_{key}"].to_numpy(), r0[f"y_{key}"].to_numpy()), (name, key)
    result = {"status": "exposed_development_only", "dataset_sha256": digest(DATASET / "manifest.json"),
              "r0_prediction_sha256": digest(BASE / "predictions_B_lgbm_group.parquet"),
              "sample_ids_sha256": __import__("hashlib").sha256("\n".join(expected_ids).encode()).hexdigest(),
              "outer_rows": len(r0), "outer_dates": int(r0.session_date.nunique()),
              "hashes": hashes, "replay_max_abs_error": replay_errors,
              "rounds": {}, "paired_primary_vs_R0": {}, "paired_primary_inner_vs_R0": {},
              "paired_targets_outer_vs_R0": {}, "paired_targets_inner_vs_R0": {}}
    base_inner = rows.iloc[folds["inner"]][["sample_id", "session_date"]].copy()
    base_inner_p = reference_predictions(make_x(data, "base"), folds["inner"])
    base_inner_y = data["y"][folds["inner"]].reshape(-1, 9)
    for j, key in enumerate(TARGETS):
        base_inner[f"y_{key}"] = base_inner_y[:, j]
        base_inner[f"p_{key}"] = base_inner_p[:, j]
    for name, frame in frames.items():
        result["rounds"][name] = {"outer": score(frame),
                                  "outer_reliability": {key: reliability(frame, key) for key in TARGETS},
                                  "outer_primary_reliability": reliability(frame, "1_1"),
                                  "outer_primary_by_date": {str(day): metrics(part.y_1_1, part.p_1_1)
                                                            for day, part in frame.groupby("session_date")}}
        if name != "R0":
            run = ROOT / f"research/after_open_3d5pct/runs/v7_b_group_{runs[name][0]}"
            inner = json.loads((run / "inner_result.json").read_text())
            result["rounds"][name]["train_inner"] = inner["targets"]
            result["rounds"][name]["fit_seconds"] = inner["fit_seconds"]
            result["rounds"][name]["best_trees"] = {v["target"]: v["best_trees"] for v in inner["learning"]}
            result["rounds"][name]["inner_primary_reliability"] = reliability(
                pd.read_parquet(run / "predictions_inner.parquet"), "1_1")
            result["paired_primary_vs_R0"][name] = date_block_ci(frame, r0)
            inner_frame = pd.read_parquet(run / "predictions_inner.parquet")
            assert inner_frame.sample_id.tolist() == base_inner.sample_id.tolist()
            result["paired_primary_inner_vs_R0"][name] = date_block_ci(inner_frame, base_inner)
            result["rounds"][name]["inner_reliability"] = {
                key: reliability(inner_frame, key) for key in TARGETS}
            result["paired_targets_outer_vs_R0"][name] = {
                key: date_block_ci(frame, r0, key, reps=500) for key in TARGETS}
            result["paired_targets_inner_vs_R0"][name] = {
                key: date_block_ci(inner_frame, base_inner, key, reps=500) for key in TARGETS}
    diagnosis = json.loads((HERE / "diagnosis.json").read_text())
    result["rounds"]["R0"]["train_inner"] = diagnosis["R0"]
    result["rounds"]["R0"]["inner_reliability"] = {
        key: reliability(base_inner, key) for key in TARGETS}
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name in frames:
        m = result["rounds"][name]["outer"]["1_1"]
        ci = result["paired_primary_vs_R0"].get(name)
        print(name, "Brier", round(m["brier"], 6), "AUC", round(m["auc"], 4),
              "mean_p", round(m["mean_p"], 4), "ECE10", round(result["rounds"][name]["outer_primary_reliability"]["ece10"], 4),
              "delta_CI", ci)


if __name__ == "__main__":
    main()
