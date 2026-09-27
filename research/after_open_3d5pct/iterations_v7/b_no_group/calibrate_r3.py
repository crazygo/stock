"""Temporal intercept calibration of the frozen R2 no-group tree ensemble."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from research.after_open_3d5pct.iterations_v7.b_no_group.runner import (
    BASE, EXPECTED_MANIFEST, ROOT, frame, load, matrix, metrics, sha, summarize,
)
from research.after_open_3d5pct.train_multiscale_v6 import project_monotone


def offset(p: np.ndarray, y: np.ndarray) -> float:
    """Calibration-in-the-large with 50 prior observations at the predicted rate."""
    mean = float(np.mean(p))
    target = float((np.sum(y) + 50 * mean) / (len(y) + 50))
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    lo, hi = -10., 10.
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.mean(expit(z + mid)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def transform(p: np.ndarray, offsets: list[float]) -> np.ndarray:
    q = expit(logit(np.clip(p.reshape(-1, 9), 1e-6, 1-1e-6)) + np.asarray(offsets))
    return project_monotone(q).reshape(-1, 3, 3)


def replay(data: dict, folds: dict, out: Path) -> float:
    x = matrix(data, "r2")
    assert list(x.columns) == json.loads((out / "columns.json").read_text())
    raw = np.column_stack([lgb.Booster(model_file=str(out / f"models/target_{j}.txt")).predict(
        x.iloc[folds["outer"]], num_threads=1) for j in range(9)])
    r2 = project_monotone(raw).reshape(-1, 3, 3)
    q = transform(r2, json.loads((out / "calibration.json").read_text())["final_offsets"])
    saved = pd.read_parquet(out / "outer_blind.parquet")
    err = max(float(np.max(np.abs(q[:, h, t] - saved[f"p_{h}_{t}"].to_numpy())))
              for h in range(3) for t in range(3))
    assert err < 1e-7
    return err


def main():
    table, data, folds = load()
    r2_dir = BASE / "runs/v7_b_no_group_r2"
    out = BASE / "runs/v7_b_no_group_r3"
    out.mkdir(exist_ok=True)
    models = out / "models"
    models.mkdir(exist_ok=True)
    for j in range(9):
        shutil.copy2(r2_dir / f"models/target_{j}.txt", models / f"target_{j}.txt")
    shutil.copy2(r2_dir / "columns.json", out / "columns.json")
    inner = pd.read_parquet(r2_dir / "inner.parquet")
    outer = pd.read_parquet(r2_dir / "outer_blind.parquet")
    assert inner.sample_id.tolist() == table.iloc[folds["inner"]].sample_id.tolist()
    assert outer.sample_id.tolist() == table.iloc[folds["outer"]].sample_id.tolist()
    pi = np.stack([np.column_stack([inner[f"p_{h}_{t}"].to_numpy() for t in range(3)])
                   for h in range(3)], axis=1)
    po = np.stack([np.column_stack([outer[f"p_{h}_{t}"].to_numpy() for t in range(3)])
                   for h in range(3)], axis=1)
    yi = data["y"][folds["inner"]]
    dates = inner.session_date.to_numpy()
    early = dates <= "2026-08-08"
    late = dates >= "2026-08-18"
    assert int(early.sum()) == 476 and int(late.sum()) == 475
    assert pd.to_datetime(table.iloc[folds["inner"][early]].label_available_at, utc=True).max() < pd.Timestamp("2026-08-18", tz="America/New_York").tz_convert("UTC")
    early_offsets = [offset(pi[early, h, t], yi[early, h, t]) for h in range(3) for t in range(3)]
    final_offsets = [offset(pi[:, h, t], yi[:, h, t]) for h in range(3) for t in range(3)]
    late_cal = transform(pi[late], early_offsets)
    full_cal = transform(pi, final_offsets)
    outer_cal = transform(po, final_offsets)
    frame(table, folds["inner"], full_cal, yi).to_parquet(out / "inner.parquet", index=False)
    frame(table, folds["outer"], outer_cal, None).to_parquet(out / "outer_blind.parquet", index=False)
    frame(table, folds["inner"][late], late_cal, yi[late]).to_parquet(out / "inner_late_temporal.parquet", index=False)
    artifact = {"method": "shrunken_logit_intercept_50", "early_fit_dates": ["2026-08-03", "2026-08-08"],
                "late_evaluation_dates": ["2026-08-18", "2026-08-24"],
                "final_fit_dates": ["2026-08-03", "2026-08-24"],
                "early_offsets": early_offsets, "final_offsets": final_offsets,
                "source_r2_models_sha256": [sha(r2_dir / f"models/target_{j}.txt") for j in range(9)]}
    (out / "calibration.json").write_text(json.dumps(artifact, indent=2) + "\n")
    report = {"round": "r3", "early_rows": int(early.sum()), "late_rows": int(late.sum()),
              "late_raw": summarize(yi[late], pi[late]),
              "late_calibrated": summarize(yi[late], late_cal),
              "inner_full_calibrated_descriptive_in_sample": summarize(yi, full_cal),
              "replay_max_abs_error": replay(data, folds, out),
              "lineage": {"dataset_manifest_sha256": EXPECTED_MANIFEST,
                          "calibration_code_sha256": sha(Path(__file__)),
                          "r2_diagnostic_sha256": sha(r2_dir / "diagnostic.json"),
                          "r2_inner_sha256": sha(r2_dir / "inner.parquet")}}
    (out / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"late_raw": report["late_raw"]["1_1"],
                      "late_calibrated": report["late_calibrated"]["1_1"],
                      "replay_error": report["replay_max_abs_error"]}, indent=2))


if __name__ == "__main__":
    main()
