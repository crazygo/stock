"""One-time exposed September scoring after R0-R3 are frozen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.after_open_3d5pct.iterations_v7.b_no_group.runner import (
    BASE, DATASET, EXPECTED_MANIFEST, HERE, R0, load, matrix, metrics, path_boundary_check, sha,
)
from research.after_open_3d5pct.train_multiscale_v6 import project_monotone


def ids_hash(ids) -> str:
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def bins(y, p):
    index = np.minimum(np.floor(np.clip(p, 0, 1) * 10).astype(int), 9)
    result = []
    for i in range(10):
        chosen = index == i
        result.append({"low": i/10, "high": (i+1)/10, "n": int(chosen.sum()),
                       "mean_p": float(np.mean(p[chosen])) if chosen.any() else None,
                       "rate": float(np.mean(y[chosen])) if chosen.any() else None})
    ece = sum(b["n"] * abs(b["mean_p"] - b["rate"]) for b in result if b["n"]) / len(y)
    return {"ece10": float(ece), "buckets": result}


def score(df):
    out = {}
    for h in range(3):
        for t in range(3):
            y = df[f"y_{h}_{t}"].to_numpy()
            p = df[f"p_{h}_{t}"].to_numpy()
            out[f"{h}_{t}"] = {**metrics(y, p), **bins(y, p)}
    return out


def ci(ref, new, target, seed):
    dates = sorted(ref.session_date.unique())
    if len(dates) < 5:
        return [None, None]
    ys = ref[f"y_{target}"].to_numpy()
    pr = ref[f"p_{target}"].to_numpy()
    pn = new[f"p_{target}"].to_numpy()
    loss_delta = (ys - pn)**2 - (ys-pr)**2
    ix_by_date = {d: np.flatnonzero(ref.session_date.to_numpy() == d) for d in dates}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(1000):
        chosen = []
        while len(chosen) < len(dates):
            start = int(rng.integers(0, len(dates)))
            chosen.extend(dates[start:min(start+5, len(dates))])
        ix = np.concatenate([ix_by_date[d] for d in chosen[:len(dates)]])
        boot.append(float(np.mean(loss_delta[ix])))
    return [float(v) for v in np.quantile(boot, [0.025, .975])]


def ensure_same(frames):
    names = list(frames)
    ref = frames[names[0]]
    for name in names[1:]:
        other = frames[name]
        assert other.sample_id.tolist() == ref.sample_id.tolist(), name
        assert other.session_date.tolist() == ref.session_date.tolist(), name
        for h in range(3):
            for t in range(3):
                assert np.array_equal(other[f"y_{h}_{t}"], ref[f"y_{h}_{t}"]), (name, h, t)


def main():
    path_boundary_check()
    table, data, folds = load()
    rdir = {r: BASE / f"runs/v7_b_no_group_{r}" for r in ("r0", "r1", "r2", "r3")}
    for r in rdir:
        assert (rdir[r] / "diagnostic.json").exists(), r
    inner = {r: pd.read_parquet(rdir[r] / "inner.parquet") for r in rdir}
    outer = {"r0": pd.read_parquet(R0 / "predictions_B_lgbm_no_group.parquet")}
    y_outer = data["y"][folds["outer"]]
    for r in ("r1", "r2", "r3"):
        o = pd.read_parquet(rdir[r] / "outer_blind.parquet")
        for h in range(3):
            for t in range(3):
                o[f"y_{h}_{t}"] = y_outer[:, h, t]
        outer[r] = o
    ensure_same(inner)
    ensure_same(outer)
    # Confirm the frozen R0 September artifact is reproducible from its original trees.
    x0 = matrix(data, "r0")
    raw0 = np.column_stack([lgb.Booster(model_file=str(R0 / f"models/B_lgbm_no_group_{j}.txt")).predict(
        x0.iloc[folds["outer"]], num_threads=1) for j in range(9)])
    replay0 = project_monotone(raw0).reshape(-1, 3, 3)
    replay0_error = max(float(np.max(np.abs(replay0[:, h, t] - outer["r0"][f"p_{h}_{t}"].to_numpy())))
                        for h in range(3) for t in range(3))
    assert replay0_error < 1e-7, replay0_error
    late_ids = pd.read_parquet(rdir["r3"] / "inner_late_temporal.parquet").sample_id.tolist()
    late = {r: df.set_index("sample_id").loc[late_ids].reset_index() for r, df in inner.items()}
    late["r3"] = pd.read_parquet(rdir["r3"] / "inner_late_temporal.parquet")
    ensure_same(late)
    targets = [f"{h}_{t}" for h in range(3) for t in range(3)]
    results = {
        "status": "exposed_2026_development_only_no_outer_selection",
        "schema": "v7_b_no_group_9target_predictions_v1",
        "dataset_manifest_sha256": EXPECTED_MANIFEST,
        "rows_parquet_sha256": sha(DATASET / "rows.parquet"),
        "features_npz_sha256": sha(DATASET / "features.npz"),
        "code_sha256": {f: sha(HERE / f) for f in ("runner.py", "calibrate_r3.py", "finalize.py")},
        "sample_ids_sha256": {"inner": ids_hash(inner["r0"].sample_id.tolist()),
                              "inner_late": ids_hash(late["r0"].sample_id.tolist()),
                              "outer": ids_hash(outer["r0"].sample_id.tolist())},
        "splits": {"train_rows": len(folds["train"]), "inner_rows": len(folds["inner"]),
                   "inner_late_rows": len(late_ids), "outer_rows": len(folds["outer"]),
                   "inner_dates": int(inner["r0"].session_date.nunique()),
                   "inner_late_dates": int(late["r0"].session_date.nunique()),
                   "outer_dates": int(outer["r0"].session_date.nunique())},
        "scores": {"inner_full": {r: score(df) for r, df in inner.items()},
                   "inner_late_temporal": {r: score(df) for r, df in late.items()},
                   "outer": {r: score(df) for r, df in outer.items()}},
        "paired_brier_delta_vs_r0_5day_ci": {
            split: {r: {target: {"point": float(scores[r][target]["brier"] - scores["r0"][target]["brier"]),
                                 "ci95": ci(frames["r0"], frames[r], target, 3566 + 100*i + 10*j + k)}
                         for k, target in enumerate(targets)}
                    for j, r in enumerate(("r1", "r2", "r3"))}
            for i, (split, frames, scores) in enumerate((
                ("inner_full", inner, {r: score(df) for r, df in inner.items()}),
                ("inner_late_temporal", late, {r: score(df) for r, df in late.items()}),
                ("outer", outer, {r: score(df) for r, df in outer.items()})))},
        "replay_max_abs_error": {"r0": replay0_error,
            **{r: json.loads((rdir[r] / "diagnostic.json").read_text())["replay_max_abs_error"]
               for r in ("r1", "r2", "r3")}},
        "model_trees": {r: [d["trees"] for d in json.loads((rdir[r] / "diagnostic.json").read_text())["target_models"]]
                        for r in ("r1", "r2")},
        "model_file_sha256": {r: [sha(rdir[r] / f"models/target_{j}.txt") for j in range(9)]
                              for r in ("r1", "r2", "r3")},
        "prediction_file_sha256": {r: {"inner": sha(rdir[r] / "inner.parquet"),
                                         "outer": sha(R0 / "predictions_B_lgbm_no_group.parquet") if r == "r0" else
                                         sha(rdir[r] / "outer_blind.parquet")}
                                   for r in rdir},
        "r3_note": "inner_full is calibrator fit-in-sample; inner_late_temporal is the time-blocked check, conditional on R2 early stopping using August"
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"outer_primary": {r: results["scores"]["outer"][r]["1_1"]["brier"] for r in rdir},
                      "inner_late_primary": {r: results["scores"]["inner_late_temporal"][r]["1_1"]["brier"] for r in rdir},
                      "r0_replay_error": replay0_error}, indent=2))


if __name__ == "__main__":
    main()
