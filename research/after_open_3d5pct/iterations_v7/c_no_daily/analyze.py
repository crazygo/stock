"""Build C/no-daily paired diagnostics from frozen completed runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import BASE, TARGETS, metric, sha

HERE = Path(__file__).resolve().parent
FINAL = BASE / "runs/v7_c_no_daily_final_20260927"
RUNS = {"R1": BASE/"runs/v7_c_no_daily_r1_3566",
        "R1control": BASE/"runs/v7_c_no_daily_r1_control_3566",
        "R2": BASE/"runs/v7_c_no_daily_r2_3566",
        "R3_seed3566": BASE/"runs/v7_c_no_daily_r3_3566",
        "R3_seed4577": BASE/"runs/v7_c_no_daily_r3_4577"}


def ece10(y, p):
    bins = np.minimum((p * 10).astype(int), 9)
    return float(sum(np.mean(bins == j) * abs(y[bins == j].mean() - p[bins == j].mean())
                     for j in range(10) if np.any(bins == j)))


def block_ci(a, b, dates, seed=20260927, samples=3000):
    """Paired Brier(a)-Brier(b), moving five-date blocks; exposed-period diagnostic."""
    days = sorted(set(dates))
    by_day = {d: np.flatnonzero(dates == d) for d in days}
    rng = np.random.default_rng(seed)
    diff = []
    for _ in range(samples):
        chosen = []
        while len(chosen) < len(days):
            start = rng.integers(0, len(days))
            chosen.extend(days[start:min(start+5, len(days))])
        ix = np.concatenate([by_day[d] for d in chosen[:len(days)]])
        diff.append(np.mean(a[ix] - b[ix]))
    return [float(x) for x in np.quantile(diff, [.025, .975])]


def main():
    full = json.loads((FINAL/"results.json").read_text())
    frames = {name: {side: pd.read_parquet(FINAL/f"{name}_{side}.parquet")
                     for side in ("inner", "outer")} for name in full}
    control = pd.read_parquet(RUNS["R1control"]/"inner.parquet")
    results = {"dataset_manifest_sha256": sha(BASE/"runs/multiscale_groups_v61_dataset_20260926_r2/manifest.json"),
               "protocol": "v7_C_no_daily_three_round_exposed_development",
               "rows": {"all": 9222, "train": 5606, "inner": 1521, "outer": 1140},
               "targets": TARGETS, "models": {}, "paired_date_block_ci_5day": {},
               "control_inner_only": {"R1control": {"primary": metric(control["y_3d_5pct"], control["p_3d_5pct"]),
                                                      "run": str(RUNS["R1control"])}},
               "all_dates_exposed": True, "no_daily_read": True, "calibration": "none_raw_and_projected_only"}
    for name, item in full.items():
        run = RUNS.get(name)
        result = {"model_sha256": item["model_sha256"], "replay_max_abs": item["max_replay_delta_inner"],
                  "inner": {}, "outer": {}}
        if run:
            report = json.loads((run/"report.json").read_text())
            result.update({"best_epoch": report["best_epoch"], "fit_seconds": report["seconds"],
                           "parameters": report["parameters"], "training_source_hashes": report["hashes"],
                           "train_curve": report["curve"]})
        for side in ("inner", "outer"):
            frame = frames[name][side]
            result[side]["dates"] = int(frame.session_date.nunique())
            result[side]["targets"] = item[side]
            result[side]["ece10"] = {t: ece10(frame[f"y_{t}"].to_numpy(), frame[f"p_{t}"].to_numpy())
                                     for t in TARGETS}
        results["models"][name] = result
    for side in ("inner", "outer"):
        base = frames["R0"][side]
        dates = base.session_date.to_numpy()
        y = base["y_3d_5pct"].to_numpy()
        for name in ("R1", "R2", "R3_seed3566", "R3_seed4577"):
            a = frames[name][side]
            for baseline_name in ("R0", "R1"):
                if name == baseline_name:
                    continue
                b = frames[baseline_name][side]
                assert (a.sample_id.to_numpy() == b.sample_id.to_numpy()).all()
                loss_a = (y-a["p_3d_5pct"].to_numpy())**2
                loss_b = (y-b["p_3d_5pct"].to_numpy())**2
                results["paired_date_block_ci_5day"][f"{side}:{name}-{baseline_name}"] = {
                    "delta_brier": float(np.mean(loss_a-loss_b)),
                    "ci95": block_ci(loss_a, loss_b, dates)}
    # R1 control stayed strictly inner-only.
    assert control.sample_id.equals(frames["R1"]["inner"].sample_id)
    y = control["y_3d_5pct"].to_numpy()
    a = (y-frames["R1"]["inner"]["p_3d_5pct"].to_numpy())**2
    b = (y-control["p_3d_5pct"].to_numpy())**2
    results["paired_date_block_ci_5day"]["inner:R1-R1control"] = {
        "delta_brier": float(np.mean(a-b)), "ci95": block_ci(a, b, control.session_date.to_numpy())}
    b0 = (y-frames["R0"]["inner"]["p_3d_5pct"].to_numpy())**2
    results["paired_date_block_ci_5day"]["inner:R1control-R0"] = {
        "delta_brier": float(np.mean(b-b0)), "ci95": block_ci(b, b0, control.session_date.to_numpy())}
    (HERE/"results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in results["paired_date_block_ci_5day"].items()
                      if k.startswith("outer:") or k == "inner:R1-R1control"}, indent=2))


if __name__ == "__main__":
    main()
