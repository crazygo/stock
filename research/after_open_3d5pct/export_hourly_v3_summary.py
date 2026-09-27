"""Export static, evidence-linked training summary for the hourly dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .hourly_v3_snapshot import sha256

PROJECT = Path(__file__).resolve().parent
OLD_RUN = PROJECT/"runs/market_dual_track_pilot_20260925"


def _block_interval(daily: pd.DataFrame, block: int, seed: int = 3505) -> list[float] | None:
    if len(daily) < block*2:
        return None
    values = daily.difference.to_numpy(float)
    rng = np.random.default_rng(seed+block)
    out = []
    for _ in range(500):
        sample = []
        while len(sample) < len(values):
            start = int(rng.integers(0, max(1, len(values)-block+1)))
            sample.extend(values[start:start+block])
        out.append(float(np.mean(sample[:len(values)])))
    return np.quantile(out, [0.025, 0.975]).tolist()


def export(run: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    run = run.resolve()
    manifest = json.loads((run/"manifest.json").read_text())
    config = json.loads((run/"config.json").read_text())
    predictions = pd.read_parquet(run/"predictions.parquet")
    trials = [json.loads(x) for x in (run/"trials.jsonl").read_text().splitlines() if x]
    metrics = pd.read_csv(run/"metrics.csv").replace({np.nan: None}).to_dict("records")
    per_hour = []
    for (fold, model, inputs, kind, hour), g in predictions.groupby(
            ["fold", "model", "input_set", "score_kind", "cutoff_et"], dropna=False):
        threshold = None
        cal_path = run/"models"/f"{fold}_{model}_{inputs}.calibration.json"
        if cal_path.exists():
            threshold = json.loads(cal_path.read_text())["threshold"]["threshold"]
        chosen = (g.probability.to_numpy(float) >= threshold) if threshold is not None and kind == "calibrated" else np.zeros(len(g), bool)
        per_hour.append({"fold": fold, "model": model, "input_set": inputs,
                         "score_kind": kind, "hour_et": hour,
                         "mature_n": len(g), "trading_days": int(g.session_date.nunique()),
                         "touches": int(g.target.sum()), "touch_rate": float(g.target.mean()),
                         "predicted_mean": float(g.probability.mean()),
                         "brier": float(np.mean((g.probability-g.target)**2)),
                         "recommendations": int(chosen.sum()),
                         "recommended_touches": int(g.target.to_numpy()[chosen].sum()),
                         "recommended_touch_rate": float(g.target.to_numpy()[chosen].mean()) if chosen.any() else None,
                         "misses": int(g.target.to_numpy()[~chosen].sum()) if threshold is not None else None,
                         "threshold": threshold})
    per_day = []
    for (fold, model, inputs, kind, day, hour), g in predictions.groupby(
            ["fold", "model", "input_set", "score_kind", "session_date", "cutoff_et"]):
        threshold = None
        cal_path = run/"models"/f"{fold}_{model}_{inputs}.calibration.json"
        if kind == "calibrated" and cal_path.exists():
            threshold = json.loads(cal_path.read_text())["threshold"]["threshold"]
        chosen = (g.probability.to_numpy(float) >= threshold) if threshold is not None else np.zeros(len(g), bool)
        per_day.append({"fold": fold, "model": model, "input_set": inputs,
                        "score_kind": kind, "session_date": day, "hour_et": hour,
                        "mature_n": len(g), "touches": int(g.target.sum()),
                        "touch_rate": float(g.target.mean()),
                        "brier": float(np.mean((g.probability-g.target)**2)),
                        "recommendations": int(chosen.sum()),
                        "recommended_touches": int(g.target.to_numpy()[chosen].sum()),
                        "false_positives": int(chosen.sum()-g.target.to_numpy()[chosen].sum()),
                        "misses": int(g.target.to_numpy()[~chosen].sum()) if threshold is not None else None})
    paired_old = []
    if (OLD_RUN/"predictions.parquet").exists():
        old = pd.read_parquet(OLD_RUN/"predictions.parquet")
        for (fold, model, inputs), newer in predictions[(predictions.score_kind == "raw") &
                                                         (predictions.cutoff_et.isin(["11:30", "12:30", "13:30", "14:30"]))].groupby(
                ["fold", "model", "input_set"]):
            older = old[(old.fold == fold) & (old.model == model) & (old.input_set == inputs)]
            joined = newer.merge(older[["sample_id", "probability"]], on="sample_id", suffixes=("_new", "_old"))
            if joined.empty:
                continue
            joined["difference"] = (joined.probability_old-joined.target)**2 - (joined.probability_new-joined.target)**2
            daily = joined.groupby("session_date").difference.mean().reset_index()
            paired_old.append({"fold": fold, "model": model, "input_set": inputs,
                               "common_n": len(joined), "common_days": len(daily),
                               "old_brier": float(np.mean((joined.probability_old-joined.target)**2)),
                               "new_brier": float(np.mean((joined.probability_new-joined.target)**2)),
                               "old_minus_new_brier": float(joined.difference.mean()),
                               "day_block_5_interval": _block_interval(daily, 5),
                               "day_block_10_interval": _block_interval(daily, 10),
                               "scope": "exposed_development_common_sample"})
    outcomes = pd.read_parquet(run/"outcomes.parquet")
    focus = predictions[(predictions.score_kind == "calibrated") &
                        (predictions.model == "lgbm") & (predictions.input_set == "HAB")].merge(
                            outcomes[["sample_id", "entry_at", "entry_price", "label_end_at"]], on="sample_id")
    focus_thresholds = {fold: json.loads((run/"models"/f"{fold}_lgbm_HAB.calibration.json").read_text())["threshold"]["threshold"]
                        for fold in focus.fold.unique()}
    samples = []
    for r in focus.itertuples():
        threshold = focus_thresholds[r.fold]
        samples.append({"sample_id": r.sample_id, "fold": r.fold, "symbol": r.symbol,
                        "session_date": r.session_date, "hour_et": r.cutoff_et,
                        "calibrated_output": float(r.probability),
                        "raw_output": float(r.raw_probability), "target": int(r.target),
                        "research_threshold_passed": bool(threshold is not None and r.probability >= threshold),
                        "entry_at": r.entry_at, "entry_price": float(r.entry_price),
                        "label_end_at": r.label_end_at})
    paths = {}
    path_dir = run/"paths"
    if path_dir.exists():
        raise FileExistsError(path_dir)
    path_dir.mkdir()
    repo = PROJECT.parents[1]
    for symbol in sorted(focus.symbol.unique()):
        source = repo/"market_data/us_5m"/symbol/"2026.parquet"
        f = pd.read_parquet(source, columns=["start_at", "session_date", "session_type",
                                            "open", "high", "low", "close"])
        f = f[(f.session_date >= "2026-08-03") & (f.session_date <= "2026-09-24") &
              (f.session_type == "regular")]
        bars = [{"t": r.start_at, "o": float(r.open), "h": float(r.high),
                 "l": float(r.low), "c": float(r.close)} for r in f.itertuples()]
        path = path_dir/f"{symbol}.json"
        path.write_text(json.dumps({"symbol": symbol, "source_sha256": sha256(source),
                                    "bars": bars}, ensure_ascii=False, separators=(",", ":")))
        paths[symbol] = sha256(path)
    summary = {"version": "hourly_training_dashboard_v3_2", "training_run": str(run),
               "manifest_sha256": sha256(run/"manifest.json"),
               "predictions_sha256": sha256(run/"predictions.parquet"),
               "scope": manifest["scope"], "historical_availability": manifest["synthetic_availability"],
               "universe_mode": config["universe_mode"], "night_session_observed": False,
               "supported_cutoffs_et": config["decision_hours_et"],
               "models": {"lgbm": "LightGBM on H/A/B summaries; fixed small parameter set",
                          "tcn": "Multi-branch causal TCN on H/post/pre/B sequences plus static scales"},
               "input_sets": {"H": "10 preceding regular sessions",
                              "HA": "H plus previous postmarket and current premarket; no night observations",
                              "HB": "H plus current regular prefix",
                              "HAB": "H plus A and B"},
               "metrics": metrics, "per_hour": per_hour, "per_day": per_day,
               "samples": samples, "paths_sha256": paths,
               "paired_old_common_sample": paired_old,
               "trials": trials, "manifest": manifest,
               "release_gate": "pending_forward_independent_validation"}
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PROJECT/"runs/hourly_once_v3_20260925")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.output or args.run/"dashboard_summary_v2.json"
    result = export(args.run, path)
    print(json.dumps({"output": str(path), "per_hour": len(result["per_hour"]),
                      "paired": len(result["paired_old_common_sample"])}, ensure_ascii=False))
