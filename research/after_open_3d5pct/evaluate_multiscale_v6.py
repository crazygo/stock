"""Read-only paired reliability and terminal-value diagnostics for a v6 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def reliability(y: np.ndarray, p: np.ndarray) -> dict:
    edges = np.linspace(0, 1, 11)
    cells = []
    for j in range(10):
        selected = (p >= edges[j]) & ((p < edges[j+1]) if j < 9 else (p <= edges[j+1]))
        cells.append({"lower": float(edges[j]), "upper": float(edges[j+1]),
                      "n": int(selected.sum()),
                      "predicted": float(np.mean(p[selected])) if selected.any() else None,
                      "observed": float(np.mean(y[selected])) if selected.any() else None})
    ece = sum(c["n"]*abs(c["observed"]-c["predicted"]) for c in cells if c["n"])
    return {"bins": cells, "ece10": float(ece / len(y)),
            "calibration_in_the_large": float(np.mean(p)-np.mean(y))}


def analyze(run: Path, output: Path, baseline_run: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    names = ("B_lgbm_group", "B_lgbm_no_group", "C_patch_group",
             "C_patch_no_group", "C_patch_no_daily")
    paths = {name: (run/f"predictions_{name}.parquet" if (run/f"predictions_{name}.parquet").exists()
                    else (baseline_run/f"predictions_{name}.parquet" if baseline_run is not None else None))
             for name in names}
    if any(path is None or not path.exists() for path in paths.values()):
        raise FileNotFoundError("missing paired prediction; supply --baseline-run")
    frames = {name: pd.read_parquet(path) for name,path in paths.items()}
    reference = frames[names[0]]
    if not all(f.sample_id.equals(reference.sample_id) and f.y_1_1.equals(reference.y_1_1)
               for f in frames.values()):
        raise ValueError("not paired on the same ordered sample and target")
    y = reference.y_1_1.to_numpy(float)
    rows = {name: reliability(y, f.p_1_1.to_numpy(float)) for name,f in frames.items()}
    nonhit = reference.y_1_1.to_numpy() == 0
    terminal = reference.terminal_3d.to_numpy(float)
    # This is a mark-to-market scenario, not a forced sale or observed fill.
    research_net = np.where(nonhit, terminal, .05) - .0012
    rows["outcome_risk_observed_same_rows"] = {
        "nonhit_n": int(nonhit.sum()),
        "nonhit_terminal_mean_gross": float(np.mean(terminal[nonhit])),
        "nonhit_terminal_q10_gross": float(np.quantile(terminal[nonhit], .1)),
        "nonhit_terminal_q01_gross": float(np.quantile(terminal[nonhit], .01)),
        "research_net_mean": float(np.mean(research_net)),
        "research_net_es95": float(np.mean(np.sort(research_net)[:max(1, int(np.ceil(.05*len(research_net))))])),
        "buy_and_sell_cost_each": .0006,
        "not_a_trading_backtest": True}
    rows["scope"] = "2026_exposed_development_11_30_only_no_real_received_at_no_calibration_fit"
    rows["source_run_by_model"] = {name:str(path.parent) for name,path in paths.items()}
    output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--baseline-run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(analyze(a.run, a.output, a.baseline_run), indent=2))
