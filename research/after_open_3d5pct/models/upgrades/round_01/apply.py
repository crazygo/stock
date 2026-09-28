"""Round 01: Logit bias selected on tune."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit

import sys
cur = Path(__file__).resolve()
while cur.parent != cur and not (cur / "research" / "after_open_3d5pct").is_dir():
    cur = cur.parent
REPO_ROOT = cur
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.after_open_3d5pct.focus_v8.core import GROUPS
AFTER = REPO_ROOT / "research" / "after_open_3d5pct"
RUNS_BASE = AFTER / "runs" / "model_registry_v1"
OUTPUT_DIR = AFTER / "runs" / "model_registry_v1_upgrades" / "round_01"

MODELS_5 = ["B_group", "B_no_group", "C_group", "C_no_group", "C_no_daily"]
ALL_MODELS = MODELS_5 + ["SNXX_3d3pct"]

GRID = np.round(np.arange(-2.0, 2.05, 0.05), 2)


def get_union_symbols() -> list[str]:
    return sorted({s for symbols in GROUPS.values() for s in symbols})


def apply_bias(p: np.ndarray, b: float) -> np.ndarray:
    p_clip = np.clip(p, 1e-4, 1.0 - 1e-4)
    return expit(logit(p_clip) + b)


def calc_acc(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p >= 0.5) == (y == 1)).mean())


def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    union_symbols = get_union_symbols()
    results = {}

    for model_id in ALL_MODELS:
        model_out = OUTPUT_DIR / model_id
        model_out.mkdir(parents=True, exist_ok=True)

        if model_id in MODELS_5:
            base_dir = RUNS_BASE / model_id / "reserved"
            y_col = "y_3d_5pct"
            p_col = "p_3d_5pct"
            use_union = True
        else:
            base_dir = RUNS_BASE / model_id
            y_col = "y"
            p_col = "p"
            use_union = False

        df_tune = pd.read_parquet(base_dir / "tune.parquet")
        df_cal = pd.read_parquet(base_dir / "cal.parquet")
        df_eval = pd.read_parquet(base_dir / "eval.parquet")

        slice_tune = df_tune[df_tune.symbol.isin(union_symbols)] if use_union else df_tune
        slice_cal = df_cal[df_cal.symbol.isin(union_symbols)] if use_union else df_cal
        slice_eval = df_eval[df_eval.symbol.isin(union_symbols)] if use_union else df_eval

        y_t = slice_tune[y_col].to_numpy().astype(int)
        p_t = slice_tune[p_col].to_numpy().astype(float)
        base_t_acc = calc_acc(p_t, y_t)

        y_c = slice_cal[y_col].to_numpy().astype(int)
        p_c = slice_cal[p_col].to_numpy().astype(float)
        base_c_acc = calc_acc(p_c, y_c)

        y_e = slice_eval[y_col].to_numpy().astype(int)
        p_e = slice_eval[p_col].to_numpy().astype(float)
        base_e_acc = calc_acc(p_e, y_e)

        # Search b on tune
        # Tie-break: highest acc, then smallest abs(b), then 0
        best_b = 0.0
        best_t_acc = -1.0
        for b in sorted(GRID, key=lambda x: (abs(x), x)):
            p_prime = apply_bias(p_t, b)
            acc = calc_acc(p_prime, y_t)
            if acc > best_t_acc:
                best_t_acc = acc
                best_b = float(b)
            elif acc == best_t_acc:
                if abs(b) < abs(best_b):
                    best_b = float(b)
                elif abs(b) == abs(best_b) and best_b != 0 and b == 0:
                    best_b = 0.0

        # Retention condition: tune accuracy must be strictly greater than baseline tune accuracy
        retained = bool(best_t_acc > base_t_acc)

        # Evaluate on cal and eval
        p_c_prime = apply_bias(p_c, best_b)
        cal_acc = calc_acc(p_c_prime, y_c)

        p_e_prime = apply_bias(p_e, best_b)
        eval_acc = calc_acc(p_e_prime, y_e)

        rel_improvement = float(eval_acc / base_e_acc - 1.0)
        passed = bool(rel_improvement >= 0.10)

        # If retained, save adjusted probability; if not retained, rollback to baseline
        for split_name, split_df in [("tune", df_tune), ("cal", df_cal), ("eval", df_eval)]:
            out_df = split_df.copy()
            if retained:
                out_df[p_col] = apply_bias(out_df[p_col].to_numpy().astype(float), best_b)
            out_df.to_parquet(model_out / f"{split_name}.parquet", index=False)

        results[model_id] = {
            "model_id": model_id,
            "b": best_b,
            "tune_baseline_accuracy": base_t_acc,
            "tune_accuracy": best_t_acc,
            "cal_baseline_accuracy": base_c_acc,
            "cal_accuracy": cal_acc,
            "eval_baseline_accuracy": base_e_acc,
            "eval_accuracy": eval_acc,
            "relative_improvement": rel_improvement,
            "retained": retained,
            "passed": passed,
        }

    out_file = OUTPUT_DIR / "result.json"
    out_file.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    res = run()
    print(json.dumps(res, indent=2))
