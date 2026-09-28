"""Round 02: Blend with causal history benchmark / SNXX feature augmentation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

cur = Path(__file__).resolve()
while cur.parent != cur and not (cur / "research" / "after_open_3d5pct").is_dir():
    cur = cur.parent
REPO_ROOT = cur
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.after_open_3d5pct.focus_v8.core import GROUPS
from research.after_open_3d5pct.models.upgrades.round_02.snxx_train import train_round2

AFTER = REPO_ROOT / "research" / "after_open_3d5pct"
RUNS_BASE = AFTER / "runs" / "model_registry_v1"
R1_DIR = AFTER / "runs" / "model_registry_v1_upgrades" / "round_01"
OUTPUT_DIR = AFTER / "runs" / "model_registry_v1_upgrades" / "round_02"

MODELS_5 = ["B_group", "B_no_group", "C_group", "C_no_group", "C_no_daily"]
ALL_MODELS = MODELS_5 + ["SNXX_3d3pct"]
WEIGHT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def get_union_symbols() -> list[str]:
    return sorted({s for symbols in GROUPS.values() for s in symbols})


def calc_acc(p: np.ndarray, y: np.ndarray) -> float:
    return float(((p >= 0.5) == (y == 1)).mean())


def run() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    union_symbols = get_union_symbols()

    # Load round 1 result
    r1_file = R1_DIR / "result.json"
    if not r1_file.exists():
        raise FileNotFoundError(f"Round 01 result not found at {r1_file}")
    r1_results = json.loads(r1_file.read_text())

    results = {}

    for model_id in ALL_MODELS:
        r1_mod = r1_results[model_id]
        # If already passed in round 1, freeze and skip
        if r1_mod.get("passed", False):
            results[model_id] = {
                "model_id": model_id,
                "status": "already_passed_round_01",
                "stopped_round": 1,
                "passed": True,
                "retained": True,
                "w": None,
                "tune_baseline_accuracy": r1_mod["tune_baseline_accuracy"],
                "tune_accuracy": r1_mod["tune_accuracy"],
                "eval_baseline_accuracy": r1_mod["eval_baseline_accuracy"],
                "eval_accuracy": r1_mod["eval_accuracy"],
                "relative_improvement": r1_mod["relative_improvement"],
            }
            continue

        model_out = OUTPUT_DIR / model_id
        model_out.mkdir(parents=True, exist_ok=True)

        if model_id in MODELS_5:
            # Load baseline parquets for history_3d_5pct and original baseline scores
            orig_base = RUNS_BASE / model_id / "reserved"
            orig_tune = pd.read_parquet(orig_base / "tune.parquet")
            orig_cal = pd.read_parquet(orig_base / "cal.parquet")
            orig_eval = pd.read_parquet(orig_base / "eval.parquet")

            # Load round 1 retained parquets (which contains p1)
            r1_tune = pd.read_parquet(R1_DIR / model_id / "tune.parquet")
            r1_cal = pd.read_parquet(R1_DIR / model_id / "cal.parquet")
            r1_eval = pd.read_parquet(R1_DIR / model_id / "eval.parquet")

            # Filter union 21
            u_r1_tune = r1_tune[r1_tune.symbol.isin(union_symbols)]
            u_orig_tune = orig_tune[orig_tune.symbol.isin(union_symbols)]
            u_r1_cal = r1_cal[r1_cal.symbol.isin(union_symbols)]
            u_orig_cal = orig_cal[orig_cal.symbol.isin(union_symbols)]
            u_r1_eval = r1_eval[r1_eval.symbol.isin(union_symbols)]
            u_orig_eval = orig_eval[orig_eval.symbol.isin(union_symbols)]

            # Check history_3d_5pct validity
            hist_t = u_orig_tune["history_3d_5pct"].to_numpy().astype(float)
            valid_t = np.isfinite(hist_t)
            del_t = int((~valid_t).sum())

            hist_e = u_orig_eval["history_3d_5pct"].to_numpy().astype(float)
            valid_e = np.isfinite(hist_e)
            del_e = int((~valid_e).sum())

            # Safety check: must keep >= 90%
            if (len(hist_t) - del_t) < 0.9 * len(hist_t) or (len(hist_e) - del_e) < 0.9 * len(hist_e):
                # Rollback
                results[model_id] = {
                    "model_id": model_id,
                    "status": "aborted_due_to_missing_history",
                    "deleted_tune": del_t,
                    "deleted_eval": del_e,
                    "retained": False,
                    "passed": False,
                }
                continue

            y_t = u_r1_tune["y_3d_5pct"].to_numpy().astype(int)[valid_t]
            p1_t = u_r1_tune["p_3d_5pct"].to_numpy().astype(float)[valid_t]
            h_t = hist_t[valid_t]

            y_e = u_r1_eval["y_3d_5pct"].to_numpy().astype(int)[valid_e]
            p1_e = u_r1_eval["p_3d_5pct"].to_numpy().astype(float)[valid_e]
            h_e = hist_e[valid_e]

            base_t_acc = r1_mod["tune_baseline_accuracy"]
            base_e_acc = r1_mod["eval_baseline_accuracy"]

            # Search w on tune
            best_w = 1.0
            best_t_acc = -1.0
            for w in WEIGHT_GRID:
                p2_t = w * p1_t + (1.0 - w) * h_t
                acc = calc_acc(p2_t, y_t)
                if acc > best_t_acc:
                    best_t_acc = acc
                    best_w = float(w)
                elif acc == best_t_acc:
                    # Tie-break: prefer larger w (less disruption to p1)
                    if w > best_w:
                        best_w = float(w)

            # Retention condition: tune accuracy > baseline tune accuracy
            retained = bool(best_t_acc > base_t_acc)

            # Cal collateral
            hist_c = u_orig_cal["history_3d_5pct"].to_numpy().astype(float)
            valid_c = np.isfinite(hist_c)
            p1_c = u_r1_cal["p_3d_5pct"].to_numpy().astype(float)[valid_c]
            y_c = u_r1_cal["y_3d_5pct"].to_numpy().astype(int)[valid_c]
            h_c = hist_c[valid_c]
            p2_c = best_w * p1_c + (1.0 - best_w) * h_c
            cal_acc = calc_acc(p2_c, y_c)

            # Eval score
            p2_e = best_w * p1_e + (1.0 - best_w) * h_e
            eval_acc = calc_acc(p2_e, y_e)
            rel_improvement = float(eval_acc / base_e_acc - 1.0)
            passed = bool(rel_improvement >= 0.10)

            # Save parquets: if retained use p2, else rollback to p1
            for sname, sdf, orig_sdf in [("tune", r1_tune, orig_tune), ("cal", r1_cal, orig_cal), ("eval", r1_eval, orig_eval)]:
                out_df = sdf.copy()
                if retained:
                    h_full = orig_sdf["history_3d_5pct"].to_numpy().astype(float)
                    p_curr = out_df["p_3d_5pct"].to_numpy().astype(float)
                    # blend where finite
                    mask = np.isfinite(h_full)
                    p_curr[mask] = best_w * p_curr[mask] + (1.0 - best_w) * h_full[mask]
                    out_df["p_3d_5pct"] = p_curr
                out_df.to_parquet(model_out / f"{sname}.parquet", index=False)

            results[model_id] = {
                "model_id": model_id,
                "w": best_w,
                "deleted_rows_tune": del_t,
                "deleted_rows_eval": del_e,
                "tune_baseline_accuracy": base_t_acc,
                "tune_accuracy": best_t_acc,
                "cal_accuracy": cal_acc,
                "eval_baseline_accuracy": base_e_acc,
                "eval_accuracy": eval_acc,
                "relative_improvement": rel_improvement,
                "retained": retained,
                "passed": passed,
            }
        else:
            # SNXX_3d3pct retraining
            snxx_payload = train_round2(model_out)
            base_t_acc = r1_mod["tune_baseline_accuracy"]
            base_e_acc = r1_mod["eval_baseline_accuracy"]

            new_t_acc = snxx_payload["splits"]["tune"]["accuracy_at_0.5"]
            new_e_acc = snxx_payload["splits"]["eval"]["accuracy_at_0.5"]
            new_c_acc = snxx_payload["splits"]["cal"]["accuracy_at_0.5"]

            retained = bool(new_t_acc > base_t_acc)
            rel_improvement = float(new_e_acc / base_e_acc - 1.0)
            passed = bool(rel_improvement >= 0.10)

            # If not retained, rollback parquets to round 1 (or baseline)
            if not retained:
                for sname in ["tune", "cal", "eval"]:
                    prev_df = pd.read_parquet(R1_DIR / model_id / f"{sname}.parquet")
                    prev_df.to_parquet(model_out / f"{sname}.parquet", index=False)

            results[model_id] = {
                "model_id": model_id,
                "scale_pos_weight": snxx_payload["selected_scale_pos_weight"],
                "tune_baseline_accuracy": base_t_acc,
                "tune_accuracy": new_t_acc,
                "cal_accuracy": new_c_acc,
                "eval_baseline_accuracy": base_e_acc,
                "eval_accuracy": new_e_acc,
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
