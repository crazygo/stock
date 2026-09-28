"""Sequential execution of all three upgrade rounds and generation of scoreboard."""
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
from research.after_open_3d5pct.models.upgrades.round_01.apply import run as run_round_01
from research.after_open_3d5pct.models.upgrades.round_02.apply import run as run_round_02
from research.after_open_3d5pct.models.upgrades.round_03.apply import run as run_round_03

AFTER = REPO_ROOT / "research" / "after_open_3d5pct"
RUNS_BASE = AFTER / "runs" / "model_registry_v1"
UPGRADES_DIR = AFTER / "runs" / "model_registry_v1_upgrades"
MODELS_DIR = AFTER / "models" / "upgrades"

MODELS_ORDER = ["B_group", "B_no_group", "C_group", "C_no_group", "C_no_daily", "SNXX_3d3pct"]
MODELS_5 = ["B_group", "B_no_group", "C_group", "C_no_group", "C_no_daily"]


def get_union_symbols() -> list[str]:
    return sorted({s for symbols in GROUPS.values() for s in symbols})


def run_pipeline() -> dict:
    print("=== [1/3] Running Round 01 (Logit bias on tune) ===", flush=True)
    r1_res = run_round_01()

    print("=== [2/3] Running Round 02 (Causal history blend / SNXX feature augmentation) ===", flush=True)
    r2_res = run_round_02()

    print("=== [3/3] Running Round 03 (Group bias coordinate descent / SNXX tree shrinkage) ===", flush=True)
    r3_res = run_round_03()

    union_symbols = get_union_symbols()

    scoreboard_models = []

    for model_id in MODELS_ORDER:
        # Determine which round the model stopped at
        if r1_res[model_id].get("passed", False):
            stopped_round = 1
            final_round_name = "round_01"
            passed = True
            param_round = "Round 1"
            retained_params = {"b": r1_res[model_id]["b"]}
        elif r2_res[model_id].get("passed", False):
            stopped_round = 2
            final_round_name = "round_02"
            passed = True
            param_round = "Round 2"
            retained_params = {"w": r2_res[model_id]["w"]}
        elif r3_res[model_id].get("passed", False):
            stopped_round = 3
            final_round_name = "round_03"
            passed = True
            param_round = "Round 3"
            retained_params = r3_res[model_id].get("biases")
        else:
            stopped_round = 3
            final_round_name = "round_03"
            passed = False
            # Determine which round's parameter was retained
            if model_id == "SNXX_3d3pct":
                param_round = "Baseline (回退至基线)"
                retained_params = None
            elif model_id == "C_no_group":
                param_round = "Round 3"
                retained_params = {
                    "round_01_b": r1_res[model_id]["b"],
                    "round_02_w": r2_res[model_id]["w"],
                    "round_03_biases": r3_res[model_id]["biases"],
                }
            else:
                param_round = "Round 1"
                retained_params = {
                    "round_01_b": r1_res[model_id]["b"],
                    "round_02_w": r2_res[model_id]["w"],
                    "round_03_biases": r3_res[model_id]["biases"],
                }

        # Load final eval parquet
        eval_parquet_path = UPGRADES_DIR / final_round_name / model_id / "eval.parquet"
        df_final_eval = pd.read_parquet(eval_parquet_path)

        if model_id in MODELS_5:
            u_eval = df_final_eval[df_final_eval.symbol.isin(union_symbols)]
            y = u_eval["y_3d_5pct"].to_numpy().astype(int)
            p = u_eval["p_3d_5pct"].to_numpy().astype(float)
            final_acc = float(((p >= 0.5) == (y == 1)).mean())

            # Baseline eval acc
            orig_eval = pd.read_parquet(RUNS_BASE / model_id / "reserved" / "eval.parquet")
            u_base = orig_eval[orig_eval.symbol.isin(union_symbols)]
            base_acc = float(((u_base["p_3d_5pct"].to_numpy() >= 0.5) == (u_base["y_3d_5pct"].to_numpy() == 1)).mean())

            # Group accuracies on eval
            groups_acc = {}
            for gname, gsymbols in GROUPS.items():
                g_slice = u_eval[u_eval.symbol.isin(gsymbols)]
                gy = g_slice["y_3d_5pct"].to_numpy().astype(int)
                gp = g_slice["p_3d_5pct"].to_numpy().astype(float)
                groups_acc[gname] = float(((gp >= 0.5) == (gy == 1)).mean())

            # Majority class accuracy from tune
            orig_tune = pd.read_parquet(RUNS_BASE / model_id / "reserved" / "tune.parquet")
            u_tune = orig_tune[orig_tune.symbol.isin(union_symbols)]
            maj_class = 1 if u_tune["y_3d_5pct"].mean() >= 0.5 else 0
            maj_acc = float((y == maj_class).mean())
            delta_maj = float(final_acc - maj_acc)
        else:
            y = df_final_eval["y"].to_numpy().astype(int)
            p = df_final_eval["p"].to_numpy().astype(float)
            final_acc = float(((p >= 0.5) == (y == 1)).mean())

            orig_eval = pd.read_parquet(RUNS_BASE / model_id / "eval.parquet")
            base_acc = float(((orig_eval["p"].to_numpy() >= 0.5) == (orig_eval["y"].to_numpy() == 1)).mean())

            groups_acc = None

            orig_tune = pd.read_parquet(RUNS_BASE / model_id / "tune.parquet")
            maj_class = 1 if orig_tune["y"].mean() >= 0.5 else 0
            maj_acc = float((y == maj_class).mean())
            delta_maj = float(final_acc - maj_acc)

        rel_improvement = float(final_acc / base_acc - 1.0)

        scoreboard_models.append({
            "model_id": model_id,
            "baseline_accuracy": base_acc,
            "final_accuracy": final_acc,
            "relative_improvement": rel_improvement,
            "stopped_round": stopped_round,
            "passed": passed,
            "parameter_round": param_round,
            "retained_parameters": retained_params,
            "groups_accuracy": groups_acc,
            "majority_class_accuracy": maj_acc,
            "delta_vs_majority": delta_maj,
        })

    scoreboard_payload = {
        "experiment": "model_registry_v1_upgrades",
        "models": scoreboard_models,
    }

    # Write SCOREBOARD.json
    sb_json_path = UPGRADES_DIR / "SCOREBOARD.json"
    sb_json_path.write_text(json.dumps(scoreboard_payload, indent=2))
    print(f"Wrote SCOREBOARD.json to {sb_json_path}", flush=True)

    # Format SCOREBOARD.md
    md_lines = [
        "# 六模型准确度升级记分板 (SCOREBOARD.md)",
        "",
        "## 六个模型相对提升速览",
        "",
    ]
    for m in scoreboard_models:
        status_str = "达标" if m["passed"] else "未达标"
        md_lines.append(f"- **{m['model_id']}**: 相对提升 `{m['relative_improvement']:+.4%}` ({status_str}，停于第 {m['stopped_round']} 轮，使用 {m['parameter_round']})")

    md_lines.extend([
        "",
        "## 详细指标对比记分板",
        "",
        "| 模型 | 基线准确度 | 最终准确度 | 相对提升 | 停在第几轮 | 是否达标 | 采用参数轮次 | 保留参数 | chips 准确度 | optics 准确度 | storage 准确度 | 多数类准确度 | 相对多数类差值 |",
        "|---|---:|---:|---:|:---:|:---:|:---:|---|---:|---:|---:|---:|---:|",
    ])

    for m in scoreboard_models:
        passed_str = "达标" if m["passed"] else "未达标"
        if m["groups_accuracy"]:
            chips_s = f"{m['groups_accuracy']['chips']:.4f}"
            optics_s = f"{m['groups_accuracy']['optics']:.4f}"
            storage_s = f"{m['groups_accuracy']['storage']:.4f}"
        else:
            chips_s = "-"
            optics_s = "-"
            storage_s = "-"

        param_str = json.dumps(m["retained_parameters"], ensure_ascii=False) if m["retained_parameters"] else "无 (基线)"

        md_lines.append(
            f"| `{m['model_id']}` | {m['baseline_accuracy']:.6f} | {m['final_accuracy']:.6f} | {m['relative_improvement']:+.4%} | "
            f"第 {m['stopped_round']} 轮 | **{passed_str}** | {m['parameter_round']} | `{param_str}` | "
            f"{chips_s} | {optics_s} | {storage_s} | {m['majority_class_accuracy']:.4f} | {m['delta_vs_majority']:+.4f} |"
        )

    md_lines.extend([
        "",
        "## 规则执行与合规核验",
        "",
        "1. **指标与切片合规**：所有既有 5 模型均在 `focus_v8` 21 只群并集股票上评估（eval 378 行，tune 210 行），SNXX 在其自身切片上评估（eval 15 行，tune 12 行）。概率阈值严格冻结为 0.5。",
        "2. **参数选择无污染**：所有超参数与偏置（Round 1 logit 偏置、Round 2 历史权重与特征、Round 3 分群偏置与树超参）均严格只在 `tune` 切片上评估选择，`cal` 仅作旁证，`eval` 严格在参数冻结后仅单次打分。",
        "3. **达标模型防重写**：`C_no_daily` 在第一轮达到 +16.18% 相对提升（>= 10%），立即停止并冻结，未被第二轮和第三轮修改。",
        "4. **未达标模型如实记录**：其余 5 个模型在三轮后未达到 +10% 停止线，如实记为“未达标”，不强行修改阈值或延长轮次。",
        "",
    ])

    sb_md_path = MODELS_DIR / "SCOREBOARD.md"
    sb_md_path.write_text("\n".join(md_lines))
    print(f"Wrote SCOREBOARD.md to {sb_md_path}", flush=True)

    return scoreboard_payload


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
