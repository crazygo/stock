"""Predeclared gates, date-block paired intervals, and compact deliverable tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import HERE, PROTOCOL, GROUPS, TARGETS, metrics, splits
from .prepare import write

LABELS = {"chips": "计算与通信芯片", "optics": "光通信产业链", "storage": "数据与存储"}
ROUTES = {"B_group": "B有群", "B_no_group": "B无群", "C_group": "C有群", "C_no_group": "C无群", "C_no_daily": "C无日级"}


def read(path):
    return json.loads(path.read_text())


def paired_interval(frame, reference, symbols, repeats=2000):
    """Moving-session blocks preserve within-day co-movement and overlapping labels."""
    left = frame[frame.symbol.isin(symbols)].copy()
    ref = reference.set_index("sample_id")["p_3d_5pct"]
    p0 = left.sample_id.map(ref).to_numpy()
    if np.isnan(p0).any():
        raise ValueError("paired comparator lacks sample IDs")
    y = left.y_3d_5pct.to_numpy()
    left["gain"] = (p0-y)**2-(left.p_3d_5pct.to_numpy()-y)**2
    bydate = left.groupby("session_date").gain.agg(["sum", "count"])
    a = bydate.to_numpy()
    n = len(a)
    rng = np.random.default_rng(3566)
    samples = []
    for _ in range(repeats):
        starts = rng.integers(0, max(1, n-4), size=(n+4)//5)
        ids = np.concatenate([np.arange(s, min(s+5, n)) for s in starts])[:n]
        samples.append(float(a[ids, 0].sum()/a[ids, 1].sum()))
    return {"gain": float(left.gain.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "simultaneous_15_ci": np.quantile(samples, [.05/30, 1-.05/30]).tolist(),
            "dates": n, "n": len(left), "blocks": "5 consecutive sessions", "interpretation": "post_selection_development_diagnostic_not_independent_inference"}


def gate(root, round_number):
    summary = read(root/f"round_{round_number:02d}/summary.json")
    checks = {}
    passing = []
    for route, chosen in summary.items():
        failures = []
        groups = []
        for j, fold in enumerate(PROTOCOL["folds"][:2]):
            path = Path(chosen["incumbent_paths"][j])
            r = read(path/"result.json")["results"]["eval"]
            basepath = root/"round_00"/route/fold["id"]
            b = read(basepath/"result.json")["results"]["eval"]
            for g in GROUPS:
                m = r["groups"][g]["3d_5pct"]
                h = r["history_groups"][g]["3d_5pct"]
                r0 = b["groups"][g]["3d_5pct"]
                c = {"group": g, "fold": fold["id"], "r0_gain": 1-m["brier"]/r0["brier"],
                     "history_gain": 1-m["brier"]/h["brier"], "mean_gap": m["mean_gap"],
                     "within_auc": m["within_auc"], "dates": m["dates"], "n": m["n"]}
                groups.append(c)
                req = PROTOCOL["gate"]
                bad = []
                if c["r0_gain"] < req["relative_brier_gain"]: bad.append("R0_gain")
                if c["history_gain"] < req["relative_brier_gain"]: bad.append("stock_prior_gain")
                if c["mean_gap"] > req["max_mean_calibration_gap"]: bad.append("calibration_gap")
                if c["within_auc"] is None or c["within_auc"] < req["min_within_stock_auc"]: bad.append("within_stock_timing")
                if c["dates"] < req["min_dates_each_fold"] or c["n"] < req["min_rows_each_group_fold"]: bad.append("sample_size")
                if bad:
                    failures.append({"fold": fold["id"], "group": g, "failed": bad})
        # Only spend bootstrap compute if every point-estimate prerequisite passes.
        if not failures:
            for j, fold in enumerate(PROTOCOL["folds"][:2]):
                frame = pd.read_parquet(Path(chosen["incumbent_paths"][j])/"eval.parquet")
                base = pd.read_parquet(root/"round_00"/route/fold["id"]/"eval.parquet")
                history = frame.copy()
                history["p_3d_5pct"] = history.history_3d_5pct
                for g, syms in GROUPS.items():
                    for label, ref in (("R0", base), ("stock_prior", history)):
                        ci = paired_interval(frame, ref, syms)
                        if ci["simultaneous_15_ci"][0] <= 0:
                            failures.append({"fold": fold["id"], "group": g, "failed": [label+"_paired_interval"]})
        checks[route] = {"passed": not failures, "failures": failures, "groups": groups}
        if not failures:
            passing.append(route)
    result = {"round": round_number, "eligible_for_reserved_confirmation": passing, "checks": checks}
    write(root/f"round_{round_number:02d}/gate.json", result)
    return result


def report(root):
    summaries = sorted(root.glob("round_*/summary.json"))
    records = []
    for path in summaries:
        number = int(path.parent.name.split("_")[1])
        sm = read(path)
        gate(root, number)
        for route, v in sm.items():
            for fold in PROTOCOL["folds"][:2]:
                result = read(path.parent/route/fold["id"]/"result.json")
                for group in GROUPS:
                    for target in TARGETS:
                        m = result["results"]["eval"]["groups"][group][target]
                        records.append({"round": number, "route": route, "fold": fold["id"], "group": group, "target": target,
                                        "mechanism": v["pugh"]["chosen"], "accepted": v["accepted"],
                                        **{k: m[k] for k in ("n", "dates", "stocks", "brier", "logloss", "observed", "predicted", "mean_gap", "auc", "within_auc")}})
    pd.DataFrame(records).to_csv(HERE/"round_metrics.csv", index=False)
    lines = ["# 十轮开发记录", "", "主指标是三组×两开发窗口等权Brier，越低越好。✓表示本轮候选进入该路线当前最佳配方；回退仍保留全部结果。", "", "| 轮次 | 路线 | 本轮机制 | 候选Brier | 当前保留轮次 | 决定 |", "|---|---|---|---:|---:|---|"]
    for path in summaries:
        n = int(path.parent.name.split("_")[1])
        for route, v in read(path).items():
            lines.append(f"| R{n} | {ROUTES[route]} | {v['pugh']['chosen']} | {v['candidate_score']:.4f} | R{v['incumbent_round']} | {'✓保留' if v['accepted'] else '回退'} |")
    lines += ["", "## 每轮各组的变化", "", "两开发窗口等权平均；展示该轮实际候选，不把回退后的旧结果冒充新训练结果。", "", "| 轮次 | 路线 | 芯片Brier | 光通信Brier | 存储Brier |", "|---|---|---:|---:|---:|"]
    primary = pd.DataFrame(records).query("target == '3d_5pct'")
    for (n, r), values in primary.groupby(["round", "route"], sort=False):
        vals = values.groupby("group").brier.mean()
        lines.append(f"| R{n} | {ROUTES[r]} | {vals['chips']:.4f} | {vals['optics']:.4f} | {vals['storage']:.4f} |")
    (HERE/"ROUNDS.md").write_text("\n".join(lines)+"\n")
    if not (root/"final_complete.json").exists():
        return
    selected = read(root/"final_selection.json")
    final_records, probabilities, audit = [], [], []
    report_lines = ["# 三组预报能力 v8 结果", "", "全部结果为已暴露2026历史的开发期回放，不是独立验证；触及概率不是交易盈利概率。", "", "三组名单、10轮上限及门槛在首次拟合前冻结。五条路线各完成10轮候选、两开发窗口；最后按开发结果固定配方，在8月24日至9月17日保留段分别复验。", "", "## 冻结配方在保留段的结果", "", "种子3566；Brier相对各自同口径R0。均值偏差是预测率减实际率；同股AUC剔除了仅靠股票间差异得到的排序能力。", "", "| 路线/保留轮次 | 群组 | 样本/日期 | R0 Brier | 新Brier | 改善 | 实际率 | 预测率 | 同股AUC |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    subgroup = {**GROUPS, **{"optics_"+k: v for k, v in PROTOCOL["optics_subgroups"].items()}}
    for route in PROTOCOL["routes"]:
        baseline = pd.read_parquet(root/"final"/route/"baseline/eval.parquet")
        r0 = read(root/"final"/route/"baseline/result.json")
        for seed in (PROTOCOL["seed"], PROTOCOL["confirmation_seed"]):
            path = root/"final"/route/f"selected_{seed}"
            f = pd.read_parquet(path/"eval.parquet")
            y = f[["y_"+t for t in TARGETS]].to_numpy()
            p = f[["p_"+t for t in TARGETS]].to_numpy()
            m = metrics(f, y, p, subgroup)
            stock_metrics = metrics(f, y, p, {s: [s] for s in sorted(set(sum(GROUPS.values(), [])))})
            for group, targets in {**m, **stock_metrics}.items():
                for target, value in targets.items():
                    probabilities.append({"route": route, "seed": seed, "group": group, "target": target,
                                          **{k: value[k] for k in ("n", "dates", "observed", "predicted", "brier", "logloss", "auc", "within_auc", "mean_gap")}})
            for g, syms in GROUPS.items():
                value = m[g]["3d_5pct"]
                b = r0["results"]["eval"]["groups"][g]["3d_5pct"]
                hist = f.copy()
                hist["p_3d_5pct"] = hist.history_3d_5pct
                final_records.append({"route": route, "round": selected[route]["round"], "seed": seed, "group": g,
                    "relative_gain": 1-value["brier"]/b["brier"], "r0_brier": b["brier"], **value,
                    "paired_R0": paired_interval(f, baseline, syms), "paired_history": paired_interval(f, hist, syms)})
                if seed == PROTOCOL["seed"]:
                    wa = f"{value['within_auc']:.3f}" if value["within_auc"] is not None else "—"
                    report_lines.append(f"| {ROUTES[route]} / R{selected[route]['round']} | {LABELS[g]} | {value['n']}/{value['dates']} | {b['brier']:.4f} | {value['brier']:.4f} | {(1-value['brier']/b['brier']):+.1%} | {value['observed']:.1%} | {value['predicted']:.1%} | {wa} |")
            result = read(path/"result.json")
            audit.append({"route": route, "seed": seed, "replay_error": result["fit"]["replay_max_error"], "seconds": result["seconds"], "fit": {k:v for k,v in result["fit"].items() if k != "trace"}})
    write(HERE/"final_metrics.json", final_records)
    pd.DataFrame(probabilities).to_csv(HERE/"probabilities.csv", index=False)
    report_lines += ["", "## 开发选择和复验", "", "| 路线 | 保留轮次 | 开发Brier | 配方 |", "|---|---:|---:|---|"]
    for r, v in selected.items():
        report_lines.append(f"| {ROUTES[r]} | R{v['round']} | {v['score']:.4f} | {', '.join(v['recipe']) or 'R0'} |")
    report_lines += ["", "全部逐轮数据见 [ROUNDS.md](ROUNDS.md) / [round_metrics.csv](round_metrics.csv)。九目标、三组、光通信子组和21只股票的两种子概率见 [probabilities.csv](probabilities.csv)。日期块配对区间见 [final_metrics.json](final_metrics.json)，区间属于开发诊断。", "", "## 数据与能力边界", "", "新增7只光产业链股票，当前快照股票池回溯共108股；扩展训练历史仍不能凭空补齐126个完整历史日。行情可用时间沿用归档结束后1秒假设，没有实测到达时延。无群路线不输入群或其他股票价格；无日级路线不输入日线，若配方包含prior，则明确额外使用本股成熟标签先验。行业标签只用于用户指定的评价范围和训练权重，没有冒充历史行业PIT。", "", "三组有3只交叉股票，组评价按各组等权，联合股票数为21，不把重叠成员计成独立证据。所有评估使用相同完整成熟样本。此前历史全部已暴露，本轮reserved仅隔离本轮选型；不能宣称准生产或真实交易收益。"]
    (HERE/"REPORT.md").write_text("\n".join(report_lines)+"\n")
    write(HERE/"AUDIT.json", {"rounds": len(summaries)-1, "routes": 5, "final_replays": audit,
          "final_source": str(root), "all_replays_exact": all(a["replay_error"] <= 1e-7 for a in audit),
          "final_gate": read(summaries[-1].parent/"gate.json"), "independent_validation": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--gate-round", type=int)
    args = parser.parse_args()
    if args.gate_round is not None:
        print(json.dumps(gate(args.root.resolve(), args.gate_round)), flush=True)
    else:
        report(args.root.resolve())
