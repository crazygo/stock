"""Sensitivity of the per-bar 09:40 model to a frozen set of LightGBM parameters.

The 30 trading days were already scored. Each cell is reported. None is selected
as a replacement model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.open_0940_touch_v1.backtest_30d_bars import (
    CONTEXT,
    FEATURES,
    OUT_DIR,
    TARGETS,
    build_frame,
)
from research.after_open_3d5pct.models.open_0940_touch_v1.backtest_30d import MIN_TRAIN

FRAME_PATH = OUT_DIR / "bar_frame.parquet"
PRIOR_PATH = OUT_DIR / "backtest_30d_bars.json"
HTML_PATH = Path(__file__).resolve().parent / "param_sweep.html"

CONFIGS = [
    {
        "name": "current",
        "label": "当前",
        "n_estimators": 40,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 10,
        "learning_rate": 0.05,
        "reg_lambda": 10.0,
        "colsample_bytree": 1.0,
    },
    {
        "name": "trees_200",
        "label": "200 棵树",
        "n_estimators": 200,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 10,
        "learning_rate": 0.05,
        "reg_lambda": 10.0,
        "colsample_bytree": 1.0,
    },
    {
        "name": "depth_6",
        "label": "深度 6",
        "n_estimators": 100,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 10,
        "learning_rate": 0.05,
        "reg_lambda": 10.0,
        "colsample_bytree": 1.0,
    },
    {
        "name": "lambda_1",
        "label": "更弱正则",
        "n_estimators": 40,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 5,
        "learning_rate": 0.05,
        "reg_lambda": 1.0,
        "colsample_bytree": 1.0,
    },
    {
        "name": "wide",
        "label": "深树加列抽样",
        "n_estimators": 200,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "learning_rate": 0.05,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.2,
    },
]


def _auc(label: np.ndarray, score: np.ndarray) -> float | None:
    order = np.argsort(score)
    y = label[order]
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return None
    ranks = np.empty(len(y), dtype=float)
    ranks[order] = np.arange(1, len(y) + 1)
    # Recompute ranks on the original order. Ties get average ranks.
    sorted_score = score[order]
    start = 0
    while start < len(y):
        stop = start + 1
        while stop < len(y) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        if stop - start > 1:
            average = 0.5 * (start + 1 + stop)
            ranks[order[start:stop]] = average
        start = stop
    sum_pos = float(ranks[label == 1].sum())
    return (sum_pos - pos * (pos + 1) / 2) / (pos * neg)


def _group(name: str) -> str:
    if name in CONTEXT:
        return "context"
    if name.startswith("d0_") and name.endswith("_range"):
        return "d0_range"
    if name.startswith("d0_"):
        return "d0_other"
    if name.endswith("_vol"):
        return "bar_vol"
    if name.endswith("_range"):
        return "bar_range"
    if name.endswith("_ret"):
        return "bar_ret"
    if name.endswith("_gap"):
        return "bar_gap"
    return "other"


def _load_frame() -> pd.DataFrame:
    if FRAME_PATH.exists():
        frame = pd.read_parquet(FRAME_PATH)
        print(json.dumps({"cached_rows": int(len(frame))}), flush=True)
        return frame
    frame = build_frame()
    FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FRAME_PATH, index=False)
    return frame


def _run_config(frame: pd.DataFrame, days: list[str], config: dict) -> dict:
    import lightgbm as lgb

    reports = []
    gain = np.zeros(len(FEATURES))
    train_aucs: list[float] = []
    for offset, day in enumerate(days, start=1):
        today = frame.loc[frame["session_date"].eq(day)].copy()
        row = {"session_date": day, "targets": {}}
        for target in TARGETS:
            key = target["id"]
            label = f"y_{key}"
            train = frame.loc[frame[f"ready_{key}"].eq(True) & frame[f"final_{key}"].astype(str).lt(day)]
            usable = train.loc[train[label].notna()]
            if len(usable) < MIN_TRAIN or usable[label].nunique() < 2 or today.empty:
                row["targets"][key] = None
                continue
            y = usable[label].to_numpy(int)
            model = lgb.LGBMClassifier(
                n_estimators=config["n_estimators"],
                num_leaves=config["num_leaves"],
                max_depth=config["max_depth"],
                min_child_samples=config["min_child_samples"],
                learning_rate=config["learning_rate"],
                reg_lambda=config["reg_lambda"],
                colsample_bytree=config["colsample_bytree"],
                n_jobs=4,
                verbosity=-1,
                random_state=3566,
            )
            model.fit(usable[FEATURES], y)
            trained = model.predict_proba(usable[FEATURES])[:, 1]
            score = _auc(y, trained)
            if score is not None:
                train_aucs.append(float(score))
            gain += model.booster_.feature_importance(importance_type="gain")
            scored = today.copy()
            scored["score"] = model.predict_proba(scored[FEATURES])[:, 1]
            scored = scored.sort_values(["score", "symbol"], ascending=[False, True])
            pick = scored.iloc[0]
            ready = bool(pick[f"ready_{key}"])
            row["targets"][key] = {
                "symbol": str(pick["symbol"]),
                "probability": float(pick["score"]),
                "hit": None if not ready else int(pick[label] == 1),
            }
        print(json.dumps({"config": config["name"], "day": day, "step": offset, "of": len(days)}), flush=True)
        reports.append(row)
    groups = {name: 0.0 for name in ("context", "d0_range", "d0_other", "bar_vol", "bar_range", "bar_ret", "bar_gap", "other")}
    total = float(gain.sum()) or 1.0
    for feature, value in zip(FEATURES, gain):
        groups[_group(feature)] += float(value) / total
    order = np.argsort(gain)[::-1][:8]
    top = [
        {"feature": FEATURES[int(pos)], "gain_share": float(gain[int(pos)] / total)}
        for pos in order
        if gain[int(pos)] > 0
    ]
    summary = {}
    for target in TARGETS:
        picks = [report["targets"][target["id"]] for report in reports if report["targets"][target["id"]] is not None]
        mature = [item for item in picks if item["hit"] is not None]
        hits = sum(item["hit"] for item in mature)
        summary[target["id"]] = {
            "mature": len(mature),
            "hits": hits,
            "rate": None if not mature else hits / len(mature),
        }
    return {
        "name": config["name"],
        "label": config["label"],
        "params": {key: value for key, value in config.items() if key not in ("name", "label")},
        "mean_train_auc": None if not train_aucs else float(np.mean(train_aucs)),
        "summary": summary,
        "gain_groups": groups,
        "top_features": top,
        "reports": reports,
    }


def _pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def render(cells: list[dict]) -> str:
    heads = "".join(f"<th>{cell['label']}</th>" for cell in cells)
    body = []
    for target in TARGETS:
        cells_html = []
        for cell in cells:
            stat = cell["summary"][target["id"]]
            cells_html.append(f"<td>{stat['hits']}/{stat['mature']}<br>{_pct(stat['rate'])}</td>")
        body.append(f"<tr><th>{target['name']}</th>{''.join(cells_html)}</tr>")
    auc_row = "".join(f"<td>{_pct(cell['mean_train_auc'])}</td>" for cell in cells)
    group_rows = []
    for key, title in (
        ("context", "20 日波动、日收益、QQQ"),
        ("d0_range", "当天 09:35 与 09:40 的振幅"),
        ("d0_other", "当天这两根的涨跌、成交量、跳空"),
        ("bar_range", "更早每一根的振幅"),
        ("bar_vol", "更早每一根的成交量"),
        ("bar_ret", "更早每一根的涨跌"),
        ("bar_gap", "更早每一根的跳空"),
    ):
        group_rows.append(
            "<tr><th>{}</th>{}</tr>".format(
                title, "".join(f"<td>{_pct(cell['gain_groups'][key])}</td>" for cell in cells)
            )
        )
    tops = []
    for cell in cells:
        items = "，".join(f"{item['feature']} {_pct(item['gain_share'])}" for item in cell["top_features"][:5])
        tops.append(f"<p><b>{cell['label']}</b>　训练 AUC {_pct(cell['mean_train_auc'])}<br>{items}</p>")
    params = []
    for cell in cells:
        spec = cell["params"]
        params.append(
            f"<li><b>{cell['label']}</b>：{spec['n_estimators']} 棵，深度 {spec['max_depth']}，"
            f"{spec['num_leaves']} 叶，min_child_samples={spec['min_child_samples']}，"
            f"学习率 {spec['learning_rate']}，reg_lambda={spec['reg_lambda']}，"
            f"colsample_bytree={spec['colsample_bytree']}</li>"
        )
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>逐根模型的参数对照</title>
<style>
body {{ margin: 0; padding: 24px; font-family: sans-serif; color: #111827; background: #fff; }}
main {{ max-width: 1080px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d1d5db; text-align: left; padding: 8px; vertical-align: top; }}
.scroll {{ overflow-x: auto; }}
</style>
</head>
<body>
<main>
<h1>同一批逐根特征，五组 LightGBM 参数</h1>
<p>特征、09:40 决策、三个标签和 2026-08-18 至 2026-09-29 这 30 个交易日都不变。这 30 天已经看过达成率，下表只描述参数改变后的变化。</p>
<ul>{''.join(params)}</ul>
<h2>到期预报达成</h2>
<div class="scroll">
<table>
<thead><tr><th>目标</th>{heads}</tr></thead>
<tbody>
{''.join(body)}
<tr><th>训练 AUC</th>{auc_row}</tr>
</tbody>
</table>
</div>
<h2>分裂增益落在哪一类列</h2>
<div class="scroll">
<table>
<thead><tr><th>列的类别</th>{heads}</tr></thead>
<tbody>{''.join(group_rows)}</tbody>
</table>
</div>
<h2>增益最高的五列</h2>
{''.join(tops)}
</main>
</body>
</html>
"""


def main() -> None:
    prior = json.loads(PRIOR_PATH.read_text())
    days = [report["session_date"] for report in prior["reports"]]
    frame = _load_frame()
    cells = []
    out = OUT_DIR / "param_sweep.json"
    for config in CONFIGS:
        cell = _run_config(frame, days, config)
        cells.append(cell)
        out.write_text(json.dumps(cells, ensure_ascii=False))
        print(json.dumps({"done": config["name"], "summary": cell["summary"], "train_auc": cell["mean_train_auc"]}, ensure_ascii=False), flush=True)
    HTML_PATH.write_text(render(cells))
    print(json.dumps({"html": str(HTML_PATH)}), flush=True)


if __name__ == "__main__":
    main()
