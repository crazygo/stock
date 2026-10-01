"""Seven frozen LightGBM settings for the 8-shape segment model.

The 30 sessions 2026-08-18 through 2026-09-29 were already scored at 12/27
by the 40-tree model. Each fixed setting is reported on that window. The
setting used for a session is chosen on the previous 20 sessions that contain
a finished shape, using only labels whose final session is already complete.
A higher fixed-setting score on these 30 sessions does not replace the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.open_0940_touch_v1.backtest_30d import MIN_TRAIN, OUT_DIR
from research.after_open_3d5pct.models.open_0940_touch_v1.shape_segment import (
    FEATURES,
    PRIOR_PATH,
    build_frame,
)

HTML_PATH = Path(__file__).resolve().parent / "shape_param_sweep.html"
FRAME_PATH = OUT_DIR / "shape_frame.parquet"
VAL_SESSIONS = 20
MIN_VAL_DAYS = 8
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
        "name": "min_child_40",
        "label": "每叶至少 40 行",
        "n_estimators": 40,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 40,
        "learning_rate": 0.05,
        "reg_lambda": 10.0,
        "colsample_bytree": 1.0,
    },
    {
        "name": "slow",
        "label": "慢学习",
        "n_estimators": 300,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 20,
        "learning_rate": 0.02,
        "reg_lambda": 20.0,
        "colsample_bytree": 1.0,
    },
    {
        "name": "weak",
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
        "min_child_samples": 30,
        "learning_rate": 0.05,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.5,
    },
]


def _auc(label: np.ndarray, score: np.ndarray) -> float | None:
    order = np.argsort(score, kind="mergesort")
    y = label[order]
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return None
    ranks = np.empty(len(y), dtype=float)
    ranks[order] = np.arange(1, len(y) + 1)
    sorted_score = score[order]
    start = 0
    while start < len(y):
        stop = start + 1
        while stop < len(y) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    sum_pos = float(ranks[label == 1].sum())
    return (sum_pos - pos * (pos + 1) / 2) / (pos * neg)


def _group(name: str) -> str:
    if name.startswith("shape_"):
        return "shape"
    if name.startswith("window_"):
        return "window"
    if name.startswith("prefix_"):
        return "prefix"
    if name == "atr_20d_pct":
        return "atr"
    if name == "minutes_from_open":
        return "clock"
    return "other"


def _train_before(frame: pd.DataFrame, day: str) -> pd.DataFrame:
    return frame.loc[frame["ready"].eq(True) & frame["final_day"].astype(str).lt(day) & frame["y"].notna()]


def _fit(train: pd.DataFrame, config: dict):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=config["n_estimators"],
        num_leaves=config["num_leaves"],
        max_depth=config["max_depth"],
        min_child_samples=config["min_child_samples"],
        learning_rate=config["learning_rate"],
        reg_lambda=config["reg_lambda"],
        colsample_bytree=config["colsample_bytree"],
        n_jobs=1,
        verbosity=-1,
        random_state=3566,
    )
    y = train["y"].to_numpy(int)
    model.fit(train[FEATURES], y)
    trained = model.predict_proba(train[FEATURES])[:, 1]
    return model, y, trained


def _score_day(model, train_score: np.ndarray, today: pd.DataFrame) -> dict:
    scored = today.copy()
    scored["score"] = model.predict_proba(scored[FEATURES])[:, 1]
    scored = scored.sort_values(["score", "symbol"], ascending=[False, True])
    pick = scored.iloc[0]
    pick_hit = None if not bool(pick["ready"]) else int(pick["y"] == 1)
    threshold = float(np.quantile(train_score, 0.80))
    chosen = scored.loc[scored["score"].ge(threshold)]
    selected_hits = []
    for item in chosen.itertuples(index=False):
        selected_hits.append(None if not bool(item.ready) else int(item.y == 1))
    return {
        "events": int(len(scored)),
        "pick": {
            "symbol": str(pick["symbol"]),
            "shape": str(pick["shape"]),
            "probability": float(pick["score"]),
            "hit": pick_hit,
        },
        "selected_hits": selected_hits,
    }


def _rate(hits: list[int | None]) -> dict:
    mature = [item for item in hits if item is not None]
    got = sum(int(item) for item in mature)
    return {
        "forecasts": len(hits),
        "mature": len(mature),
        "hits": got,
        "rate": None if not mature else got / len(mature),
    }


def _load_frame() -> pd.DataFrame:
    if FRAME_PATH.exists():
        frame = pd.read_parquet(FRAME_PATH)
        print(json.dumps({"cached_rows": int(len(frame))}), flush=True)
        return frame
    frame = build_frame()
    FRAME_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FRAME_PATH, index=False)
    print(json.dumps({"cached_rows": int(len(frame)), "wrote": str(FRAME_PATH)}), flush=True)
    return frame


def _run_config(frame: pd.DataFrame, days: list[str], config: dict) -> dict:
    gain = np.zeros(len(FEATURES))
    train_aucs: list[float] = []
    day_rows = []
    for offset, day in enumerate(days, start=1):
        train = _train_before(frame, day)
        today = frame.loc[frame["session_date"].eq(day)].copy()
        if len(train) < MIN_TRAIN or train["y"].nunique() < 2 or today.empty:
            day_rows.append({"session_date": day, "events": int(len(today)), "pick": None, "selected_hits": []})
            print(json.dumps({"config": config["name"], "day": day, "step": offset, "skipped": True}), flush=True)
            continue
        model, y, trained = _fit(train, config)
        score = _auc(y, trained)
        if score is not None:
            train_aucs.append(float(score))
        gain += model.booster_.feature_importance(importance_type="gain")
        day_rows.append({"session_date": day, **_score_day(model, trained, today)})
        print(json.dumps({"config": config["name"], "day": day, "step": offset}), flush=True)
    groups = {name: 0.0 for name in ("shape", "window", "prefix", "atr", "clock", "other")}
    total = float(gain.sum()) or 1.0
    for feature, value in zip(FEATURES, gain):
        groups[_group(feature)] += float(value) / total
    order = np.argsort(gain)[::-1][:8]
    top = [
        {"feature": FEATURES[int(pos)], "gain_share": float(gain[int(pos)] / total)}
        for pos in order
        if gain[int(pos)] > 0
    ]
    top_hits = [None if row["pick"] is None else row["pick"]["hit"] for row in day_rows]
    selected_hits: list[int | None] = []
    for row in day_rows:
        selected_hits.extend(row["selected_hits"])
    return {
        "name": config["name"],
        "label": config["label"],
        "params": {key: value for key, value in config.items() if key not in ("name", "label")},
        "mean_train_auc": None if not train_aucs else float(np.mean(train_aucs)),
        "top1": _rate(top_hits),
        "selected": _rate(selected_hits),
        "gain_groups": groups,
        "top_features": top,
        "days": day_rows,
    }


def _validation_choice(frame: pd.DataFrame, day: str, history: list[str]) -> tuple[str, dict]:
    earlier = [item for item in history if item < day]
    window = earlier[-VAL_SESSIONS:]
    detail = {"fallback": "current", "val_days": 0, "scores": []}
    if len(window) < MIN_VAL_DAYS:
        return "current", detail
    val_start = window[0]
    select_train = _train_before(frame, val_start)
    validation = frame.loc[
        frame["session_date"].isin(window)
        & frame["ready"].eq(True)
        & frame["final_day"].astype(str).lt(day)
        & frame["y"].notna()
    ]
    mature_days = int(validation["session_date"].nunique())
    detail["val_days"] = mature_days
    if len(select_train) < MIN_TRAIN or select_train["y"].nunique() < 2 or mature_days < MIN_VAL_DAYS:
        return "current", detail
    best: tuple[float, float, float, int] | None = None
    chosen = "current"
    for index, config in enumerate(CONFIGS):
        model, _y, trained = _fit(select_train, config)
        scored = validation.copy()
        scored["score"] = model.predict_proba(scored[FEATURES])[:, 1]
        threshold = float(np.quantile(trained, 0.80))
        top_hits = []
        for _session, group in scored.groupby("session_date", sort=True):
            group = group.sort_values(["score", "symbol"], ascending=[False, True])
            top_hits.append(int(group.iloc[0]["y"] == 1))
        chosen_rows = scored.loc[scored["score"].ge(threshold), "y"]
        top_rate = sum(top_hits) / len(top_hits)
        selected_rate = -1.0 if chosen_rows.empty else float((chosen_rows == 1).mean())
        capacity = float(config["n_estimators"] * config["num_leaves"])
        rank = (top_rate, selected_rate, -capacity, -index)
        detail["scores"].append(
            {
                "name": config["name"],
                "top_rate": top_rate,
                "selected_rate": selected_rate,
                "mature_days": len(top_hits),
            }
        )
        if best is None or rank > best:
            best = rank
            chosen = config["name"]
    detail["fallback"] = None
    return chosen, detail


def _run_nested(frame: pd.DataFrame, days: list[str], cells: list[dict]) -> dict:
    by_name = {cell["name"]: {row["session_date"]: row for row in cell["days"]} for cell in cells}
    history = sorted(str(item) for item in frame["session_date"].unique())
    rows = []
    for offset, day in enumerate(days, start=1):
        name, detail = _validation_choice(frame, day, history)
        stored = by_name[name][day]
        pick = stored["pick"]
        rows.append(
            {
                "session_date": day,
                "config": name,
                "val_days": detail["val_days"],
                "events": stored["events"],
                "pick": pick,
                "selected_hits": stored["selected_hits"],
                "validation": detail["scores"],
            }
        )
        print(
            json.dumps(
                {
                    "nested": day,
                    "step": offset,
                    "config": name,
                    "hit": None if pick is None else pick["hit"],
                }
            ),
            flush=True,
        )
    top_hits = [None if row["pick"] is None else row["pick"]["hit"] for row in rows]
    selected_hits: list[int | None] = []
    for row in rows:
        selected_hits.extend(row["selected_hits"])
    counts: dict[str, int] = {config["name"]: 0 for config in CONFIGS}
    for row in rows:
        counts[row["config"]] += 1
    return {"top1": _rate(top_hits), "selected": _rate(selected_hits), "counts": counts, "days": rows}


def _segment_base(frame: pd.DataFrame, days: list[str]) -> dict:
    test = frame.loc[frame["session_date"].isin(days) & frame["ready"].eq(True) & frame["y"].notna()]
    hits = int(test["y"].sum()) if len(test) else 0
    return {"events": int(len(test)), "hits": hits, "rate": None if test.empty else hits / len(test)}


def _pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.1f}%"


def _cell_html(stat: dict) -> str:
    return f"<td>{stat['hits']}/{stat['mature']}<br>{_pct(stat['rate'])}</td>"


def render(cells: list[dict], nested: dict, base: dict) -> str:
    heads = "".join(f"<th>{cell['label']}</th>" for cell in cells)
    top_row = "".join(_cell_html(cell["top1"]) for cell in cells)
    selected_row = "".join(_cell_html(cell["selected"]) for cell in cells)
    auc_row = "".join(f"<td>{_pct(cell['mean_train_auc'])}</td>" for cell in cells)
    group_rows = []
    for key, title in (
        ("shape", "8 类图形"),
        ("window", "图形区间"),
        ("prefix", "区间前 24 根"),
        ("atr", "20 日振幅"),
        ("clock", "离开盘的分钟"),
    ):
        group_rows.append(
            "<tr><th>{}</th>{}</tr>".format(
                title, "".join(f"<td>{_pct(cell['gain_groups'][key])}</td>" for cell in cells)
            )
        )
    params = []
    for cell in cells:
        spec = cell["params"]
        params.append(
            f"<li><b>{cell['label']}</b>：{spec['n_estimators']} 棵，深度 {spec['max_depth']}，"
            f"{spec['num_leaves']} 叶，min_child_samples={spec['min_child_samples']}，"
            f"学习率 {spec['learning_rate']}，reg_lambda={spec['reg_lambda']}，"
            f"colsample_bytree={spec['colsample_bytree']}。"
            f"每天留一只 {cell['top1']['hits']}/{cell['top1']['mature']} = {_pct(cell['top1']['rate'])}，"
            f"训练分前 20% {cell['selected']['hits']}/{cell['selected']['mature']} = {_pct(cell['selected']['rate'])}。</li>"
        )
    labels = {config["name"]: config["label"] for config in CONFIGS}
    usage = "，".join(f"{labels[name]} {count} 天" for name, count in nested["counts"].items() if count)
    day_rows = []
    for row in nested["days"]:
        pick = row["pick"]
        if pick is None:
            day_rows.append(f"<tr><th>{row['session_date']}</th><td>{labels[row['config']]}</td><td>无预报</td></tr>")
            continue
        word = "未到期" if pick["hit"] is None else ("达成" if pick["hit"] else "未达成")
        day_rows.append(
            f"<tr><th>{row['session_date']}</th><td>{labels[row['config']]}</td>"
            f"<td>{pick['symbol']} · {pick['shape']} · {_pct(pick['probability'])} · {word}</td></tr>"
        )
    nested_top = nested["top1"]
    nested_selected = nested["selected"]
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>图形段模型的七组参数</title>
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
<h1>图形段模型，七组 LightGBM 参数</h1>
<p>样本仍是 8 种图形走完的区间加上前面 24 根。标签仍是其后常规盘最高价，收到决策日之后第 2 个交易日，碰到 +5%。考试日仍是 2026-08-18 至 2026-09-29。这 30 天里全部到期图形段是 {base['hits']}/{base['events']} = {_pct(base['rate'])}。</p>
<p>按考试日之前最近 {VAL_SESSIONS} 个出现过图形的交易日选择参数，再只用更早的样本重训并预报当天。这样选出来的结果：每天留一只 {nested_top['hits']}/{nested_top['mature']} = {_pct(nested_top['rate'])}，训练分前 20% {nested_selected['hits']}/{nested_selected['mature']} = {_pct(nested_selected['rate'])}。各天用到的参数：{usage}。</p>
<p>下表是七组参数从头到尾固定时的分数。固定参数谁更高，不拿来替换模型，因为这些分数来自已经看过的 30 天。</p>
<ul>{''.join(params)}</ul>
<h2>到期预报达成</h2>
<div class="scroll">
<table>
<thead><tr><th>口径</th>{heads}</tr></thead>
<tbody>
<tr><th>每天留一只</th>{top_row}</tr>
<tr><th>训练分前 20%</th>{selected_row}</tr>
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
<h2>按更早 20 天选出的参数，每天留一只</h2>
<table>
<thead><tr><th>交易日</th><th>当天参数</th><th>最高概率</th></tr></thead>
<tbody>{''.join(day_rows)}</tbody>
</table>
</main>
</body>
</html>
"""


def main() -> None:
    prior = json.loads(PRIOR_PATH.read_text())
    days = [report["session_date"] for report in prior["reports"]]
    frame = _load_frame()
    base = _segment_base(frame, days)
    print(json.dumps({"segment_base": base, "rows": int(len(frame))}), flush=True)
    cells = []
    out = OUT_DIR / "shape_param_sweep.json"
    for config in CONFIGS:
        cell = _run_config(frame, days, config)
        cells.append(cell)
        out.write_text(json.dumps({"base": base, "cells": cells}, ensure_ascii=False))
        print(
            json.dumps(
                {
                    "done": config["name"],
                    "top1": cell["top1"],
                    "selected": cell["selected"],
                    "train_auc": cell["mean_train_auc"],
                }
            ),
            flush=True,
        )
    nested = _run_nested(frame, days, cells)
    payload = {"base": base, "cells": cells, "nested": nested}
    out.write_text(json.dumps(payload, ensure_ascii=False))
    HTML_PATH.write_text(render(cells, nested, base))
    print(json.dumps({"nested": nested["top1"], "selected": nested["selected"], "counts": nested["counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
