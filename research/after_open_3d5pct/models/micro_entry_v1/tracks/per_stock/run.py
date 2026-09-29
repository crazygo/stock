"""One frozen tree per name. Features, label, window, and purge stay put."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.after_open_3d5pct.models.micro_entry_v1.tracks.common import (
    BASELINE_META,
    BASELINE_SCORED,
    FEATURES,
    annotate_current,
    beats,
    first_trigger,
    load_frame,
    pack,
    walk,
)
from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT

FIT = ["AMD", "MRVL", "MU", "LITE", "ALAB"]
ABSENT = ["SMTC", "SOXS", "SOXX"]
OUT = ROOT / "research/after_open_3d5pct/runs/micro_entry_v1/tracks/per_stock/result.json"
REPORT = Path(__file__).resolve().parent / "REPORT.md"
ABSENT_NOTE = (
    "核准的历史加载器没有这些名字 2024 年起的常规 5 分钟线，"
    "八周训练加两周考试填不满。"
)


def _check_baseline() -> dict:
    meta = json.loads(BASELINE_META.read_text())
    precision = meta["test_first"]["precision"]
    if precision is None or not (0.462 <= float(precision) <= 0.473):
        raise SystemExit(
            f"baseline test_first.precision {precision} is outside 0.462-0.473"
        )
    return meta


def _pooled_on_name(scored: pd.DataFrame, symbol: str) -> dict:
    out = {}
    for split, key in (("test", "test_first"), ("validation", "validation_first")):
        part = scored.loc[(scored["symbol"] == symbol) & (scored["split"] == split)]
        out[key] = first_trigger(part)
    return out


def _reached_90(own: dict) -> bool:
    for key in ("test_first", "validation_first"):
        block = own[key]
        precision = block.get("precision")
        if precision is None or block.get("buys", 0) < 15 or precision < 0.90:
            return False
    return True


def _fit_one(frame: pd.DataFrame, scored: pd.DataFrame, symbol: str) -> dict:
    subframe = frame.loc[frame["symbol"] == symbol].copy()
    done = walk(subframe, list(FEATURES), annotate_current)
    own = pack(done)
    del done
    reference = _pooled_on_name(scored, symbol)
    return {
        "symbol": symbol,
        "rows": int(len(subframe)),
        "in_frame": True,
        "own": own,
        "pooled_on_name": reference,
        "beats": beats(own, reference),
        "reached_90": _reached_90(own),
    }


def _pct(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{100 * float(value):.{digits}f}%"


def _digits(left, right) -> int:
    if left is None or right is None or left == right:
        return 1
    if round(100 * float(left), 1) == round(100 * float(right), 1):
        return 2
    return 1


def _cell(block: dict, other: dict | None = None) -> str:
    if not block or block.get("precision") is None:
        buys = 0 if not block else block.get("buys", 0)
        return f"{buys}笔 / —"
    digits = _digits(block.get("precision"), None if other is None else other.get("precision"))
    return f"{block['buys']}笔 / {_pct(block['precision'], digits)}"


def _base(block: dict) -> str:
    return _pct(block.get("first_clock_base"))


def _gap(own: dict, ref: dict) -> str:
    if own.get("precision") is None or ref.get("precision") is None:
        return f"无命中率（{own.get('buys', 0)}笔对{ref.get('buys', 0)}笔）"
    delta = (float(own["precision"]) - float(ref["precision"])) * 100
    shown = f"{delta:+.1f}" if abs(delta) >= 0.05 else f"{delta:+.2f}"
    return f"{shown}点（{own['buys']}笔对{ref['buys']}笔）"


def _bottleneck(records: list[dict]) -> str:
    parts = []
    for record in records:
        own = record["own"]
        ref = record["pooled_on_name"]
        parts.append(
            f"{record['symbol']} 下一周 {_gap(own['test_first'], ref['test_first'])}"
            f"、再下一周 {_gap(own['validation_first'], ref['validation_first'])}"
        )
    beaters = [record["symbol"] for record in records if record["beats"]]
    if beaters:
        tail = "、".join(beaters) + "两周都更高，其余没有"
    else:
        tail = "没有一只两周都更高"
    return "瓶颈：" + "；".join(parts) + "。" + tail + "，上界就是本股这一棵树。"


def _report(meta: dict, records: list[dict], missing: list[dict], bottleneck: str) -> str:
    test = meta["test_first"]
    validation = meta["validation_first"]
    lines = [
        "# 一股一棵树",
        "",
        "每只可拟合的股票只用自己的 2 小时窗口。树、六个特征、标签、窗口和八周剔除都不改。"
        "买入线是这只股票自己训练分数的 80 分位。当天只保留第一次越过这条线的窗口。",
        "",
        f"全体模型合在一起是下一周 {test['buys']} 笔、{_pct(test['precision'])}，"
        f"再下一周 {validation['buys']} 笔、{_pct(validation['precision'])}。"
        "单股不跟这个合计数比，只跟全体模型落在这一只股票上的第一次越线比。",
        "",
        "| 股票 | 下一周买入与命中率 | 本股全体模型同一周 | 再下一周买入与命中率 | 本股全体模型那一周 | 是否两周都更高 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        own = record["own"]
        ref = record["pooled_on_name"]
        lines.append(
            "| {symbol} | {a} | {b} | {c} | {d} | {beat} |".format(
                symbol=record["symbol"],
                a=_cell(own["test_first"], ref["test_first"]),
                b=_cell(ref["test_first"], own["test_first"]),
                c=_cell(own["validation_first"], ref["validation_first"]),
                d=_cell(ref["validation_first"], own["validation_first"]),
                beat="是" if record["beats"] else "否",
            )
        )
    for item in missing:
        lines.append(f"| {item['symbol']} | 无 | 无 | 无 | 无 | 否 |")
    lines.extend(["", "本股首个窗口的自然命中率（与买入线无关）："])
    for record in records:
        own = record["own"]
        ref = record["pooled_on_name"]
        lines.append(
            f"- {record['symbol']} 下一周 {_base(own['test_first'])}，"
            f"再下一周 {_base(own['validation_first'])}。"
            f"全体模型落在该股上的首窗基率是 {_base(ref['test_first'])} / {_base(ref['validation_first'])}。"
        )
    alab = next(record for record in records if record["symbol"] == "ALAB")
    others = next(record for record in records if record["symbol"] != "ALAB")
    alab_val = alab["own"]["last_triplet"]["validation"]
    other_val = others["own"]["last_triplet"]["validation"]
    lines.extend([
        "",
        "ALAB 在帧里少一段交易日，单股滑窗按它自己的交易日切周。"
        f"它最后一组验证是 {alab_val[0]} 至 {alab_val[1]}，"
        f"其余四只和全体模型停在 {other_val[0]} 至 {other_val[1]}。"
        "表上 ALAB 的两周因此不是同一组日子。对照仍是全体模型落在 ALAB 行上的第一次越线。",
        "",
        "SMTC、SOXS、SOXX 不在这张 2 小时帧里。" + ABSENT_NOTE,
        "",
        bottleneck,
        "",
    ])
    if any(record["reached_90"] for record in records):
        names = "、".join(record["symbol"] for record in records if record["reached_90"])
        lines.append(f"{names} 两周都至少 15 笔且命中率达到 90%。")
    else:
        lines.append("没有一只同时达到 15 笔和 90%。")
    lines.extend([
        "",
        "逐行数字在 `research/after_open_3d5pct/runs/micro_entry_v1/tracks/per_stock/result.json`，该目录不进 Git。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    meta = _check_baseline()
    frame = load_frame()
    present = set(frame["symbol"].astype(str).unique())
    scored = pd.read_parquet(
        BASELINE_SCORED,
        columns=["symbol", "session_date", "decision_at", "y_3d_5pct", "score", "level", "split"],
    )
    records = []
    for symbol in FIT:
        if symbol not in present:
            raise SystemExit(f"{symbol} is missing from the 2-hour frame")
        print(json.dumps({"fit": symbol, "rows": int((frame["symbol"] == symbol).sum())}), flush=True)
        record = _fit_one(frame, scored, symbol)
        records.append(record)
        print(json.dumps({
            "symbol": symbol,
            "beats": record["beats"],
            "test_buys": record["own"]["test_first"]["buys"],
            "test_precision": record["own"]["test_first"]["precision"],
            "validation_buys": record["own"]["validation_first"]["buys"],
            "validation_precision": record["own"]["validation_first"]["precision"],
            "pooled_test_precision": record["pooled_on_name"]["test_first"]["precision"],
            "pooled_validation_precision": record["pooled_on_name"]["validation_first"]["precision"],
        }, default=str), flush=True)
    missing = []
    for symbol in ABSENT:
        missing.append({
            "symbol": symbol,
            "in_frame": symbol in present,
            "fitted": False,
            "note": ABSENT_NOTE,
        })
    bottleneck = _bottleneck(records)
    payload = {
        "track": "per_stock",
        "baseline_test_first_precision": meta["test_first"]["precision"],
        "names": records,
        "not_fit": missing,
        "bottleneck": bottleneck,
        "any_reached_90": any(record["reached_90"] for record in records),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    REPORT.write_text(_report(meta, records, missing, bottleneck))
    print(json.dumps({"wrote": str(OUT), "report": str(REPORT), "bottleneck": bottleneck}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
