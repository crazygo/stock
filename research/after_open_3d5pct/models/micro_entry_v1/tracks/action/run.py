"""Action policy on the frozen 2-hour scores. Does not refit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[6]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.after_open_3d5pct.models.micro_entry_v1.tracks.common import (  # noqa: E402
    BASELINE_META,
    BASELINE_SCORED,
    beats,
    first_trigger,
)

HERE = Path(__file__).resolve().parent
RESULT = ROOT / "research/after_open_3d5pct/runs/micro_entry_v1/tracks/action/result.json"
REPORT = HERE / "REPORT.md"
NEEDED = ["symbol", "session_date", "decision_at", "impulse", "score", "level", "split", "y_3d_5pct"]


def gate_new_high(scored: pd.DataFrame, column: str) -> pd.DataFrame:
    """Keep score. Ineligible rows get level=+inf so first_trigger skips them.

    Row 0 of a symbol-day is eligible on score alone. A later row is eligible
    only when column > cummax(column).shift(1). A missing value does not pass `>`.
    """
    out = scored.sort_values(["symbol", "session_date", "decision_at"], kind="mergesort").reset_index(drop=True)
    keys = [out["symbol"], out["session_date"]]
    prior = out.groupby(keys, sort=False)[column].cummax().groupby(keys, sort=False).shift(1)
    first = out.groupby(keys, sort=False).cumcount().eq(0)
    eligible = first | (out[column] > prior)
    out.loc[~eligible, "level"] = np.inf
    return out


def week_triggers(scored: pd.DataFrame, column: str) -> dict:
    """Gate each exam week on its own. Do not pool splits or drop triplets."""
    out = {}
    for split, key in (("test", "test_first"), ("validation", "validation_first")):
        part = scored.loc[scored["split"] == split]
        out[key] = first_trigger(gate_new_high(part, column))
    return out


def _clocks(gated: pd.DataFrame) -> list[tuple[str, str]]:
    ordered = gated.sort_values(["symbol", "session_date", "decision_at"])
    taken = ordered.loc[ordered["score"] >= ordered["level"]]
    taken = taken.groupby(["symbol", "session_date"], sort=False).head(1)
    return sorted((row.symbol, row.decision_at.strftime("%H:%M")) for row in taken.itertuples())


def _level(gated: pd.DataFrame, symbol: str, clock: str) -> float:
    hit = gated.loc[(gated["symbol"] == symbol) & (gated["decision_at"].dt.strftime("%H:%M") == clock), "level"]
    if len(hit) != 1:
        raise AssertionError(f"{symbol} {clock} matched {len(hit)} rows")
    return float(hit.iloc[0])


def test_eligibility() -> None:
    """Six handmade rows. No market data."""
    frame = pd.DataFrame(
        [
            {"symbol": "B", "session_date": "2026-01-05", "decision_at": "2026-01-05 10:05:00", "impulse": 2.0, "score": 0.99, "level": 0.8, "y_3d_5pct": 1.0},
            {"symbol": "A", "session_date": "2026-01-05", "decision_at": "2026-01-05 10:15:00", "impulse": np.nan, "score": 0.99, "level": 0.8, "y_3d_5pct": 1.0},
            {"symbol": "A", "session_date": "2026-01-05", "decision_at": "2026-01-05 10:00:00", "impulse": 1.0, "score": 0.50, "level": 0.8, "y_3d_5pct": 0.0},
            {"symbol": "B", "session_date": "2026-01-05", "decision_at": "2026-01-05 10:00:00", "impulse": np.nan, "score": 0.85, "level": 0.8, "y_3d_5pct": 0.0},
            {"symbol": "A", "session_date": "2026-01-05", "decision_at": "2026-01-05 10:10:00", "impulse": 1.5, "score": 0.91, "level": 0.8, "y_3d_5pct": 1.0},
            {"symbol": "A", "session_date": "2026-01-05", "decision_at": "2026-01-05 10:05:00", "impulse": 1.0, "score": 0.90, "level": 0.8, "y_3d_5pct": 0.0},
        ]
    )
    if len(frame) != 6:
        raise AssertionError(len(frame))
    frame["decision_at"] = pd.to_datetime(frame["decision_at"])
    original_score = frame["score"].to_numpy(copy=True)
    original_level = frame["level"].to_numpy(copy=True)

    gated = gate_new_high(frame, "impulse")
    if not np.array_equal(frame["score"].to_numpy(), original_score):
        raise AssertionError("caller score changed")
    if not np.array_equal(frame["level"].to_numpy(), original_level):
        raise AssertionError("caller level changed")
    if len(gated) != 6:
        raise AssertionError(len(gated))
    # A: first window stays; equal impulse does not pass; 1.5 does; NaN does not.
    expected_impulse = {
        ("A", "10:00"): 0.8,
        ("A", "10:05"): np.inf,
        ("A", "10:10"): 0.8,
        ("A", "10:15"): np.inf,
        ("B", "10:00"): 0.8,
        ("B", "10:05"): np.inf,
    }
    for (symbol, clock), level in expected_impulse.items():
        got = _level(gated, symbol, clock)
        if not (np.isposinf(got) if np.isposinf(level) else got == level):
            raise AssertionError(f"impulse {symbol} {clock}: {got}")
    by_clock = {
        (row.symbol, row.decision_at.strftime("%H:%M")): float(row.score) for row in gated.itertuples()
    }
    if by_clock != {("A", "10:00"): 0.50, ("A", "10:05"): 0.90, ("A", "10:10"): 0.91, ("A", "10:15"): 0.99, ("B", "10:00"): 0.85, ("B", "10:05"): 0.99}:
        raise AssertionError(by_clock)
    if _clocks(gated) != [("A", "10:10"), ("B", "10:00")]:
        raise AssertionError(_clocks(gated))
    summary = first_trigger(gated)
    if summary["buys"] != 2 or summary["precision"] != 0.5:
        raise AssertionError(summary)

    scored = gate_new_high(frame, "score")
    expected_score = {
        ("A", "10:00"): 0.8,
        ("A", "10:05"): 0.8,
        ("A", "10:10"): 0.8,
        ("A", "10:15"): 0.8,
        ("B", "10:00"): 0.8,
        ("B", "10:05"): 0.8,
    }
    for (symbol, clock), level in expected_score.items():
        got = _level(scored, symbol, clock)
        if got != level:
            raise AssertionError(f"score {symbol} {clock}: {got}")
    if _clocks(scored) != [("A", "10:05"), ("B", "10:00")]:
        raise AssertionError(_clocks(scored))


def _same_trigger(left: dict, right: dict) -> bool:
    if left["buys"] != right["buys"] or left["days"] != right["days"]:
        return False
    if left["precision"] is None or right["precision"] is None:
        return left["precision"] is None and right["precision"] is None
    return abs(left["precision"] - right["precision"]) < 1e-9


def _fmt_pct(value) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def _fmt_lift(value) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.2f}个百分点"


def _row(name: str, result: dict) -> str:
    test = result["test_first"]
    val = result["validation_first"]
    cells = [
        name,
        str(test["buys"]),
        _fmt_pct(test["precision"]),
        _fmt_lift(test["lift"]),
        str(val["buys"]),
        _fmt_pct(val["precision"]),
        _fmt_lift(val["lift"]),
    ]
    return "| " + " | ".join(cells) + " |"


def _hits_90(result: dict) -> bool:
    for key in ("test_first", "validation_first"):
        item = result[key]
        if item["buys"] < 15 or item["precision"] is None or item["precision"] < 0.90:
            return False
    return True


def _fail_bits(candidate: dict, reference: dict) -> str:
    bits = []
    for key, name in (("test_first", "下一周"), ("validation_first", "再下一周")):
        left = candidate[key]
        right = reference[key]
        if left["buys"] < 15:
            bits.append(f"{name}只有 {left['buys']} 笔")
        elif left["precision"] is None or right["precision"] is None or not left["precision"] > right["precision"]:
            bits.append(
                f"{name}命中率 {_fmt_pct(left['precision'])}（{left['buys']} 笔）没有高于 {_fmt_pct(right['precision'])}"
            )
    return "，".join(bits)


def _bottleneck(baseline: dict, round_1: dict, round_1_beats: bool, round_2, round_2_beats) -> str:
    if not round_1_beats:
        return (
            "瓶颈是第1轮：后窗必须冲量严格高于当天更早的每一窗，"
            + _fail_bits(round_1, baseline)
            + "，因此没有跑第2轮。"
        )
    if round_2 is None or round_2_beats is None:
        raise RuntimeError("round 2 missing after round 1 passed")
    if not round_2_beats:
        return (
            "瓶颈停在第1轮：第1轮两周都高于当前规则，第2轮改成当天分数严格高于更早的每一窗后，"
            + _fail_bits(round_2, round_1)
            + "。"
        )
    return "第2轮两周都高于第1轮；按预登记停在第2轮，不再加第三条规则。"


def _write_report(lines: list[str]) -> None:
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _baseline_view(payload: dict) -> dict:
    return {"test_first": payload["test_first"], "validation_first": payload["validation_first"]}


def main() -> int:
    test_eligibility()
    print("eligibility test passed", flush=True)
    if not BASELINE_META.exists():
        _write_report(["# 行动门禁", "", "baseline.json 不存在，没有改分数，也没有重拟合。"])
        return 2
    try:
        baseline_payload = json.loads(BASELINE_META.read_text())
    except json.JSONDecodeError as exc:
        _write_report(["# 行动门禁", "", f"baseline.json 无法解析：{exc}"])
        return 2
    baseline = _baseline_view(baseline_payload)
    precision = baseline["test_first"]["precision"]
    if precision is None or not (0.462 <= float(precision) <= 0.473):
        _write_report([
            "# 行动门禁",
            "",
            f"baseline.json 下一周命中率是 {precision}，不在 0.462–0.473。已停止，没有重调门槛。",
            "",
            "预期大约 4539 笔、46.8%，再下一周大约 4495 笔、45.8%。",
        ])
        return 2

    scored = pd.read_parquet(BASELINE_SCORED, columns=NEEDED)
    recomputed = {
        "test_first": first_trigger(scored.loc[scored["split"] == "test"]),
        "validation_first": first_trigger(scored.loc[scored["split"] == "validation"]),
    }
    if not (_same_trigger(recomputed["test_first"], baseline["test_first"]) and _same_trigger(recomputed["validation_first"], baseline["validation_first"])):
        _write_report([
            "# 行动门禁",
            "",
            "同一份 baseline_scored 用 first_trigger 重算，和 baseline.json 不一致。已停止，没有重调门槛。",
            "",
            f"json 下一周 {baseline['test_first']}",
            "",
            f"重算下一周 {recomputed['test_first']}",
            "",
            f"json 再下一周 {baseline['validation_first']}",
            "",
            f"重算再下一周 {recomputed['validation_first']}",
        ])
        return 2

    round_1 = week_triggers(scored, "impulse")
    round_1_beats = beats(round_1, baseline)
    round_2 = None
    round_2_beats = None
    if round_1_beats:
        round_2 = week_triggers(scored, "score")
        round_2_beats = beats(round_2, round_1)

    payload = {
        "baseline": baseline,
        "round_1": round_1,
        "round_2": round_2,
        "beats": round_1_beats,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table = [
        "# 行动门禁",
        "",
        "分数和训练池 80 分位门槛沿用已算好的基准，没有重新拟合。第1轮只允许当天冲量严格创新高的窗口；当天第一窗只看是否过线。",
        "",
        "| 规则 | 下一周买入笔数 | 命中率 | 比当天第一窗 | 再下一周买入笔数 | 命中率 | 比当天第一窗 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        _row("当前", baseline),
        _row("第1轮", round_1),
    ]
    if round_2 is not None:
        table.append(_row("第2轮", round_2))
    table.append("")
    table.append(_bottleneck(baseline, round_1, round_1_beats, round_2, round_2_beats))
    showed = [baseline, round_1] + ([round_2] if round_2 is not None else [])
    if not any(_hits_90(item) for item in showed):
        table.append("")
        table.append("没有出现两周都至少 15 笔且命中率达到 90% 的结果。")
    _write_report(table)
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
