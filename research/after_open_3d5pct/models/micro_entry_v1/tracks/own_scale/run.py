"""Replace the pooled 80th percentile with each stock's own past-60 score level."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.micro_entry_v1.tracks.common import (
    BASELINE_META,
    FEATURES,
    FRAME,
    beats,
    load_frame,
    pack,
    walk,
)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parents[3] / "runs" / "micro_entry_v1" / "tracks" / "own_scale"
REPORT = HERE / "REPORT.md"
LOG = Path("/tmp/micro_entry_baseline.log")
LEVEL_WINDOW = 60
LEVEL_MIN_PRIOR = 40
LEVEL_QUANTILE = 0.80
GAP_WINDOW = 60
GAP_MIN_PRIOR = 20
LOOKBACK_DAYS = 10
ROUND2_FEATURES = list(FEATURES) + ["impulse_gap60"]

_state = {"frame": None, "round": 0, "n": 0}


def past_60_level(score: pd.Series) -> pd.Series:
    """80th percentile of the previous 60 scores. shift(1) drops the current score.

    Fewer than 40 earlier scores becomes +inf, so that row cannot buy.
    """
    level = score.rolling(LEVEL_WINDOW, min_periods=LEVEL_MIN_PRIOR).quantile(LEVEL_QUANTILE).shift(1)
    return level.fillna(np.inf)


def score_levels(scored: pd.DataFrame) -> pd.DataFrame:
    ordered = scored.sort_values(["symbol", "decision_at"], kind="mergesort").copy()
    ordered["level"] = ordered.groupby("symbol", sort=False)["score"].transform(past_60_level)
    return ordered


def prior_impulse_gap(impulse: pd.Series) -> pd.Series:
    """Current impulse minus the mean of up to 60 earlier impulses.

    The current impulse is outside that mean. Fewer than 20 earlier impulses stays NaN.
    """
    prior = impulse.rolling(GAP_WINDOW, min_periods=GAP_MIN_PRIOR).mean().shift(1)
    return impulse - prior


def with_impulse_gap60(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ordered = out.sort_values(["symbol", "decision_at"], kind="mergesort")
    gap = ordered.groupby("symbol", sort=False)["impulse"].transform(prior_impulse_gap)
    out["impulse_gap60"] = gap
    return out


def _attach(block: pd.DataFrame, keyed: pd.DataFrame) -> pd.DataFrame:
    scored = block.merge(
        keyed.loc[:, ["symbol", "decision_at", "score", "level"]],
        on=["symbol", "decision_at"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    missed = int(scored["score"].isna().sum())
    if missed:
        raise RuntimeError(f"own-scale score missed {missed} of {len(scored)} rows")
    return scored


def annotate(train, test, validation, model, features):
    del train
    frame = _state["frame"]
    start = pd.Timestamp(test["decision_at"].min()) - pd.Timedelta(days=LOOKBACK_DAYS)
    end = pd.Timestamp(validation["decision_at"].max())
    mask = (frame["decision_at"] >= start) & (frame["decision_at"] <= end)
    hist = frame.loc[mask, ["symbol", "decision_at", *features]].copy()
    hist["score"] = model.predict_proba(hist.loc[:, features])[:, 1]
    leveled = score_levels(hist)
    keyed = leveled.loc[:, ["symbol", "decision_at", "score", "level"]]
    scored_test = _attach(test, keyed)
    scored_validation = _attach(validation, keyed)
    _state["n"] += 1
    if _state["n"] == 1 or _state["n"] % 20 == 0:
        print(json.dumps({
            "round": _state["round"],
            "triplets": _state["n"],
            "test_level_nunique": int(scored_test["level"].nunique(dropna=False)),
            "validation_level_nunique": int(scored_validation["level"].nunique(dropna=False)),
        }), flush=True)
    return scored_test, scored_validation


def _release(result: dict) -> dict:
    result.pop("scored_test", None)
    result.pop("scored_validation", None)
    return result


def _log_text() -> str:
    if not LOG.exists():
        return ""
    return LOG.read_text(errors="replace")


def _write_report(text: str) -> None:
    REPORT.write_text(text)


def _stop_dead(reason: str) -> None:
    tail = "\n".join(_log_text().splitlines()[-40:])
    _write_report(
        "# 本股自己的 80 分位\n\n"
        f"基准没有跑完，这一轨停在比较之前。{reason}\n\n"
        "```\n"
        f"{tail}\n"
        "```\n"
    )
    raise SystemExit(1)


def _stop_discrepancy(payload: dict) -> None:
    precision = payload.get("test_first", {}).get("precision")
    _write_report(
        "# 本股自己的 80 分位\n\n"
        "停在第1轮之前，没有重调。\n\n"
        f"基准 test_first.precision 是 {precision}，约定范围是 0.462 到 0.473。\n"
        "买入笔数约定大约是下一周 4539、再下一周 4495。\n"
    )
    raise SystemExit(1)


def wait_baseline(timeout_s: float = 50 * 60) -> dict:
    import pyarrow.parquet as pq

    deadline = time.time() + timeout_s
    while True:
        rows = 0
        if FRAME.exists():
            try:
                rows = int(pq.ParquetFile(FRAME).metadata.num_rows)
            except Exception:
                rows = 0
        payload = None
        if rows > 100_000 and BASELINE_META.exists():
            try:
                payload = json.loads(BASELINE_META.read_text())
            except json.JSONDecodeError:
                payload = None
        if payload and "test_first" in payload and "validation_first" in payload:
            precision = payload["test_first"].get("precision")
            if not isinstance(precision, (int, float)) or not (0.462 <= float(precision) <= 0.473):
                _stop_discrepancy(payload)
            print(json.dumps({"frame_rows": rows, "baseline_precision": precision}), flush=True)
            return payload
        log = _log_text()
        if "Traceback" in log and payload is None:
            _stop_dead("日志里有 Traceback，且 baseline.json 还不能解析。")
        if time.time() >= deadline:
            _stop_dead("等了 50 分钟，基准帧或 baseline.json 仍未就绪。")
        time.sleep(15)


def _hits_90(result: dict) -> bool:
    for key in ("test_first", "validation_first"):
        block = result[key]
        if block["buys"] < 15 or block["precision"] is None or block["precision"] < 0.90:
            return False
    return True


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def _pp(value) -> str:
    if value is None:
        return "—"
    return f"{value * 100:+.2f}"


def _md_row(name: str, result: dict) -> str:
    left = result["test_first"]
    right = result["validation_first"]
    return (
        f"| {name} | {left['buys']} | {_pct(left['precision'])} | {_pp(left['lift'])} "
        f"| {right['buys']} | {_pct(right['precision'])} | {_pp(right['lift'])} |"
    )


def _week_clause(label: str, block: dict) -> str:
    if block["precision"] is None:
        return f"{label}没有买入（{block['buys']} 笔）"
    return f"{label}{_pct(block['precision'])}（{block['buys']} 笔）"


def _bottleneck(baseline: dict, round1: dict, round2: dict | None, round1_beats: bool, round2_beats: bool | None) -> str:
    if not round1_beats:
        base_left = baseline["test_first"]
        base_right = baseline["validation_first"]
        left = round1["test_first"]
        right = round1["validation_first"]
        return (
            "瓶颈是本股过去 60 窗的 80 分位比全体 80 分位松："
            f"下一周 {left['buys']} 笔、{_pct(left['precision'])}，对当前 {base_left['buys']} 笔、{_pct(base_left['precision'])}；"
            f"再下一周 {right['buys']} 笔、{_pct(right['precision'])}，对当前 {base_right['buys']} 笔、{_pct(base_right['precision'])}；"
            "两周都没有更高，所以没有做第2轮。"
        )
    assert round2 is not None and round2_beats is not None
    if not round2_beats:
        return (
            "第1轮是上界，第2轮是瓶颈："
            f"加上 impulse_gap60 之后，{_week_clause('下一周', round2['test_first'])}、"
            f"{_week_clause('再下一周', round2['validation_first'])}，"
            "没有两周都高于第1轮。"
        )
    return (
        "第2轮两周都高于第1轮，按协议停在这里，不再改窗口、分位，也不加第三个特征。"
    )


def _ninety(round1: dict, round2: dict | None) -> str:
    names = []
    if _hits_90(round1):
        names.append("第1轮")
    if round2 is not None and _hits_90(round2):
        names.append("第2轮")
    if not names:
        if round2 is None:
            left = round1["test_first"]
            right = round1["validation_first"]
            return (
                f"两周买入都多于 15 笔（{left['buys']} 和 {right['buys']}），"
                f"但命中率只有 {_pct(left['precision'])} 和 {_pct(right['precision'])}，没有达到 90%。"
            )
        return "第1轮和第2轮都没有在下一周和再下一周同时达到15笔且90%。"
    return f"{'、'.join(names)}在下一周和再下一周都至少15笔且命中率不低于90%。"


def _report(baseline: dict, round1: dict, round2: dict | None, round1_beats: bool, round2_beats: bool | None, bound_round: int) -> str:
    rows = [
        _md_row("当前（全体 80 分位，当天第一次越线）", baseline),
        _md_row("第1轮（本股过去 60 窗的 80 分位）", round1),
    ]
    if round2 is not None:
        rows.append(_md_row("第2轮（impulse 减去本股过去 60 窗均值）", round2))
    gap_line = ""
    if round2 is not None:
        gap_line = (
            "第2轮只多一个数：这次推动减去这只股票过去最多 60 个窗口推动的平均，当前这次不放进平均。"
            "不够 20 个窗口就空着，不填 0。\n\n"
        )
    def _auc(value) -> str:
        if value is None:
            return "—"
        return f"{value:.3f}"

    auc_line = f"第1轮训练排序 {_auc(round1['mean_train_auc'])}。"
    if round2 is not None:
        auc_line = (
            f"训练排序第1轮 {_auc(round1['mean_train_auc'])}，第2轮 {_auc(round2['mean_train_auc'])}。"
        )
    return (
        "# 本股自己的 80 分位\n\n"
        "2026-09-29。当前做法是 21 只股票放在一起，取训练分数的 80 分位，当天第一次越过才买。"
        "第1轮不改特征，买入线改成这一只股票自己过去 60 个窗口分数的 80 分位，当前这个窗口不参与。"
        "不够 40 个窗口就不买。比较用的是当天第一次越线。比当天第一窗的单位是百分点。\n\n"
        f"{gap_line}"
        "| 方案 | 下一周买入笔数 | 命中率 | 比当天第一窗 | 再下一周买入笔数 | 命中率 | 比当天第一窗 |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n"
        f"{_bottleneck(baseline, round1, round2, round1_beats, round2_beats)}\n\n"
        f"{_ninety(round1, round2)}\n\n"
        f"{'这一轨停在第1轮，不替换当前的全体 80 分位。' if bound_round == 1 and not round1_beats else f'上界是第{bound_round}轮。'}"
        f"{auc_line}"
        f"第1轮训完 {round1['fitted_triplets']} 组，跳过 {round1['skipped_triplets']} 组。"
        "分数和当前模型相同，变的只是买入线。\n\n"
        "汇总在 `research/after_open_3d5pct/runs/micro_entry_v1/tracks/own_scale/result.json`。\n"
    )


def _firsts(result: dict) -> dict:
    return {"test_first": result["test_first"], "validation_first": result["validation_first"]}


def main() -> None:
    started = time.perf_counter()
    baseline = wait_baseline()
    _state["frame"] = load_frame()
    _state["round"] = 1
    _state["n"] = 0
    round1 = _release(walk(_state["frame"], list(FEATURES), annotate))
    round1_beats = beats(round1, baseline)
    print(json.dumps({
        "round": 1,
        "beats_baseline": round1_beats,
        "test_first": round1["test_first"],
        "validation_first": round1["validation_first"],
    }, default=str), flush=True)
    round2 = None
    round2_beats = None
    if round1_beats:
        _state["frame"] = with_impulse_gap60(_state["frame"])
        _state["round"] = 2
        _state["n"] = 0
        round2 = _release(walk(_state["frame"], list(ROUND2_FEATURES), annotate))
        round2_beats = beats(round2, round1)
        print(json.dumps({
            "round": 2,
            "beats_round1": round2_beats,
            "test_first": round2["test_first"],
            "validation_first": round2["validation_first"],
        }, default=str), flush=True)
    bound = round2 if round2_beats else round1
    bound_round = 2 if round2_beats else 1
    payload = pack(bound)
    payload["track"] = "own_scale"
    payload["bound_round"] = bound_round
    payload["beats_baseline"] = beats(bound, baseline)
    payload["round1_beats_baseline"] = round1_beats
    payload["round2_beats_round1"] = round2_beats
    payload["round2_ran"] = round2 is not None
    payload["reached_15_buys_and_90"] = _hits_90(bound)
    payload["baseline_first"] = _firsts(baseline)
    payload["round1_first"] = _firsts(round1)
    payload["round2_first"] = None if round2 is None else _firsts(round2)
    payload["seconds"] = round(time.perf_counter() - started, 1)
    if "scored_test" in payload or "scored_validation" in payload:
        raise RuntimeError("scored frames leaked into result.json")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_report(_report(baseline, round1, round2, round1_beats, round2_beats, bound_round))
    print(json.dumps({"bound_round": bound_round, "seconds": payload["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
