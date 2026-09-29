"""Crossing shapes on the frozen first-cross entry. Later rounds only split the same 24-bar path."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.micro_entry_v1.rows import horizon_last
from research.after_open_3d5pct.models.micro_entry_v1.tracks.common import (
    BASELINE_META,
    FEATURES,
    FRAME,
    annotate_current,
    beats,
    load_frame,
    pack,
    walk,
)
from research.after_open_3d5pct.models.pugh_ge5.build_rows import _regular_from_2024, union_symbols

LENGTH = 24
STEP = 1
KEYS = ["symbol", "session_date", "decision_at"]
OUT = FRAME.parent / "tracks" / "shape" / "result.json"
REPORT = Path(__file__).resolve().parent / "REPORT.md"
BASELINE_LOG = Path("/tmp/micro_entry_baseline.log")
QUARTERS = ("q1", "q2", "q3", "q4")
EIGHTHS = ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8")
BARS = tuple(f"b{index}" for index in range(LENGTH))
SHAPE_COLUMNS = [
    "symbol", "session_date", "decision_at", "early_return", "late_return", "half_gap",
    *QUARTERS, *EIGHTHS, *BARS,
]


def half_returns(open0: float, closes: np.ndarray) -> tuple[float, float, float]:
    """Early is close[11] / open0 - 1. Late is close[23] / close[11] - 1. Gap is late - early."""
    missing = (np.nan, np.nan, np.nan)
    path = np.asarray(closes, dtype=float)
    if path.shape != (LENGTH,):
        return missing
    o0 = float(open0)
    midpoint = float(path[11])
    last = float(path[23])
    if (not np.isfinite(o0)) or o0 <= 0 or (not np.isfinite(midpoint)) or midpoint <= 0 or not np.isfinite(last):
        return missing
    early = midpoint / o0 - 1.0
    late = last / midpoint - 1.0
    if not np.isfinite(early) or not np.isfinite(late):
        return missing
    gap = late - early
    if not np.isfinite(gap):
        return missing
    return float(early), float(late), float(gap)


def _segment_returns(open0: float, closes: np.ndarray, ends: tuple[int, ...]) -> tuple[float, ...]:
    """Chain of contiguous returns. One bad denominator or non-finite term blanks every slot."""
    missing = tuple(float("nan") for _ in ends)
    path = np.asarray(closes, dtype=float)
    if path.shape != (LENGTH,):
        return missing
    previous = float(open0)
    terms: list[float] = []
    for index in ends:
        close = float(path[index])
        if (not np.isfinite(previous)) or previous <= 0 or not np.isfinite(close):
            return missing
        term = close / previous - 1.0
        if not np.isfinite(term):
            return missing
        terms.append(float(term))
        previous = close
    return tuple(terms)


def quarter_returns(open0: float, closes: np.ndarray) -> tuple[float, float, float, float]:
    """Four contiguous 6-bar returns: c5/o0, c11/c5, c17/c11, c23/c17."""
    found = _segment_returns(open0, closes, (5, 11, 17, 23))
    return found[0], found[1], found[2], found[3]


def eighth_returns(open0: float, closes: np.ndarray) -> tuple[float, ...]:
    """Eight contiguous 3-bar returns ending on closes 2, 5, 8, 11, 14, 17, 20, 23."""
    return _segment_returns(open0, closes, (2, 5, 8, 11, 14, 17, 20, 23))


def bar_returns(open0: float, closes: np.ndarray) -> tuple[float, ...]:
    """b0 is close[0] / open0 - 1. Each later bar is close[k] / close[k - 1] - 1."""
    missing = tuple(float("nan") for _ in range(LENGTH))
    path = np.asarray(closes, dtype=float)
    if path.shape != (LENGTH,):
        return missing
    previous = float(open0)
    terms: list[float] = []
    for close in path:
        price = float(close)
        if (not np.isfinite(previous)) or previous <= 0 or not np.isfinite(price):
            return missing
        term = price / previous - 1.0
        if not np.isfinite(term):
            return missing
        terms.append(float(term))
        previous = price
    return tuple(terms)


def peer_impulse(impulse: pd.Series, decision_at: pd.Series) -> np.ndarray:
    """Leave-one-out mean of other names' impulses at the same decision_at. Pandas sum/count skip NaN."""
    values = impulse.to_numpy(dtype=float)
    order = pd.DataFrame({"stamp": decision_at.to_numpy(), "value": values})
    grouped_sum = order.groupby("stamp", sort=False)["value"].transform("sum").to_numpy(dtype=float)
    grouped_count = order.groupby("stamp", sort=False)["value"].transform("count").to_numpy(dtype=float)
    known = ~np.isnan(values)
    denom = np.where(known, grouped_count - 1.0, grouped_count)
    numer = np.where(known, grouped_sum - values, grouped_sum)
    out = np.full(len(values), np.nan, dtype=float)
    np.divide(numer, denom, out=out, where=denom >= 1)
    return out


def window_shape_rows(bars: pd.DataFrame, symbol: str, length: int = LENGTH, step: int = STEP) -> pd.DataFrame:
    """Same admission as rows_from_bars: contiguous window, entry is the next bar, horizon exists."""
    if bars.empty or length < 2 or step < 1:
        return pd.DataFrame(columns=SHAPE_COLUMNS)
    frame = bars.sort_values("start").reset_index(drop=True)
    day = frame["start"].dt.strftime("%Y-%m-%d").to_numpy()
    start = frame["start"].to_numpy()
    end = frame["end"].to_numpy()
    open_ = frame["open"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    minutes = frame["minutes"].to_numpy(dtype=int)
    contiguous = np.zeros(len(frame), dtype=bool)
    contiguous[1:] = start[1:] == end[:-1]
    rows: list[tuple] = []
    left = 0
    while left < len(frame):
        right = left
        while right + 1 < len(frame) and day[right + 1] == day[left]:
            right += 1
        for end_index in range(left + length - 1, right, step):
            entry_index = end_index + 1
            window_start = end_index - length + 1
            if entry_index > right or window_start < left:
                continue
            if not contiguous[window_start + 1:end_index + 1].all() or start[entry_index] != end[end_index]:
                continue
            if horizon_last(minutes, entry_index) is None:
                continue
            entry = float(open_[entry_index])
            if not np.isfinite(entry) or entry <= 0:
                continue
            path = close[window_start:end_index + 1]
            open0 = float(open_[window_start])
            early, late, gap = half_returns(open0, path)
            quarters = quarter_returns(open0, path)
            eighths = eighth_returns(open0, path)
            bars = bar_returns(open0, path)
            rows.append((symbol, day[left], end[end_index], early, late, gap, *quarters, *eighths, *bars))
        left = right + 1
    return pd.DataFrame(rows, columns=SHAPE_COLUMNS)


def build_geometry() -> pd.DataFrame:
    frames = []
    for symbol in union_symbols():
        bars, _floor = _regular_from_2024(symbol)
        built = window_shape_rows(bars, symbol)
        print(f"{symbol} shape_rows={len(built)}", flush=True)
        frames.append(built)
    if not frames:
        return pd.DataFrame(columns=SHAPE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _align_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str)
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.strftime("%Y-%m-%d")
    out["decision_at"] = pd.to_datetime(out["decision_at"]).astype("datetime64[ns]")
    return out


def _examples(frame: pd.DataFrame) -> list[dict]:
    copy = frame.head(8).copy()
    for column in copy.columns:
        if np.issubdtype(copy[column].dtype, np.datetime64):
            copy[column] = copy[column].astype(str)
    return copy.to_dict(orient="records")


def key_mismatch(frame: pd.DataFrame, geometry: pd.DataFrame) -> dict | None:
    left = frame[KEYS]
    right = geometry[KEYS]
    duplicate_frame = int(left.duplicated().sum())
    duplicate_geometry = int(right.duplicated().sum())
    compared = left.merge(right, on=KEYS, how="outer", indicator=True)
    frame_only = compared.loc[compared["_merge"].eq("left_only"), KEYS]
    geometry_only = compared.loc[compared["_merge"].eq("right_only"), KEYS]
    if duplicate_frame or duplicate_geometry or len(frame_only) or len(geometry_only) or len(frame) != len(geometry):
        return {
            "frame_rows": int(len(frame)),
            "geometry_rows": int(len(geometry)),
            "frame_duplicate_keys": duplicate_frame,
            "geometry_duplicate_keys": duplicate_geometry,
            "frame_only": int(len(frame_only)),
            "geometry_only": int(len(geometry_only)),
            "frame_only_examples": _examples(frame_only),
            "geometry_only_examples": _examples(geometry_only),
            "frame_counts": {key: int(value) for key, value in frame.groupby("symbol").size().items()},
            "geometry_counts": {key: int(value) for key, value in geometry.groupby("symbol").size().items()},
        }
    return None


def attach_geometry(frame: pd.DataFrame, geometry: pd.DataFrame) -> tuple[pd.DataFrame | None, dict | None]:
    left = _align_keys(frame)
    right = _align_keys(geometry)
    mismatch = key_mismatch(left, right)
    if mismatch is not None:
        return None, mismatch
    right = right.copy()
    right["_shape_key"] = 1
    merged = left.merge(right, on=KEYS, how="left", validate="one_to_one")
    if len(merged) != len(left) or not merged["_shape_key"].eq(1).all():
        return None, key_mismatch(left, right) or {"merge": "row count changed"}
    finite = merged["half_gap"].notna()
    if finite.any():
        gap = merged.loc[finite, "late_return"] - merged.loc[finite, "early_return"]
        if not np.allclose(gap.to_numpy(float), merged.loc[finite, "half_gap"].to_numpy(float), atol=1e-12):
            return None, {"merge": "half_gap is not late_return - early_return"}
    return merged.drop(columns="_shape_key"), None


def wait_frame() -> int:
    import pyarrow.parquet as pq
    while True:
        if FRAME.exists():
            try:
                rows = int(pq.read_metadata(FRAME).num_rows)
            except Exception:
                rows = 0
            if rows > 100000:
                print(f"frame rows={rows}", flush=True)
                return rows
        time.sleep(15)


def load_baseline() -> dict:
    if not BASELINE_META.exists():
        tail = BASELINE_LOG.read_text()[-4000:] if BASELINE_LOG.exists() else "baseline.json missing"
        raise SystemExit(tail)
    try:
        payload = json.loads(BASELINE_META.read_text())
    except json.JSONDecodeError:
        tail = BASELINE_LOG.read_text()[-4000:] if BASELINE_LOG.exists() else "baseline.json did not parse"
        raise SystemExit(tail)
    precision = payload.get("test_first", {}).get("precision")
    if precision is None or not (0.462 <= float(precision) <= 0.473):
        raise SystemExit(f"test_first.precision={precision} outside 0.462-0.473")
    return payload


def walk_features(name: str, features: list[str], frame: pd.DataFrame) -> dict:
    print(f"walk {name}", flush=True)
    started = time.perf_counter()
    done = walk(frame, features, annotate_current)
    packed = pack(done)
    print(
        f"done {name} seconds={time.perf_counter() - started:.1f} "
        f"test_buys={packed['test_first']['buys']} test_precision={packed['test_first']['precision']} "
        f"validation_buys={packed['validation_first']['buys']} validation_precision={packed['validation_first']['precision']}",
        flush=True,
    )
    return packed


def _cells(block: dict) -> tuple[str, str, str]:
    precision = block.get("precision")
    lift = block.get("lift")
    return (
        str(int(block["buys"])),
        "—" if precision is None else f"{precision * 100:.2f}%",
        "—" if lift is None else f"{lift * 100:+.2f}",
    )


def _row(name: str, result: dict) -> str:
    left = _cells(result["test_first"])
    right = _cells(result["validation_first"])
    return f"| {name} | {left[0]} | {left[1]} | {left[2]} | {right[0]} | {right[1]} | {right[2]} |"


def _at_90(result: dict) -> bool:
    for key in ("test_first", "validation_first"):
        block = result[key]
        if block["buys"] < 15 or block["precision"] is None or block["precision"] < 0.90:
            return False
    return True


def bottleneck_sentence(beats_round4: bool) -> str:
    if beats_round4:
        return "瓶颈是逐根收益也超过了八段三根，24 根窗口不能再拆，按规则停在这一步。"
    return "瓶颈是逐根收益没有同时超过八段三根，上界停在八段三根收益。"


def write_report(baseline: dict, rows: list[tuple[str, dict]], sentence: str, reached_90: bool) -> None:
    body = [
        "# 越线形状",
        "",
        "动作仍是当天第一次越过训练池 80 分位。只换加进去的几何。",
        "",
        "| 行 | 下一周买入笔数 | 下一周命中率 | 下一周比当天第一窗 | 再下一周买入笔数 | 再下一周命中率 | 再下一周比当天第一窗 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in rows:
        body.append(_row(name, result))
    body.extend([
        "",
        "比当天第一窗是这一周命中率减去当天第一窗每笔都进的命中率，单位是百分点。",
        "",
        sentence,
        "",
    ])
    if reached_90:
        body.append("有一条两周都至少 15 笔，且命中率不低于 90%。")
    else:
        body.append("没有一条两周同时至少 15 笔且命中率不低于 90%。")
    body.extend([
        "",
        "逐行数字在 `research/after_open_3d5pct/runs/micro_entry_v1/tracks/shape/result.json`，该目录不进 Git。",
        "",
    ])
    REPORT.write_text("\n".join(body))


def _write(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _trusted_round2(saved: dict) -> dict:
    """Round 2 is already scored. Do not refit it."""
    found = saved.get("round2", {}).get("early_return_late_return")
    if not found:
        raise SystemExit("round2.early_return_late_return missing")
    test = found["test_first"]
    validation = found["validation_first"]
    if int(test["buys"]) != 5727 or int(validation["buys"]) != 5707:
        raise SystemExit(f"round 2 buys drifted: {test['buys']} {validation['buys']}")
    if abs(float(test["precision"]) - 0.48419766020604155) > 1e-15:
        raise SystemExit(f"round 2 test precision drifted: {test['precision']}")
    if abs(float(validation["precision"]) - 0.4774837918345891) > 1e-15:
        raise SystemExit(f"round 2 validation precision drifted: {validation['precision']}")
    return found


def _trusted_round4(saved: dict) -> dict:
    """Round 4 is already scored. Compare to the file, and do not refit it."""
    found = saved.get("round4", {}).get("e1_to_e8")
    if not found:
        raise SystemExit("round4.e1_to_e8 missing")
    test = found["test_first"]
    validation = found["validation_first"]
    if int(test["buys"]) != 5564 or int(validation["buys"]) != 5516:
        raise SystemExit(f"round 4 buys drifted: {test['buys']} {validation['buys']}")
    print(
        f"round4 file test_precision={test['precision']} validation_precision={validation['precision']}",
        flush=True,
    )
    return found


def _history_rows(saved: dict) -> list[tuple[str, dict]]:
    rows = [
        ("当前", saved["baseline"]),
        ("前半对后半", saved["round1"]["half_gap"]),
        ("前半收益和后半收益分开", saved["round2"]["early_return_late_return"]),
        ("其他股票同一时刻的冲量", saved["round1"]["peer_impulse"]),
        ("四段六根收益", saved["round3"]["q1_q2_q3_q4"]),
        ("八段三根收益", saved["round4"]["e1_to_e8"]),
    ]
    return rows


def main() -> None:
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "research.after_open_3d5pct.models.micro_entry_v1.tracks.shape.test_shape"
    )
    if not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful():
        raise SystemExit(1)
    wait_frame()
    saved = json.loads(OUT.read_text())
    round4 = _trusted_round4(saved)
    frame = load_frame()
    geometry = build_geometry()
    attached, mismatch = attach_geometry(frame, geometry)
    if mismatch is not None:
        saved["mismatch"] = mismatch
        saved["bottleneck"] = "瓶颈是逐根收益和原表对不上，合并不是一一对应，已停下。"
        _write(saved)
        raise SystemExit("merge is not one-to-one")
    frame = attached
    finite_bars = float(frame["b0"].notna().mean())
    print(f"attached rows={len(frame)} bar_finite={finite_bars:.4f}", flush=True)
    bar_features = list(FEATURES) + list(BARS)
    if any(name in QUARTERS or name in EIGHTHS for name in bar_features):
        raise SystemExit("bar model must not include quarter or eighth returns")
    packed = walk_features("b0_to_b23", bar_features, frame)
    packed["features"] = bar_features
    packed["beats_round4"] = beats(packed, round4)
    saved["round5"] = {"b0_to_b23": packed}
    saved["bar_finite_share"] = finite_bars
    rows = _history_rows(saved)
    rows.append(("逐根收益", packed))
    sentence = bottleneck_sentence(packed["beats_round4"])
    reached = any(_at_90(item) for _name, item in rows if _name != "当前")
    saved["bottleneck"] = sentence
    saved["claims_90"] = reached
    _write(saved)
    write_report(saved["baseline"], rows, sentence, reached)
    print(sentence, flush=True)


if __name__ == "__main__":
    main()
