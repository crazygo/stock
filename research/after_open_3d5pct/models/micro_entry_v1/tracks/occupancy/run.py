"""Skip a new buy while this symbol's kept 1170-minute label is still open."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.micro_entry_v1.tracks.common import (
    BASELINE_META,
    BASELINE_SCORED,
    beats,
    first_trigger,
)

HERE = Path(__file__).resolve().parent
REPORT = HERE / "REPORT.md"
RESULT = HERE.parents[3] / "runs/micro_entry_v1/tracks/occupancy/result.json"
LOG = Path("/tmp/micro_entry_baseline.log")
PRECISION_BAND = (0.462, 0.473)


def first_cross(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol-day: the earliest window with score >= level."""
    ordered = scored.sort_values(["symbol", "session_date", "decision_at"])
    taken = ordered.loc[ordered["score"] >= ordered["level"]]
    return taken.groupby(["symbol", "session_date"], sort=False).head(1).reset_index(drop=True)


def select_kept(candidates: pd.DataFrame, touched=None) -> pd.DataFrame:
    """Walk each symbol by decision_at. A kept label blocks while label_end > decision_at.

    A skipped row does not occupy. touched(open_row, decision_at) may release the block
    once that open trade's high has already reached entry * 1.05. None means never release.
    """
    if candidates.empty:
        return candidates.copy()
    ordered = candidates.sort_values(["symbol", "decision_at", "session_date"], kind="mergesort")
    parts: list[pd.DataFrame] = []
    for _symbol, group in ordered.groupby("symbol", sort=False):
        open_rows: list[dict] = []
        kept: list[dict] = []
        for row in group.to_dict("records"):
            decision_at = pd.Timestamp(row["decision_at"])
            still = [prev for prev in open_rows if pd.Timestamp(prev["label_end"]) > decision_at]
            open_rows = still
            if still and (touched is None or any(not touched(prev, decision_at) for prev in still)):
                continue
            open_rows.append(row)
            kept.append(row)
        if kept:
            parts.append(pd.DataFrame(kept))
    if not parts:
        return ordered.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


class _RegularBars:
    """Regular-session bars for one symbol. Misaligned clocks count as not touched."""

    def __init__(self, frame: pd.DataFrame):
        if frame is None or frame.empty or not {"start", "end", "high"}.issubset(frame.columns):
            self.start = np.array([], dtype="datetime64[ns]")
            self.end = np.array([], dtype="datetime64[ns]")
            self.high = np.array([], dtype=float)
            return
        ordered = frame.sort_values("start")
        self.start = pd.to_datetime(ordered["start"]).to_numpy(dtype="datetime64[ns]")
        self.end = pd.to_datetime(ordered["end"]).to_numpy(dtype="datetime64[ns]")
        self.high = ordered["high"].to_numpy(dtype=float)

    def reached(self, entry_at, entry, decision_at) -> bool:
        if self.start.size == 0 or not np.isfinite(entry) or entry <= 0:
            return False
        entry_ns = np.datetime64(pd.Timestamp(entry_at).to_numpy(), "ns")
        decision_ns = np.datetime64(pd.Timestamp(decision_at).to_numpy(), "ns")
        left = int(np.searchsorted(self.start, entry_ns, side="left"))
        if left >= self.start.size or self.start[left] != entry_ns:
            return False
        right = int(np.searchsorted(self.end, decision_ns, side="right")) - 1
        if right < left or self.end[right] != decision_ns:
            return False
        highs = self.high[left:right + 1]
        if highs.size == 0 or not np.isfinite(highs).any():
            return False
        return bool(np.nanmax(highs) >= float(entry) * 1.05)


def _load_books(symbols: list[str]) -> dict[str, _RegularBars]:
    from research.after_open_3d5pct.models.pugh_ge5.build_rows import _regular_from_2024

    books = {}
    for symbol in symbols:
        bars, _floor = _regular_from_2024(symbol)
        books[symbol] = _RegularBars(bars)
        print(f"bars {symbol} rows={0 if bars is None else len(bars)}", flush=True)
    return books


def _touch_from(books: dict[str, _RegularBars]):
    def touched(open_row: dict, decision_at) -> bool:
        book = books.get(open_row["symbol"])
        if book is None:
            return False
        return book.reached(open_row["entry_at"], open_row["entry"], decision_at)

    return touched


def week_metrics(scored: pd.DataFrame, kept: pd.DataFrame) -> dict:
    """Buys and precision change. The first-window base stays the one first_trigger reports."""
    base = first_trigger(scored)
    precision = None if kept.empty else float(kept["y_3d_5pct"].mean())
    return {
        "days": base["days"],
        "first_clock_base": base["first_clock_base"],
        "buys": int(len(kept)),
        "precision": precision,
        "lift": None if precision is None else precision - base["first_clock_base"],
    }


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


def _pack(round_result: dict) -> dict:
    return {
        "test_first": round_result["test_first"],
        "validation_first": round_result["validation_first"],
    }


def _same_trigger(found: dict, published: dict) -> bool:
    if found["days"] != published["days"] or found["buys"] != published["buys"]:
        return False
    for key in ("precision", "first_clock_base", "lift"):
        if abs(float(found[key]) - float(published[key])) > 1e-9:
            return False
    return True


def _write_stop(text: str) -> None:
    REPORT.write_text(text)
    print(text, file=sys.stderr)


def _log_tail(lines: int = 40) -> str:
    if not LOG.exists():
        return "(no baseline log)"
    body = LOG.read_text(errors="replace").splitlines()
    return "\n".join(body[-lines:])


def wait_baseline() -> dict:
    """Poll until baseline.json parses and the scored parquet is present. Do not refit."""
    deadline = time.time() + 50 * 60
    while True:
        if BASELINE_META.exists() and BASELINE_SCORED.exists():
            try:
                meta = json.loads(BASELINE_META.read_text())
            except json.JSONDecodeError:
                meta = None
            if isinstance(meta, dict) and "test_first" in meta and "validation_first" in meta:
                return meta
        if time.time() >= deadline:
            _write_stop(
                "# 占仓\n\n"
                "基线没有在 50 分钟内变成可读的 json，或旁边没有 baseline_scored.parquet。没有重训。\n\n"
                "```\n"
                f"{_log_tail()}\n"
                "```\n"
            )
            raise SystemExit(1)
        time.sleep(15)


def _handmade() -> None:
    """No market bars. Open label blocks; an equal end does not; a skip does not occupy."""
    frame = pd.DataFrame([
        {
            "symbol": "S",
            "session_date": "2024-01-02",
            "decision_at": pd.Timestamp("2024-01-02 11:30"),
            "entry_at": pd.Timestamp("2024-01-02 11:30"),
            "label_end": pd.Timestamp("2024-01-04 11:30"),
            "entry": 10.0,
            "y_3d_5pct": 1.0,
        },
        {
            "symbol": "S",
            "session_date": "2024-01-03",
            "decision_at": pd.Timestamp("2024-01-03 11:30"),
            "entry_at": pd.Timestamp("2024-01-03 11:30"),
            "label_end": pd.Timestamp("2024-01-20 11:30"),
            "entry": 10.0,
            "y_3d_5pct": 0.0,
        },
        {
            "symbol": "S",
            "session_date": "2024-01-04",
            "decision_at": pd.Timestamp("2024-01-04 11:30"),
            "entry_at": pd.Timestamp("2024-01-04 11:30"),
            "label_end": pd.Timestamp("2024-01-08 11:30"),
            "entry": 10.0,
            "y_3d_5pct": 1.0,
        },
        {
            "symbol": "E",
            "session_date": "2024-01-02",
            "decision_at": pd.Timestamp("2024-01-02 11:30"),
            "entry_at": pd.Timestamp("2024-01-02 11:30"),
            "label_end": pd.Timestamp("2024-01-03 11:30"),
            "entry": 10.0,
            "y_3d_5pct": 1.0,
        },
        {
            "symbol": "E",
            "session_date": "2024-01-03",
            "decision_at": pd.Timestamp("2024-01-03 11:30"),
            "entry_at": pd.Timestamp("2024-01-03 11:30"),
            "label_end": pd.Timestamp("2024-01-06 11:30"),
            "entry": 10.0,
            "y_3d_5pct": 0.0,
        },
    ])
    kept = select_kept(frame)
    blocked = kept.loc[kept["symbol"].eq("S"), "session_date"].tolist()
    if blocked != ["2024-01-02", "2024-01-04"]:
        raise AssertionError(blocked)
    released = kept.loc[kept["symbol"].eq("E"), "session_date"].tolist()
    if released != ["2024-01-02", "2024-01-03"]:
        raise AssertionError(released)
    if "2024-01-03" in blocked:
        raise AssertionError("skipped day occupied the next decision")

    early = frame.loc[frame["symbol"].eq("S")].copy()
    early["label_end"] = pd.Timestamp("2024-01-20 11:30")

    def _touched(open_row, decision_at) -> bool:
        return pd.Timestamp(decision_at) == pd.Timestamp("2024-01-04 11:30")

    early_kept = select_kept(early, touched=_touched)["session_date"].tolist()
    if early_kept != ["2024-01-02", "2024-01-04"]:
        raise AssertionError(early_kept)

    bars = pd.DataFrame({
        "start": pd.to_datetime(["2024-01-02 11:30", "2024-01-02 11:35", "2024-01-02 11:40"]),
        "end": pd.to_datetime(["2024-01-02 11:35", "2024-01-02 11:40", "2024-01-02 11:45"]),
        "high": [110.0, 100.0, 130.0],
    })
    book = _RegularBars(bars)
    if book.reached("2024-01-02 11:30", 100.0, "2024-01-02 11:50"):
        raise AssertionError("missing decision bar counted as a touch")
    if not book.reached("2024-01-02 11:30", 100.0, "2024-01-02 11:40"):
        raise AssertionError("aligned +5% high was ignored")
    if book.reached("2024-01-02 11:30", 106.0, "2024-01-02 11:40"):
        raise AssertionError("high below entry * 1.05 counted")
    if book.reached("2024-01-02 11:31", 100.0, "2024-01-02 11:40"):
        raise AssertionError("missing entry bar counted as a touch")
    if book.reached("2024-01-02 11:30", 120.0, "2024-01-02 11:40"):
        raise AssertionError("high after the decision counted")


def _evaluate(scored: pd.DataFrame, touched=None) -> dict:
    packed = {}
    kept_by_split = {}
    for split, key in (("test", "test_first"), ("validation", "validation_first")):
        block = scored.loc[scored["split"].eq(split)]
        if block.groupby(["symbol", "session_date"])["triplet"].nunique().max() != 1:
            raise RuntimeError(f"{split} symbol-day spans more than one triplet")
        candidates = first_cross(block)
        kept = select_kept(candidates, touched=touched)
        packed[key] = week_metrics(block, kept)
        kept_by_split[split] = (candidates, kept)
    return {"metrics": packed, "detail": kept_by_split}


def _print_round(name: str, metrics: dict, detail: dict) -> None:
    for split, key in (("test", "test_first"), ("validation", "validation_first")):
        candidates, kept = detail[split]
        skipped = candidates.merge(
            kept[["symbol", "decision_at"]],
            on=["symbol", "decision_at"],
            how="left",
            indicator=True,
        )
        skipped = skipped.loc[skipped["_merge"].eq("left_only")]
        skip_rate = None if skipped.empty else float(skipped["y_3d_5pct"].mean())
        row = metrics[key]
        print(
            f"{name} {split} buys={row['buys']} precision={row['precision']} "
            f"lift={row['lift']} candidates={len(candidates)} skipped={len(skipped)} "
            f"skipped_precision={skip_rate}",
            flush=True,
        )


def main() -> None:
    _handmade()
    meta = wait_baseline()
    published = float(meta["test_first"]["precision"])
    if not (PRECISION_BAND[0] <= published <= PRECISION_BAND[1]):
        _write_stop(
            "# 占仓\n\n"
            f"baseline.json 的 test_first.precision = {published}，不在 0.462–0.473。没有改阈值。\n"
        )
        raise SystemExit(1)
    columns = [
        "symbol", "session_date", "decision_at", "entry_at", "entry", "y_3d_5pct",
        "label_end", "score", "level", "triplet", "split",
    ]
    scored = pd.read_parquet(BASELINE_SCORED, columns=columns)
    current = {}
    for split, key in (("test", "test_first"), ("validation", "validation_first")):
        current[key] = first_trigger(scored.loc[scored["split"].eq(split)])
        if not _same_trigger(current[key], meta[key]):
            _write_stop(
                "# 占仓\n\n"
                f"重算的 {key} 和 baseline.json 不一致。没有改规则。\n\n"
                f"重算 {current[key]}\n\n"
                f"文件 {meta[key]}\n"
            )
            raise SystemExit(1)
    round1 = _evaluate(scored, touched=None)
    _print_round("round1", round1["metrics"], round1["detail"])
    round1_beats = beats(round1["metrics"], current)
    payload = {
        "current": _pack({"test_first": current["test_first"], "validation_first": current["validation_first"]}),
        "round1": {**_pack(round1["metrics"]), "beats_baseline": round1_beats},
        "round2": None,
        "bound": "round1" if round1_beats else "current",
    }
    if round1_beats:
        symbols = sorted(scored["symbol"].unique())
        books = _load_books(symbols)
        round2 = _evaluate(scored, touched=_touch_from(books))
        _print_round("round2", round2["metrics"], round2["detail"])
        round2_beats = beats(round2["metrics"], round1["metrics"])
        payload["round2"] = {**_pack(round2["metrics"]), "beats_round1": round2_beats}
        if round2_beats:
            payload["bound"] = "round2"
    expected = ("after_open_3d5pct", "runs", "micro_entry_v1", "tracks", "occupancy", "result.json")
    if RESULT.parts[-6:] != expected:
        raise RuntimeError(f"result path {RESULT}")
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(_plain(payload), indent=2))
    print(f"wrote {RESULT} bound={payload['bound']}", flush=True)


if __name__ == "__main__":
    main()
