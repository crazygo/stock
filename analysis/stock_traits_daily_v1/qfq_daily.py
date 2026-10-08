"""OpenD daily K (QFQ) layer for --price-basis qfq_daily.

Rules (see README "QFQ daily mode"):
* One security = one source + one adjustment basis for the whole series. A series is
  either one freshly fetched OpenD K_DAY/QFQ response, or one reused file that passed all
  checks. Rows from different sources / bases are never spliced.
* QFQ history is rewritten after every corporate action (Futu QFQ can be linear
  price*A+B). A reused file is only trusted when it ends at the cutoff, was written after
  the cutoff close, has no internal last_close/close break and no split-like jump. When a
  fresh response exists, every reuse candidate is compared on the overlap; a difference is
  recorded as rebase_detected and the fresh series is used.
* Rows after the cutoff session (or available after the run's info cutoff) are dropped.
* New quota (securities not yet in the rolling 7-day window) is capped by QuotaBudget.
"""
from __future__ import annotations

import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from common import REPO, Calendar, sha256_file
from sources import SPLIT_RATIOS

PRICE_BASIS_QFQ = {
    "provider": "futu_opend",
    "bar": "1d (OpenD K_DAY)",
    "autype": "QFQ",
    "currency": "listing currency (USD for US)",
    "session": "regular (exchange daily bar)",
    "daily_close_rule": "close of the OpenD daily K bar; whole series from one QFQ response (or one verified reused file)",
    "adjustment_note": "QFQ is re-based after corporate actions; series are never spliced across sources or responses",
}
LAYER_FRESH = "opend_day_qfq"
LAYER_REUSED = "reused_qfq"
SH = timezone(timedelta(hours=8))


def _rel(p: Path) -> str:
    p = Path(p)
    return str(p.relative_to(REPO)) if p.is_absolute() and p.is_relative_to(REPO) else str(p)


def load_dayk(path: Path) -> pd.DataFrame:
    """Read a raw OpenD daily K parquet (futu columns) or a normalized daily file."""
    df = pd.read_parquet(path)
    if "time_key" in df.columns:
        d = df["time_key"].astype(str).str[:10]
    elif "session_date" in df.columns:
        d = df["session_date"].astype(str).str[:10]
    elif "date" in df.columns:
        d = df["date"].astype(str).str[:10]
    else:
        raise ValueError(f"no time_key/session_date/date column in {path.name}: {list(df.columns)[:12]}")
    need = ["open", "high", "low", "close", "volume"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns {miss}")
    out = pd.DataFrame({"session_date": d, **{c: pd.to_numeric(df[c], errors="coerce") for c in need}})
    if "last_close" in df.columns:
        out["last_close"] = pd.to_numeric(df["last_close"], errors="coerce")
    return out.sort_values("session_date").drop_duplicates("session_date", keep="last").reset_index(drop=True)


def derive_daily_dayk(df: pd.DataFrame, cal: Calendar, start: str, end: str, info_cutoff: datetime,
                      layer: str) -> tuple[dict, dict, list]:
    """Return (rows, problems, future_dropped) for sessions in [start, end]."""
    rows, problems, future = {}, {}, []
    for r in df.to_dict("records"):
        d = r["session_date"]
        if d > end:
            future.append(d)
            continue
        if d < start:
            continue
        if not cal.is_session(d):
            problems[d] = "bar_on_non_session_day"
            continue
        close_at = cal.close_at(d)
        if close_at > info_cutoff:
            future.append(d)
            continue
        o, h, l, c, v = (float(r[k]) for k in ("open", "high", "low", "close", "volume"))
        if not all(math.isfinite(x) and x > 0 for x in (o, h, l, c)) or l > min(o, c) + 1e-9 or h < max(o, c) - 1e-9:
            problems[d] = "invalid_ohlc"
            continue
        rows[d] = {"session_date": d, "open": o, "high": h, "low": l, "close": c,
                   "volume": float(v) if math.isfinite(v) else 0.0, "regular_bars": 1, "expected_bars": 1,
                   "bar_start": cal.open_at(d).isoformat(), "bar_end": close_at.isoformat(),
                   "available_at": close_at.isoformat(), "layer": layer}
    return rows, problems, future


def last_close_breaks(df: pd.DataFrame, tol_abs: float = 0.0015, tol_rel: float = 2e-4) -> list[dict]:
    """Inside one response last_close[t] equals close[t-1]; a mismatch means splicing or a partial rebase."""
    if "last_close" not in df.columns or len(df) < 2:
        return []
    out = []
    prev = None
    for r in df.to_dict("records"):
        if prev is not None and math.isfinite(r.get("last_close", float("nan"))) and r["last_close"] > 0:
            if abs(r["last_close"] - prev["close"]) > tol_abs + tol_rel * abs(prev["close"]):
                out.append({"session_date": r["session_date"], "last_close": r["last_close"], "prev_close": prev["close"]})
        prev = r
    return out


def split_like_jumps(rows: dict) -> list[dict]:
    out, prev = [], None
    for d in sorted(rows):
        if prev is not None:
            ratio = rows[prev]["close"] / rows[d]["close"]
            if abs(math.log(ratio)) > math.log(1.8) and any(abs(ratio / t - 1) < 0.04 for k in SPLIT_RATIOS for t in (k, 1 / k)):
                out.append({"session_date": d, "ratio": round(ratio, 4)})
        prev = d
    return out


def compare_series(fresh: dict, other: dict) -> dict:
    """Overlap comparison (close/open/high/low). Tolerance ~ OpenD 3-decimal rounding."""
    common = sorted(set(fresh) & set(other))
    max_abs = max_rel = 0.0
    diffs = []
    for d in common:
        worst = 0.0
        for k in ("open", "high", "low", "close"):
            a, b = fresh[d][k], other[d][k]
            dif = abs(a - b)
            if dif > 0.0015 + 2e-4 * abs(a):
                worst = max(worst, dif / abs(a))
            max_abs = max(max_abs, dif)
            max_rel = max(max_rel, dif / abs(a))
        if worst:
            diffs.append(d)
    return {"overlap_sessions": len(common), "max_abs_diff": round(max_abs, 6), "max_rel_diff": round(max_rel, 8),
            "differing_sessions": len(diffs), "first_differing": diffs[0] if diffs else None,
            "last_differing": diffs[-1] if diffs else None, "rebase_detected": bool(diffs)}


class ReuseQfq:
    """Read-only candidates from existing OpenD QFQ daily directories (files named <security_id>.parquet)."""

    def __init__(self, dirs: list[str] | None):
        self.dirs = [Path(d) if Path(d).is_absolute() else REPO / d for d in (dirs or [])]
        self.read_hashes: dict[str, str] = {}
        self._cache: dict = {}

    def candidates(self, sid: str, cal: Calendar, start: str, end: str, info_cutoff: datetime) -> list[dict]:
        if sid in self._cache:
            return self._cache[sid]
        out = []
        cutoff_close = cal.close_at(end)
        for base in self.dirs:
            p = base / f"{sid}.parquet"
            if not p.exists():
                continue
            h = sha256_file(p)
            self.read_hashes[str(p)] = h
            st = p.stat()
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
            c = {"path": _rel(p), "dir": _rel(base), "sha256": h, "bytes": st.st_size,
                 "mtime": mtime.astimezone(SH).isoformat(), "suspicious": [], "_abs": p}
            try:
                df = load_dayk(p)
            except Exception as exc:  # noqa: BLE001
                c.update(state="unreadable", error=f"{type(exc).__name__}: {str(exc)[:200]}")
                c["suspicious"].append("unreadable")
                out.append(c)
                continue
            rows, problems, future = derive_daily_dayk(df, cal, start, end, info_cutoff, f"{LAYER_REUSED}:{_rel(base)}")
            req = cal.sessions_between(start, end) if hasattr(cal, "sessions_between") else [d for d in cal.dates if start <= d <= end]
            first = min(rows) if rows else None
            gaps = [d for d in req if first and d >= first and d not in rows]
            lcb = last_close_breaks(df[(df["session_date"] >= start) & (df["session_date"] <= end)])
            jumps = split_like_jumps(rows)
            last = max(rows) if rows else None
            if not rows:
                c["suspicious"].append("no_rows_in_window")
            if last and last < end:
                c["suspicious"].append("ends_before_cutoff")
            if mtime < cutoff_close:
                c["suspicious"].append("written_before_cutoff_close")
            if gaps:
                c["suspicious"].append("internal_gaps")
            if lcb:
                c["suspicious"].append("last_close_break")
            if jumps:
                c["suspicious"].append("split_like_jump_in_qfq_series")
            if problems:
                c["suspicious"].append("bad_rows")
            c.update(state="ok" if not c["suspicious"] else "suspicious", first=first, last=last, sessions=len(rows),
                     gaps=len(gaps), last_close_breaks=lcb[:5], split_like_jumps=jumps[:5],
                     future_rows_dropped=len(future), problems=problems, _rows=rows)
            out.append(c)
        self._cache[sid] = out
        return out


def public(c: dict) -> dict:
    return {k: v for k, v in c.items() if not k.startswith("_")}


class QuotaBudget:
    """Caps *new* history-kline quota (securities not already in Futu's rolling 7-day window)."""

    def __init__(self, quota: dict | None, max_new: int, min_remaining: int, now: datetime,
                 window_days: float = 7.0, margin_hours: float = 12.0):
        q = quota or {}
        self.known = isinstance(q.get("used"), int) and isinstance(q.get("remain"), int)
        self.used_before, self.remain_before = q.get("used"), q.get("remain")
        self.max_new, self.min_remaining = max_new, min_remaining
        # free = requested inside the window with a safety margin (request_time is OpenD local time; assume UTC+8)
        self.codes: set[str] = set()
        self.expiring: list[str] = []
        limit = now - timedelta(days=window_days) + timedelta(hours=margin_hours)
        times = q.get("request_times") or {}
        for code in q.get("codes") or []:
            t = times.get(code)
            try:
                tt = datetime.fromisoformat(str(t).replace(" ", "T"))
                tt = tt if tt.tzinfo else tt.replace(tzinfo=SH)
            except Exception:  # noqa: BLE001
                tt = None
            if tt is not None and tt < limit:
                self.expiring.append(code)
            else:
                self.codes.add(code)
        self.new_used = 0
        self.new_codes: list[str] = []
        self.free_fetches = 0
        self.skipped: list[dict] = []

    def is_free(self, code: str) -> bool:
        return code in self.codes

    def check(self, code: str) -> tuple[bool, str]:
        if code in self.codes:
            return True, "already_in_7day_window"
        if not self.known:
            return False, "quota_unknown (probe failed)"
        if self.new_used + 1 > self.max_new:
            return False, f"quota_max_new {self.max_new} reached"
        if self.remain_before - self.new_used - 1 < self.min_remaining:
            return False, f"would leave fewer than {self.min_remaining} remaining (remain_before {self.remain_before})"
        return True, "new_quota"

    def consume(self, code: str) -> None:
        if code in self.codes:
            self.free_fetches += 1
            return
        self.new_used += 1
        self.new_codes.append(code)
        self.codes.add(code)

    def plan_ok(self, planned_new: int) -> tuple[bool, str]:
        if not self.known:
            return False, "quota_unknown (probe failed)"
        if planned_new > self.max_new:
            return False, f"planned new quota {planned_new} > quota_max_new {self.max_new}"
        if self.remain_before - planned_new < self.min_remaining:
            return False, f"remain {self.remain_before} - planned {planned_new} < quota_min_remaining {self.min_remaining}"
        return True, "ok"

    def summary(self) -> dict:
        return {"known": self.known, "used_before": self.used_before, "remain_before": self.remain_before,
                "quota_max_new": self.max_new, "quota_min_remaining": self.min_remaining,
                "window_codes_free": len(self.codes) - self.new_used, "expiring_soon_treated_as_new": len(self.expiring),
                "new_used": self.new_used, "free_fetches": self.free_fetches,
                "skipped_quota_budget": len(self.skipped), "skipped": self.skipped[:50]}


def acquire_qfq(sec: dict, ctx: dict) -> tuple[dict, dict, list, dict]:
    """Pick exactly one source for this security. Returns (rows, problems, layers, source_info)."""
    cal: Calendar = ctx["cal"]
    sid = sec["security_id"]
    start, end = ctx["required"][0], ctx["required"][-1]
    policy = ctx["qfq_reuse_policy"]
    budget: QuotaBudget = ctx["quota"]
    layers: list = []
    cands = ctx["reuse_qfq"].candidates(sid, cal, start, end, ctx["info_cutoff"])
    good = [c for c in cands if c["state"] == "ok"]
    best = max(good, key=lambda c: (c["sessions"], c["mtime"])) if good else None
    layers.append({"layer": LAYER_REUSED, "state": "candidates" if cands else "miss",
                   "candidates": [{k: v for k, v in public(c).items() if k != "problems"} for c in cands]})
    want_fresh = (policy == "never" or best is None or (policy == "verify_free" and budget.is_free(sid)))
    fresh_rows, fresh_problems, fresh_info = None, {}, None
    if want_fresh:
        raw = ctx["capture_tmp"] / "opend_day_qfq" / f"{sid}.parquet"
        raw.parent.mkdir(parents=True, exist_ok=True)
        resume = ctx.get("reuse_raw") and Path(ctx["reuse_raw"]) / f"{sid}.parquet"
        entry: dict = {"layer": LAYER_FRESH}
        res = None
        if resume and resume.exists():
            shutil.copy2(resume, raw)
            entry.update(state="reused_raw_from_interrupted_attempt", reused_from=_rel(resume),
                         raw_fetched_at=datetime.fromtimestamp(resume.stat().st_mtime, tz=timezone.utc).astimezone(SH).isoformat())
            res = {"ok": True, "out": str(raw)}
        else:
            op = ctx["opend"]
            import time as _t
            if _t.monotonic() > ctx["deadline"]:
                entry.update(state="skipped_time_budget")
            elif not op.available():
                pr = op.probe()
                entry.update(state="unavailable", error_type=pr.get("error_type"), error=pr.get("error"))
            else:
                ok, why = budget.check(sid)
                entry["quota"] = why
                if not ok:
                    entry.update(state="skipped_quota_budget")
                    budget.skipped.append({"security_id": sid, "reason": why})
                else:
                    res = op.fetch_day(sid, start, end, raw)
                    budget.consume(sid)  # a request may count even when it fails: be conservative
                    entry.update({k: v for k, v in res.items() if k not in ("out", "action", "host", "port")})
                    entry["requested_range"] = [start, end]
        if res and res.get("ok") and res.get("out"):
            try:
                df = load_dayk(raw)
                fresh_rows, fresh_problems, future = derive_daily_dayk(df, cal, start, end, ctx["info_cutoff"], LAYER_FRESH)
                lcb = last_close_breaks(df[(df["session_date"] >= start) & (df["session_date"] <= end)])
                fresh_info = {"type": "opend_fresh", "layer": LAYER_FRESH, "raw_path": _rel(raw), "raw_sha256": sha256_file(raw),
                              "autype": "QFQ", "ktype": "K_DAY", "future_rows_dropped": len(future),
                              "last_close_breaks": lcb[:5],
                              **({"reused_raw_from": entry.get("reused_from"), "raw_fetched_at": entry.get("raw_fetched_at")}
                                 if entry.get("reused_from") else {"fetched_at": datetime.now(timezone.utc).astimezone(SH).isoformat()})}
                entry.setdefault("state", "fetched")
                entry.update(sessions=len(fresh_rows), future_rows_dropped=len(future))
            except Exception as exc:  # noqa: BLE001
                entry.update(state="unreadable_response", error=f"{type(exc).__name__}: {str(exc)[:200]}")
                fresh_rows = None
        elif res is not None and "state" not in entry:
            entry["state"] = "failed" if not res.get("ok") else "empty_response"
        layers.append(entry)
    else:
        layers.append({"layer": LAYER_FRESH, "state": "not_needed",
                       "reason": "verified reuse candidate and security not free in quota window (policy %s)" % policy})
    checks = []
    if fresh_rows:
        for c in cands:
            if c.get("_rows"):
                checks.append({"candidate": c["path"], **compare_series(fresh_rows, c["_rows"])})
        source = {**fresh_info, "rebase_checks": checks,
                  "rebase_detected_vs_reuse": any(x["rebase_detected"] for x in checks)}
        return fresh_rows, fresh_problems, layers, source
    if best is not None:
        others = [c for c in good if c is not best]
        for c in others:
            checks.append({"candidate": c["path"], **compare_series(best["_rows"], c["_rows"])})
        source = {"type": "reused_file", "layer": f"{LAYER_REUSED}:{best['dir']}", "path": best["path"],
                  "sha256": best["sha256"], "mtime": best["mtime"], "autype": "QFQ", "ktype": "K_DAY",
                  "rebase_checks": checks, "rebase_detected_vs_reuse": any(x["rebase_detected"] for x in checks),
                  "rebase_verification": "candidate passed: ends at cutoff, written after cutoff close, no last_close "
                                         "break, no split-like jump; no fresh response compared (not free in quota window)"}
        if want_fresh:
            source["fallback_reason"] = "fresh fetch not available: " + str(layers[-1].get("state"))
        return dict(best["_rows"]), dict(best["problems"]), layers, source
    return {}, fresh_problems, layers, {"type": "none"}
