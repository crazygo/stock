#!/usr/bin/env python3
"""stock-data-backfill v1: publish an immutable daily-close data manifest.

Example (real run on the box):
  python analysis/stock_traits_daily_v1/backfill.py --cutoff auto --task init

Writes only to: --capture-root (raw parquet, git-ignored) and --data-root (manifests,
ledger, R2 cache; git-ignored). Never uploads to R2, never edits watchlists or trades.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (DEFAULT_CALENDAR, DEFAULT_CAPTURE_ROOT, DEFAULT_DATA_ROOT, DEFAULT_LOCAL_5M,  # noqa: E402
                    DEFAULT_SEC_FACTS, DEFAULT_UNIVERSE_FILE, MANIFEST_SCHEMA, REPO, AlreadyRunning, Calendar,
                    assert_own_data_root,
                    append_jsonl, git_commit_of, iso, iso_sh, make_readonly, new_id, parse_ts, publish_dir,
                    read_json, rel, run_lock, sha256_bytes, canonical_json, sha256_file, utcnow, write_json_atomic,
                    write_json_new)
from sources import (PRICE_BASIS, LocalLayer, OpenDLayer, R2Layer, content_hash, derive_daily,  # noqa: E402
                     normalize_opend_5m, split_events, suspected_splits, write_daily_parquet)
from qfq_daily import PRICE_BASIS_QFQ, QuotaBudget, ReuseQfq, acquire_qfq, public  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_AI_BASKET = REPO / "analysis/ai_value_chain_map_v1/ai_basket_members.json"
DEFAULT_AI_SCREEN = REPO / "analysis/ai_value_chain_map_v1/quality/screen_current.json"
DS_PREFIX = {"futu_5m_none": "us_daily_close_5m_none", "qfq_daily": "daily_qfq_opend_day"}

FIVE_TRAITS_MIN_CLOSES = 141
DEFAULT_HISTORY_SESSIONS = 161  # 141 closes for five_traits_v2 + 20 sessions margin (stale cutoff / short curve)
LEVERAGED_RE = re.compile(r"(\d(\.\d)?\s*[xX倍])|三倍|两倍|做多|做空|杠杆|反向|inverse|leveraged|bear|bull|ultra", re.I)


# ------------------------------------------------------------------ universe
def classify(code: str, stock_type: str, name: str) -> tuple[str, bool, str | None]:
    market = code.split(".", 1)[0] if "." in code else None
    if market != "US":
        return "non_us_market", False, f"market {market or 'unknown'} not in US daily scope"
    st = (stock_type or "").upper()
    if st == "STOCK":
        return "stock", True, None
    if st == "ETF":
        return ("leveraged_or_inverse_etf" if LEVERAGED_RE.search(name or "") else "etf"), True, None
    if st in ("WARRANT", "DRVT", "BOND", "PLATE", "PLATESET", "INDEX"):
        return "non_security_or_derivative", False, f"stock_type {st}"
    return "unknown_identity", False, f"stock_type {st or 'unknown'}"


def load_sec_issuers(path: Path) -> dict:
    try:
        d = read_json(path)
        return {t: f"SEC_CIK:{c['cik']}" for t, c in d.get("companies", {}).items() if c.get("cik")}
    except Exception:
        return {}


def build_universe(args, opend: OpenDLayer, errors: list) -> dict:
    sec_issuers = load_sec_issuers(Path(args.sec_facts))
    attempts = []
    members, source = None, None
    if getattr(args, "universe", "watchlist") == "ai_basket":
        members, source = load_ai_basket(args)
    if members is None and args.symbols:
        members = [{"code": s if "." in s else f"US.{s}", "name": "", "stock_type": "UNKNOWN"} for s in args.symbols]
        source = {"type": "explicit_list", "observed_at": iso(utcnow()), "note": "explicit list overrides watchlist"}
    if members is None and opend.enabled:
        probe = opend.probe()
        attempts.append({"source": "opend_live", "host": opend.host, "port": opend.port, **probe})
        if probe.get("ok"):
            g = opend.run(["groups"])
            names = [x.get("group_name") for x in (g.get("groups") or [])]
            attempts.append({"step": "groups", "ok": g.get("ok"), "groups": names, "error": g.get("error")})
            if args.group in names:
                time.sleep(3.2)  # OpenD watchlist frequency limit
                m = opend.run(["members", "--group", args.group])
                attempts.append({"step": "members", "ok": m.get("ok"), "error": m.get("error")})
                if m.get("ok"):
                    members = m["members"]
                    source = {"type": "opend_live", "group": args.group, "observed_at": iso(utcnow()),
                              "host": opend.host, "port": opend.port}
            else:
                errors.append({"stage": "universe", "type": "config", "error": f"group {args.group!r} not found in {names}"})
        else:
            errors.append({"stage": "universe", "type": probe.get("error_type", "unknown"),
                           "error": f"OpenD {opend.host}:{opend.port} unavailable: {probe.get('error')}"})
    if members is None:
        f = Path(args.universe_file)
        d = read_json(f)
        snaps = [s for s in d.get("snapshots", []) if s.get("group") == args.group and s.get("status") == "ok"]
        if not snaps:
            raise SystemExit(f"universe file {f} has no ok snapshot for group {args.group!r}")
        members = snaps[0]["members"]
        source = {"type": "repo_snapshot_file", "group": args.group, "path": rel(f), "sha256": sha256_file(f),
                  "observed_at": snaps[0].get("received_at") or d.get("observed_at"),
                  "git_commit": git_commit_of(f), "membership_note": d.get("membership"),
                  "staleness_note": "snapshot of the live group at observed_at; later watchlist edits are unknown"}
    out = []
    for m in members:
        code = str(m["code"])
        cls, in_scope, reason = classify(code, str(m.get("stock_type", "")), str(m.get("name", "")))
        if source["type"] == "explicit_list":
            cls, in_scope, reason = ("unknown_identity", code.startswith("US."), None)
        if source["type"] == "ai_value_chain_screen" and not in_scope:
            mk = code.split(".", 1)[0]
            reason = {"HK": "HK: no HK exchange calendar in this pipeline yet (gap/cutoff checks need one); not fetched",
                      "JP": "JP: market not supported by OpenD quote rights here"}.get(mk, reason)
        ticker = code.split(".", 1)[-1]
        rec = {"security_id": code, "ticker": ticker, "name": m.get("name"), "stock_type": m.get("stock_type"),
               "class": cls, "in_scope": in_scope, "out_of_scope_reason": reason,
               "issuer_id": sec_issuers.get(ticker) if cls == "stock" and code.startswith("US.") else None}
        if m.get("tags"):
            rec["tags"] = m["tags"]
        out.append(rec)
    members_hash = sha256_bytes(canonical_json(sorted(x["security_id"] for x in out)))
    counts: dict = {}
    for x in out:
        counts[x["class"]] = counts.get(x["class"], 0) + 1
    if source["type"] == "ai_value_chain_screen":
        g: dict = {}
        for x in out:
            gr = (x.get("tags") or {}).get("grade")
            g[gr] = g.get(gr, 0) + 1
        counts_extra = {"by_grade": g, "by_market": {}}
        for x in out:
            mk = x["security_id"].split(".", 1)[0]
            counts_extra["by_market"][mk] = counts_extra["by_market"].get(mk, 0) + 1
    else:
        counts_extra = {}
    return {"source": source, "query_attempts": attempts, "members": out, "members_sha256": members_hash,
            "counts": {"total": len(out), "in_scope": sum(x["in_scope"] for x in out), "by_class": counts, **counts_extra}}


def load_ai_basket(args) -> tuple[list, dict]:
    """AI value-chain basket filtered by research-priority grade from quality/screen_current.json."""
    bp, sp = Path(args.ai_basket_file), Path(args.ai_screen_file)
    basket, screen = read_json(bp), read_json(sp)
    recs = screen.get("records", {})
    grades = list(args.grades or [])
    members, missing_grade = [], []
    for m in basket.get("members", []):
        code = str(m["code"])
        r = recs.get(code)
        if r is None:
            missing_grade.append(code)
            continue
        if grades and r.get("grade") not in grades:
            continue
        tags = {"grade": r.get("grade"), "verified_grade": r.get("verified_grade"), "tier": r.get("tier", m.get("tier")),
                "groups": list(r.get("value_groups") or m.get("groups") or []), "primary_group": r.get("primary_group"),
                "value_role": r.get("value_role"), "screen_status": r.get("status")}
        members.append({"code": code, "name": m.get("name") or r.get("name"), "stock_type": "STOCK", "tags": tags})
    groups = screen.get("groups", {})
    group_labels = {k: {"label": v.get("label"), "order": v.get("order"), "count_all": v.get("count")}
                    for k, v in sorted(groups.items(), key=lambda kv: (kv[1].get("order", 9), kv[0]))}
    label = args.universe_label or f"AI 价值链 {'/'.join(grades) if grades else '全部'}"
    source = {"type": "ai_value_chain_screen", "label": label, "grades": grades,
              "basket_path": rel(bp), "basket_sha256": sha256_file(bp), "basket_built_at": basket.get("built_at"),
              "basket_total": len(basket.get("members", [])),
              "screen_path": rel(sp), "screen_sha256": sha256_file(sp), "screen_run_id": screen.get("run_id"),
              "screen_as_of": screen.get("as_of"), "screen_rule_version": screen.get("rule_version"),
              "screen_rating_type": screen.get("rating_type"),
              "screen_grade_counts": (screen.get("summary") or {}).get("grade_counts"),
              "basket_members_without_screen_record": missing_grade,
              "group_labels": group_labels, "observed_at": screen.get("as_of"),
              "note": "grade = research priority (not a quality/return rating); stock_type assumed STOCK for basket members"}
    return members, source


# ------------------------------------------------------------------ per security
def group_intervals(dates: list[str], session_index: dict, reasons: dict) -> list[dict]:
    out = []
    for d in dates:
        if out and session_index[d] == session_index[out[-1]["end"]] + 1:
            out[-1]["end"] = d
            out[-1]["sessions"] += 1
        else:
            out.append({"start": d, "end": d, "sessions": 1, "reason": reasons.get(d, "no_bars_in_any_layer")})
    return out


def process_security(sec: dict, ctx: dict) -> dict:
    cal: Calendar = ctx["cal"]
    sym, sid = sec["ticker"], sec["security_id"]
    required = ctx["required"]
    start, end = required[0], required[-1]
    years = sorted({int(d[:4]) for d in required})
    t0 = time.monotonic()
    layers, rows, problems = [], {}, {}
    prev = ctx["prev_datasets"].get(sid)
    # Previously published capture (our own local data) ...
    prev_rows = {}
    prev_file = resolve_path(prev.get("path")) if prev else None
    if prev_file is not None and prev_file.exists():
        for r in pd.read_parquet(prev_file).to_dict("records"):
            if start <= r["session_date"] <= end:
                prev_rows[r["session_date"]] = r
    # 1) Local 5m
    df, refs = ctx["local"].load(sym, years)
    if df is None:
        layers.append({"layer": "local", "state": "error" if refs else "miss", "base": rel(ctx["local"].base),
                       **({"files": refs} if refs else {})})
    else:
        r, p = derive_daily(df, cal, start, end, ctx["info_cutoff"], "local")
        rows.update(r)
        problems.update(p)
        layers.append({"layer": "local", "state": "hit" if r else "no_sessions_in_range", "files": refs,
                       "sessions": len(r), "first": min(r) if r else None, "last": max(r) if r else None})
    for d, r in prev_rows.items():  # carry earlier fetched rows (e.g. from OpenD) not present locally
        if d not in rows:
            rows[d] = {**r, "layer": f"previous_capture:{r.get('layer')}"}
    missing = [d for d in required if d not in rows]
    wanted = select_wanted(missing, rows, prev, ctx["task"])
    # 2) R2 (same 5m NONE basis)
    if wanted:
        for y in years:
            yw = [d for d in wanted if int(d[:4]) == y]
            if not yw:
                continue
            res = ctx["r2"].check(sym, y, cal.close_at(yw[0]))
            entry = {"layer": "r2", "year": y, **{k: v for k, v in res.items() if k != "path"}}
            if res["state"] == "downloaded":
                r, p = derive_daily(pd.read_parquet(res["path"]), cal, start, end, ctx["info_cutoff"], "r2")
                conflicts = [d for d in r if d in rows and abs(rows[d]["close"] - r[d]["close"]) > 1e-6]
                added = [d for d in r if d not in rows]
                for d in added:
                    rows[d] = r[d]
                entry.update(sessions=len(r), added=len(added), conflicts=conflicts,
                             still_stale=bool(yw and max(r, default="") < yw[-1]))
                for d, why in p.items():
                    problems.setdefault(d, why)
            layers.append(entry)
        missing = [d for d in required if d not in rows]
        wanted = select_wanted(missing, rows, prev, ctx["task"])
    else:
        layers.append({"layer": "r2", "state": "not_needed"})
    # 3) OpenD fallback
    reuse = ctx.get("reuse_raw") and Path(ctx["reuse_raw"]) / f"{sid}.parquet"
    if wanted and reuse and reuse.exists():
        # Raw OpenD 5m (NONE) fetched by an interrupted earlier attempt: copy into this capture, no new request.
        import shutil
        raw = ctx["capture_tmp"] / "opend_5m" / f"{sid}.parquet"
        raw.parent.mkdir(parents=True, exist_ok=True)
        entry = {"layer": "opend", "state": "reused_raw_from_interrupted_attempt", "reused_from": rel(reuse),
                 "raw_fetched_at": iso(datetime.fromtimestamp(reuse.stat().st_mtime).astimezone()),
                 "requested_range": [wanted[0], wanted[-1]]}
        try:
            shutil.copy2(reuse, raw)
            r, p = derive_daily(normalize_opend_5m(raw, sym), cal, start, end, ctx["info_cutoff"], "opend")
            added = [d for d in r if d not in rows]
            for d in added:
                rows[d] = r[d]
            entry.update(added=len(added), raw_path=rel(raw), raw_sha256=sha256_file(raw))
            for d, why in p.items():
                problems.setdefault(d, why)
            layers.append(entry)
            wanted = []
        except Exception as exc:  # unreadable leftover -> fall back to a normal fetch
            entry.update(state="reuse_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            layers.append(entry)
            raw.unlink(missing_ok=True)
    if wanted:
        op: OpenDLayer = ctx["opend"]
        if time.monotonic() > ctx["deadline"]:
            layers.append({"layer": "opend", "state": "skipped_budget"})
        elif not op.available():
            pr = op.probe()
            layers.append({"layer": "opend", "state": "unavailable", "error_type": pr.get("error_type"),
                           "error": pr.get("error"), "host": op.host, "port": op.port})
        else:
            raw = ctx["capture_tmp"] / "opend_5m" / f"{sid}.parquet"
            raw.parent.mkdir(parents=True, exist_ok=True)
            res = op.fetch_5m(sid, wanted[0], wanted[-1], raw)
            entry = {"layer": "opend", **{k: v for k, v in res.items() if k not in ("out",)}}
            if res.get("ok") and res.get("out"):
                r, p = derive_daily(normalize_opend_5m(raw, sym), cal, start, end, ctx["info_cutoff"], "opend")
                added = [d for d in r if d not in rows]
                for d in added:
                    rows[d] = r[d]
                entry.update(state="fetched", added=len(added), raw_path=rel(raw), raw_sha256=sha256_file(raw))
                for d, why in p.items():
                    problems.setdefault(d, why)
            else:
                entry["state"] = "failed"
            layers.append(entry)
    elif not any(l["layer"] == "opend" for l in layers):
        layers.append({"layer": "opend", "state": "not_needed"})
    return finalize_dataset(sec, ctx, rows, problems, layers, t0, prev, prev_rows, None)


def process_security_qfq(sec: dict, ctx: dict) -> dict:
    """qfq_daily mode: exactly one source per security (fresh OpenD K_DAY QFQ or one verified reused file)."""
    sid = sec["security_id"]
    required = ctx["required"]
    start, end = required[0], required[-1]
    t0 = time.monotonic()
    prev = ctx["prev_datasets"].get(sid)
    prev_rows = {}
    prev_file = resolve_path(prev.get("path")) if prev else None
    if prev_file is not None and prev_file.exists():  # only for change reporting; never merged into the series
        for r in pd.read_parquet(prev_file).to_dict("records"):
            if start <= r["session_date"] <= end:
                prev_rows[r["session_date"]] = r
    rows, problems, layers, source = acquire_qfq(sec, ctx)
    return finalize_dataset(sec, ctx, rows, problems, layers, t0, prev, prev_rows, source)


def finalize_dataset(sec, ctx, rows, problems, layers, t0, prev, prev_rows, source) -> dict:
    cal: Calendar = ctx["cal"]
    sid = sec["security_id"]
    required = ctx["required"]
    start, end = required[0], required[-1]
    qfq = ctx.get("price_basis_mode") == "qfq_daily"
    if qfq:  # invariant: one source per security
        layer_names = {r["layer"] for r in rows.values()}
        if len(layer_names) > 1:
            raise RuntimeError(f"{sid}: refusing to splice sources {sorted(layer_names)}")
    # ---------------------------------------------------------- coverage
    idx = {d: i for i, d in enumerate(cal.dates)}
    obs = sorted(rows)
    missing = [d for d in required if d not in rows]
    first, last = (obs[0], obs[-1]) if obs else (None, None)
    pre = [d for d in missing if first is None or d < first]
    trailing = [d for d in missing if last is not None and d > last]
    internal = [d for d in missing if d not in pre and d not in trailing]
    complete_through = None
    if first:
        for d in required[required.index(first):]:
            if d in rows:
                complete_through = d
            else:
                break
    flags = []
    if not obs:
        status = "unavailable"
    else:
        if internal:
            flags.append("missing_intervals")
        if trailing:
            flags.append("stale")
        if len([d for d in required if d >= first]) < FIVE_TRAITS_MIN_CLOSES and pre:
            flags.append("short_history")
        status = flags[0] if flags else "complete"
    daily = [rows[d] for d in obs]
    anomalies = {
        "zero_volume_sessions": [r["session_date"] for r in daily if r["volume"] == 0],
        "partial_regular_bars": [{"session_date": r["session_date"], "bars": r["regular_bars"], "expected": r["expected_bars"]}
                                 for r in daily if r["regular_bars"] < r["expected_bars"]],
        "excluded_days": [{"session_date": d, "reason": why} for d, why in sorted(problems.items())],
        "suspected_unadjusted_splits": suspected_splits(daily),
    }
    sym = sec["ticker"]
    if qfq:
        ca = {"state": "not_applicable", "reason": "QFQ series are already adjusted; R2 split events not applied"}
    else:
        ca = ctx["r2"].corporate_actions(sym) if ctx["check_ca"] else {"state": "not_checked"}
    if ca.get("state") == "ok":
        anomalies["corporate_action_splits"] = split_events(Path(ca["path"]), start, end)
    anomalies["corporate_actions_source"] = {k: v for k, v in ca.items() if k != "path"}
    if qfq:  # in an adjusted series only a split-ratio jump is an adjustment defect; extreme moves stay as flags
        breaks = sorted({x["session_date"] for x in anomalies["suspected_unadjusted_splits"] if x["rule"] == "split_ratio"})
        anomalies["qfq_note"] = "suspected_unadjusted_splits on a QFQ series = adjustment anomaly; split_ratio rule -> return_break"
    else:
        breaks = sorted({x["session_date"] for x in anomalies["suspected_unadjusted_splits"]} |
                        {x["ex_date"] for x in anomalies.get("corporate_action_splits", [])})
    if breaks:
        flags.append("suspected_corporate_action")
    anomalies["return_breaks"] = breaks  # consumers must not build returns across these sessions
    # ---------------------------------------------------------- versioning
    ds = {"dataset_id": f"{ctx.get('ds_prefix', 'us_daily_close_5m_none')}:{sid}", "security_id": sid, "issuer_id": sec.get("issuer_id"),
          "source_role": "required", "kind": "market_daily",
          "requested_range": {"start": start, "end": end, "sessions": len(required)},
          "actual_range": {"first": first, "last": last, "sessions": len(obs)},
          "complete_through": complete_through, "status": status, "flags": flags,
          "pre_history_missing_sessions": len(pre), "trailing_missing": trailing,
          "missing_intervals": group_intervals(internal, idx, problems),
          "anomalies": anomalies, "layers": layers, "checked_at": iso(ctx["started"]),
          "elapsed_seconds": round(time.monotonic() - t0, 3)}
    if qfq:
        ds["price_basis"] = "opend_day_qfq"
        ds["source"] = source
        if source and source.get("rebase_detected_vs_reuse"):
            flags.append("qfq_rebase_detected_vs_reuse")
    if not obs:
        ds.update(version_id=None, content_sha256=None, path=None, file_sha256=None, capture_id=None,
                  received_at=None, change_type="unavailable")
        return ds
    chash = content_hash(daily)
    if prev and prev.get("content_sha256") == chash:
        ds.update({k: prev[k] for k in ("version_id", "content_sha256", "path", "file_sha256", "capture_id", "received_at")})
        ds.update(change_type="unchanged", supersedes=prev.get("supersedes"))
        return ds
    version_id = f"{sid}@{chash[:16]}"
    out = ctx["capture_tmp"] / "daily" / f"{sid}.parquet"
    write_daily_parquet(daily, out)
    final = ctx["capture_final"] / "daily" / f"{sid}.parquet"
    change = {"change_type": "new" if not prev else "extended"}
    if prev and prev_rows:
        revised = [d for d in obs if d in prev_rows and any(abs(float(prev_rows[d][k]) - float(rows[d][k])) > 1e-9
                                                             for k in ("open", "high", "low", "close", "volume"))]
        added = [d for d in obs if d not in prev_rows]
        dropped = [d for d in prev_rows if d not in rows]
        change = {"change_type": "revised" if revised or dropped else "extended", "revised_sessions": revised,
                  "added_sessions": added, "dropped_sessions": dropped}
    ds.update(version_id=version_id, content_sha256=chash, path=rel(final), file_sha256=None,
              capture_id=ctx["capture_id"], received_at=iso(utcnow()),
              supersedes=prev.get("version_id") if prev else None, **change)
    ds["_tmp_path"] = str(out)
    return ds


def resolve_path(p: str | None) -> Path | None:
    if not p:
        return None
    return Path(p) if Path(p).is_absolute() else REPO / p


def select_wanted(missing: list[str], rows: dict, prev: dict | None, task: str) -> list[str]:
    if not missing:
        return []
    last = max(rows) if rows else None
    if task == "init" or last is None:
        return missing
    trailing = [d for d in missing if d > last]
    if task == "incremental":
        return trailing
    # repair: previously recorded gaps + trailing
    gaps = set()
    for iv in (prev or {}).get("missing_intervals", []):
        gaps.update(d for d in missing if iv["start"] <= d <= iv["end"])
    return sorted(gaps | set(trailing))


# ------------------------------------------------------------------ main
def resolve_cutoff(cal: Calendar, arg: str, now: datetime) -> tuple[str, str]:
    if arg == "auto":
        d = cal.last_closed_session(now)
        return d, f"auto: latest session in {cal.calendar_id} with close_at <= run start {iso(now)}"
    if not cal.is_session(arg):
        raise SystemExit(f"--cutoff {arg} is not a session in {cal.calendar_id}")
    if cal.close_at(arg) > now:
        raise SystemExit(f"--cutoff {arg} has not closed yet at {iso(now)} (no future data)")
    return arg, "explicit --cutoff"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", default="auto", help="session date YYYY-MM-DD or 'auto' (latest closed session)")
    ap.add_argument("--task", choices=["init", "incremental", "repair"], default="init")
    ap.add_argument("--observation-mode", choices=["live", "historical_reconstruction"], default="live")
    ap.add_argument("--group", default="特别关注")
    ap.add_argument("--universe-file", default=str(DEFAULT_UNIVERSE_FILE))
    ap.add_argument("--symbols", nargs="*", help="explicit securities; overrides the watchlist")
    ap.add_argument("--universe", choices=["watchlist", "ai_basket"], default="watchlist",
                    help="watchlist = OpenD group/snapshot (--group); ai_basket = AI value-chain basket filtered by --grades")
    ap.add_argument("--ai-basket-file", default=str(DEFAULT_AI_BASKET))
    ap.add_argument("--ai-screen-file", default=str(DEFAULT_AI_SCREEN))
    ap.add_argument("--grades", nargs="*", default=None, help="research-priority grades to keep, e.g. A+ A")
    ap.add_argument("--universe-label", help="page/snapshot label for the universe")
    ap.add_argument("--price-basis", choices=["futu_5m_none", "qfq_daily"], default="futu_5m_none",
                    help="futu_5m_none = daily close derived from 5m NONE bars (default); "
                         "qfq_daily = OpenD daily K, QFQ, one source per security")
    ap.add_argument("--qfq-reuse-dir", action="append", default=[],
                    help="read-only dir of existing OpenD QFQ daily files <security_id>.parquet (repeatable)")
    ap.add_argument("--qfq-reuse-policy", choices=["verify_free", "prefer_reuse", "never"], default="verify_free",
                    help="verify_free: fetch fresh when the security is already in the 7-day quota window (free) and "
                         "compare with reuse candidates; otherwise reuse a verified candidate; otherwise fetch (budget)")
    ap.add_argument("--quota-max-new", type=int, default=260, help="max new securities charged to the history-kline quota")
    ap.add_argument("--quota-min-remaining", type=int, default=400, help="never leave fewer remaining quota than this")
    ap.add_argument("--quota-on-exceed", choices=["abort", "cap"], default="abort",
                    help="abort: stop before any fetch when the plan exceeds the budget; cap: fetch until the cap, skip the rest")
    ap.add_argument("--history-sessions", type=int, default=DEFAULT_HISTORY_SESSIONS)
    ap.add_argument("--budget-seconds", type=float, default=240)
    ap.add_argument("--r2-timeout", type=float, default=5)
    ap.add_argument("--no-r2", action="store_true")
    ap.add_argument("--no-corporate-actions", action="store_true")
    ap.add_argument("--opend-host", default="127.0.0.1")
    ap.add_argument("--opend-port", type=int, default=11111)
    ap.add_argument("--opend-timeout", type=float, default=20)
    ap.add_argument("--no-opend", action="store_true")
    ap.add_argument("--reuse-opend-raw-dir", help="opend_5m/ (or opend_day_qfq/ in qfq_daily mode) dir of an interrupted "
                                                  "attempt; its raw files are copied into this capture instead of re-requesting OpenD")
    ap.add_argument("--calendar", default=str(DEFAULT_CALENDAR))
    ap.add_argument("--local-5m-dir", default=str(DEFAULT_LOCAL_5M))
    ap.add_argument("--capture-root", default=str(DEFAULT_CAPTURE_ROOT))
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--sec-facts", default=str(DEFAULT_SEC_FACTS))
    ap.add_argument("--now", help=argparse.SUPPRESS)  # fixtures only
    return ap.parse_args(argv)


def run(args, r2_factory=None, opend_runner=None) -> dict:
    started = parse_ts(args.now) if args.now else utcnow()
    t_start = time.monotonic()
    deadline = t_start + args.budget_seconds
    data_root, capture_root = Path(args.data_root), Path(args.capture_root)
    data_run_id = new_id("data", started)
    capture_id = f"cap-{data_run_id[5:]}"
    stages, errors = {}, []
    ledger = data_root / "ledger.jsonl"
    attempt = {"data_run_id": data_run_id, "started_at": iso(started), "argv": sys.argv[1:], "task": args.task}
    assert_own_data_root(data_root)  # before lock/ledger: never write into another pipeline's root
    with run_lock(data_root / "backfill.lock"):
        try:
            cal = Calendar(Path(args.calendar))
            cutoff, cutoff_basis = resolve_cutoff(cal, args.cutoff, started)
            required = cal.sessions_through(cutoff, args.history_sessions)
            if len(required) < args.history_sessions:
                errors.append({"stage": "plan", "type": "calendar_missing",
                               "error": f"calendar {cal.calendar_id} starts {cal.start_date}; only {len(required)} sessions"})
            if r2_factory is None:
                sys.path.insert(0, str(REPO / "scripts"))
                from r2_client import R2Client
                r2_factory = lambda: R2Client(timeout=args.r2_timeout, deadline=deadline)  # noqa: E731
            r2 = R2Layer(r2_factory, "us_5m", data_root / "r2_cache", enabled=not args.no_r2)
            opend = OpenDLayer(args.opend_host, args.opend_port, args.opend_timeout, enabled=not args.no_opend,
                               runner=opend_runner)
            mode = getattr(args, "price_basis", "futu_5m_none")
            qfq = mode == "qfq_daily"
            if qfq:
                opend.probe_detail = True
            ts = time.monotonic()
            universe = build_universe(args, opend, errors)
            stages["universe_seconds"] = round(time.monotonic() - ts, 3)
            prev_manifest, prev_ref = None, None
            cur = data_root / "current.json"
            if cur.exists():
                prev_ref = read_json(cur)
                prev_manifest = read_json(data_root / prev_ref["manifest_path"])
            basis = PRICE_BASIS_QFQ if qfq else PRICE_BASIS
            same_basis = (prev_manifest or {}).get("requirements", {}).get("market_daily", {}).get("price_basis") == basis
            prev_datasets = {d["security_id"]: d for d in (prev_manifest or {}).get("datasets", [])
                             if d.get("kind") == "market_daily" and d.get("version_id")} if same_basis else {}
            capture_tmp = capture_root / f".tmp-{capture_id}"
            capture_final = capture_root / capture_id
            ctx = dict(cal=cal, required=required, info_cutoff=started, local=LocalLayer(Path(args.local_5m_dir)),
                       r2=r2, opend=opend, prev_datasets=prev_datasets, task=args.task, deadline=deadline,
                       capture_tmp=capture_tmp, capture_final=capture_final, capture_id=capture_id,
                       started=started, check_ca=not args.no_corporate_actions,
                       reuse_raw=getattr(args, "reuse_opend_raw_dir", None), price_basis_mode=mode,
                       ds_prefix=DS_PREFIX[mode], reuse_qfq=ReuseQfq(getattr(args, "qfq_reuse_dir", None)),
                       qfq_reuse_policy=getattr(args, "qfq_reuse_policy", "verify_free"), quota=None)
            if qfq:
                pr = opend.probe() if opend.enabled else {"ok": False, "error": "disabled"}
                ctx["quota"] = QuotaBudget(pr.get("history_kl_quota") if pr.get("ok") else None,
                                           args.quota_max_new, args.quota_min_remaining, started)
            # Stage: plan (local-only view of what is missing; no remote requests)
            ts = time.monotonic()
            plan = {"data_run_id": data_run_id, "cutoff_session": cutoff, "required_range": [required[0], required[-1]],
                    "sessions": len(required), "task": args.task, "securities": []}
            for sec in universe["members"]:
                if not sec["in_scope"]:
                    continue
                if qfq:
                    cands = ctx["reuse_qfq"].candidates(sec["security_id"], cal, required[0], required[-1], started)
                    good = [c for c in cands if c["state"] == "ok"]
                    free = ctx["quota"].is_free(sec["security_id"])
                    pol = ctx["qfq_reuse_policy"]
                    fetch = pol == "never" or not good or (pol == "verify_free" and free)
                    resume = ctx["reuse_raw"] and (Path(ctx["reuse_raw"]) / f"{sec['security_id']}.parquet").exists()
                    plan["securities"].append({"security_id": sec["security_id"], "reuse_ok": len(good),
                                               "reuse_suspicious": [c["suspicious"] for c in cands if c["state"] != "ok"],
                                               "in_quota_window": free, "fetch_fresh": bool(fetch),
                                               "resume_raw": bool(resume),
                                               "new_quota": bool(fetch and not free and not resume)})
                    continue
                df, _ = ctx["local"].load(sec["ticker"], sorted({int(d[:4]) for d in required}))
                have = set()
                if df is not None:
                    have = set(derive_daily(df, cal, required[0], required[-1], started, "local")[0])
                miss = [d for d in required if d not in have]
                plan["securities"].append({"security_id": sec["security_id"], "local_sessions": len(have),
                                           "missing_sessions": len(miss),
                                           "missing_first": miss[0] if miss else None,
                                           "missing_last": miss[-1] if miss else None})
            stages["plan_seconds"] = round(time.monotonic() - ts, 3)
            if qfq:
                planned_new = sum(1 for x in plan["securities"] if x["new_quota"])
                plan["quota"] = {**ctx["quota"].summary(), "planned_new": planned_new,
                                 "planned_fresh_fetches": sum(1 for x in plan["securities"] if x["fetch_fresh"]),
                                 "planned_reuse": sum(1 for x in plan["securities"] if not x["fetch_fresh"])}
                ok, why = (ctx["quota"].plan_ok(planned_new) if opend.available()
                            else (True, "opend unavailable: nothing will be fetched"))
                plan["quota"]["plan_check"] = why
                if not ok and args.quota_on_exceed == "abort":
                    attempt["plan_quota"] = plan["quota"]
                    raise QuotaBudgetExceeded(why)
                if not ok:
                    errors.append({"stage": "plan", "type": "quota_budget", "error": why + " (cap mode: fetch until cap)"})
            ts = time.monotonic()
            datasets = []
            for sec in universe["members"]:
                if not sec["in_scope"]:
                    datasets.append({"dataset_id": f"{ctx['ds_prefix']}:{sec['security_id']}",
                                     "security_id": sec["security_id"], "kind": "market_daily",
                                     "source_role": "required", "status": "unsupported",
                                     "reason": sec["out_of_scope_reason"], "version_id": None})
                    continue
                if time.monotonic() > deadline:
                    p = prev_datasets.get(sec["security_id"])
                    d = {**p, "change_type": "carried_forward", "carried_reason": "budget_exhausted",
                         "checked_at": None} if p else {
                        "dataset_id": f"{ctx['ds_prefix']}:{sec['security_id']}", "security_id": sec["security_id"],
                        "kind": "market_daily", "source_role": "required", "status": "unavailable",
                        "reason": "budget_exhausted_before_processing", "version_id": None}
                    datasets.append(d)
                    continue
                try:
                    datasets.append(process_security_qfq(sec, ctx) if qfq else process_security(sec, ctx))
                except Exception as exc:
                    errors.append({"stage": "acquire", "security_id": sec["security_id"], "type": type(exc).__name__,
                                   "error": str(exc)[:400]})
                    p = prev_datasets.get(sec["security_id"])
                    datasets.append({**p, "change_type": "carried_forward", "carried_reason": "error"} if p else {
                        "dataset_id": f"{ctx['ds_prefix']}:{sec['security_id']}", "security_id": sec["security_id"],
                        "kind": "market_daily", "source_role": "required", "status": "unavailable",
                        "reason": f"error: {type(exc).__name__}", "version_id": None})
            stages["acquire_seconds"] = round(time.monotonic() - ts, 3)
            quota_info = None
            if qfq:
                b = ctx["quota"]
                if b.skipped:
                    errors.append({"stage": "acquire", "type": "quota_budget",
                                   "error": f"{len(b.skipped)} securities skipped by quota budget", "skipped": b.skipped[:50]})
                after = opend.quota_now() if (opend.enabled and opend.available()) else {"error": "opend unavailable"}
                quota_info = {**b.summary(), "after": after,
                              "observed_new_used": (after.get("used") - b.used_before)
                              if isinstance(after.get("used"), int) and isinstance(b.used_before, int) else None,
                              "window": "rolling 7 days per Futu docs; repeat requests for one security not re-counted"}
            # Auxiliary sources
            datasets.extend(aux_datasets(universe, Path(args.sec_facts), started))
            # Stage: validate + publish capture
            ts = time.monotonic()
            new_files = [d for d in datasets if d.get("_tmp_path")]
            for d in new_files:
                tmp = Path(d.pop("_tmp_path"))
                back = pd.read_parquet(tmp).to_dict("records")
                if content_hash(back) != d["content_sha256"]:
                    raise RuntimeError(f"capture validation failed for {d['security_id']}")
                d["file_sha256"] = sha256_file(tmp)
            if new_files:
                publish_dir(capture_tmp, capture_final)
                make_readonly(capture_final)
            elif capture_tmp.exists():
                import shutil
                shutil.rmtree(capture_tmp)
            stages["capture_publish_seconds"] = round(time.monotonic() - ts, 3)
            # Changes
            changes = [{"dataset_id": d["dataset_id"], "security_id": d["security_id"], "change_type": d["change_type"],
                        "new_version": d.get("version_id"), "old_version": d.get("supersedes"),
                        "revised_sessions": d.get("revised_sessions", []),
                        "added_sessions_count": len(d.get("added_sessions", []))}
                       for d in datasets if d.get("kind") == "market_daily" and d.get("change_type") in ("new", "extended", "revised")]
            legacy = {**ctx["local"].read_hashes, **ctx["reuse_qfq"].read_hashes}
            legacy_ok = all(sha256_file(Path(p)) == h for p, h in legacy.items())
            md = [d for d in datasets if d.get("kind") == "market_daily" and d.get("source_role") == "required"]
            in_scope = [d for d in md if d.get("status") != "unsupported"]
            status = "complete" if in_scope and all(d.get("status") == "complete" for d in in_scope) else "partial"
            finished = utcnow() if not args.now else started
            manifest = {
                "schema_version": MANIFEST_SCHEMA, "contract_version": "data-contract v1", "data_run_id": data_run_id,
                "observation_mode": args.observation_mode, "task": args.task,
                "requested_cutoff": cal.close_at(cutoff).astimezone(cal.close_at(cutoff).tzinfo).isoformat(),
                "requested_cutoff_et": cal.by_date[cutoff]["close_at_et"], "cutoff_session": cutoff,
                "cutoff_basis": cutoff_basis, "info_cutoff": iso(started),
                "started_at": iso(started), "started_at_shanghai": iso_sh(started),
                "finished_at": iso(finished), "status": status,
                "universe": universe,
                "requirements": {
                    "market_daily": {"market": "US", "granularity": "1d (OpenD daily K)" if qfq else "1d (derived from 5m)",
                                     "price_basis": basis, "price_basis_mode": mode,
                                     **({"qfq_reuse_dirs": [rel(p) for p in ctx["reuse_qfq"].dirs],
                                         "qfq_reuse_policy": ctx["qfq_reuse_policy"],
                                         "source_rule": "one source and one adjustment basis per security; no splicing"}
                                        if qfq else {}),
                                     "calendar": {"id": cal.calendar_id, "path": rel(cal.path), "sha256": cal.sha256,
                                                  "covers": [cal.start_date, cal.end_date]},
                                     "history_sessions": args.history_sessions, "range": [required[0], required[-1]],
                                     "role": "required", "warmup_note": "five_traits_v2 needs 141 consecutive closes (S); default window adds 20 sessions margin; a full 60-calendar-day reconstructed curve needs ~42 more sessions"},
                    "news": {"role": "auxiliary", "status": "unsupported", "reason": "no news channel connected in repo/box"},
                    "financials": {"role": "auxiliary", "status": "reused_old_cache",
                                   "reason": "SEC companyfacts cache reused read-only; not re-downloaded"}},
                "datasets": datasets, "changes": changes,
                "legacy_inputs": {"files_read": len(legacy), "unchanged_after_run": legacy_ok,
                                  "sha256": {rel(Path(p)): h for p, h in legacy.items()}},
                "previous_manifest": prev_ref,
                "resources": {"stages_seconds": stages, "total_seconds": round(time.monotonic() - t_start, 3),
                              "budget_seconds": args.budget_seconds,
                              "r2": {**r2.counters.__dict__, "enabled": r2.enabled},
                              "opend": {**opend.counters.__dict__, "probe": _slim_probe(opend.probe_result)},
                              **({"history_kl_quota": quota_info} if qfq else {}),
                              "known_costs": {"r2": "unknown (operation/egress pricing not metered here)",
                                              "opend": "history-kline quota before run: see opend.probe.history_kl_quota (used/remain, distinct securities in the rolling 7-day window; repeat requests for the same security are not re-counted); fees n/a", "llm_tokens": 0}},
                "errors": errors,
                "summary": summarize(datasets),
            }
            run_tmp = data_root / "runs" / f".tmp-{data_run_id}"
            write_json_new(run_tmp / "plan.json", plan)
            write_json_new(run_tmp / "manifest.json", manifest)
            validate_manifest(run_tmp / "manifest.json")
            final_dir = data_root / "runs" / data_run_id
            publish_dir(run_tmp, final_dir)
            make_readonly(final_dir)
            mpath = final_dir / "manifest.json"
            write_json_atomic(cur, {"data_run_id": data_run_id, "manifest_path": rel(mpath, data_root),
                                    "manifest_sha256": sha256_file(mpath), "published_at": iso(utcnow()),
                                    "status": status,
                                    "note": "points to latest published manifest; does not imply all data complete"})
            attempt.update(status=status, manifest=rel(mpath), finished_at=iso(utcnow()))
            append_jsonl(ledger, attempt)
            return {"manifest_path": str(mpath), "data_run_id": data_run_id, "status": status, "manifest": manifest}
        except BaseException as exc:
            attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=iso(utcnow()),
                           traceback=traceback.format_exc()[-1500:])
            append_jsonl(ledger, attempt)
            raise


class QuotaBudgetExceeded(RuntimeError):
    pass


def _slim_probe(p):
    if not p:
        return p
    q = dict(p)
    if isinstance(q.get("history_kl_quota"), dict):
        hq = dict(q["history_kl_quota"])
        if "codes" in hq:
            hq["codes_count"] = len(hq.pop("codes"))
        hq.pop("request_times", None)
        q["history_kl_quota"] = hq
    return q


def aux_datasets(universe: dict, sec_path: Path, started) -> list[dict]:
    out = []
    issuers = sorted({m["issuer_id"] for m in universe["members"] if m.get("issuer_id")})
    unknown_issuer = [m["security_id"] for m in universe["members"] if m["in_scope"] and not m.get("issuer_id")]
    out.append({"dataset_id": "news:all_issuers", "kind": "news", "source_role": "auxiliary", "status": "unsupported",
                "reason": "未接入: no news/announcement channel connected; not the same as 'checked, no news'",
                "issuers": issuers, "securities_without_issuer_id": unknown_issuer, "version_id": None,
                "checked_at": None})
    if sec_path.exists():
        d = read_json(sec_path)
        covered = [i for i in issuers]
        out.append({"dataset_id": "financials:sec_companyfacts_annual", "kind": "financials", "source_role": "auxiliary",
                    "status": "reused_old_cache", "path": rel(sec_path), "file_sha256": sha256_file(sec_path),
                    "version_id": f"sec_facts@{sha256_file(sec_path)[:16]}", "received_at": d.get("retrieved_at_utc"),
                    "checked_at": iso(started), "issuers_covered": covered,
                    "note": "old cache reused; no new filings fetched in this run (not a daily financial update)"})
    return out


def summarize(datasets: list[dict]) -> dict:
    md = [d for d in datasets if d.get("kind") == "market_daily"]
    by_status, by_change = {}, {}
    for d in md:
        by_status[d.get("status")] = by_status.get(d.get("status"), 0) + 1
        ct = d.get("change_type") or "n/a"
        by_change[ct] = by_change.get(ct, 0) + 1
    cts = sorted({d.get("complete_through") for d in md if d.get("complete_through")})
    out = {"market_daily_by_status": by_status, "market_daily_by_change": by_change,
           "complete_through_values": cts}
    if any("source" in d for d in md):
        by_src: dict = {}
        rebase = []
        for d in md:
            src = d.get("source") or {}
            k = src.get("type") or ("unsupported" if d.get("status") == "unsupported" else "none")
            if k == "reused_file":
                k = f"reused_file:{src.get('layer', '').split(':', 1)[-1]}"
            if k == "opend_fresh" and src.get("reused_raw_from"):
                k = "opend_fresh(resumed_raw)"
            by_src[k] = by_src.get(k, 0) + 1
            if src.get("rebase_detected_vs_reuse"):
                rebase.append(d["security_id"])
        out.update(market_daily_by_source=by_src, qfq_rebase_detected_vs_reuse=rebase)
    return out


def validate_manifest(path: Path) -> None:
    m = read_json(path)
    assert m["schema_version"] == MANIFEST_SCHEMA
    for d in m["datasets"]:
        if d.get("kind") == "market_daily" and d.get("version_id") and d.get("path"):
            target = resolve_path(d["path"])
            if not target.exists():
                raise RuntimeError(f"manifest references missing file {d['path']}")
            if d.get("file_sha256") and sha256_file(target) != d["file_sha256"]:
                raise RuntimeError(f"hash mismatch for {d['path']}")
            if d["actual_range"]["last"] and d["actual_range"]["last"] > m["cutoff_session"]:
                raise RuntimeError(f"future data in {d['dataset_id']}")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        res = run(args)
    except AlreadyRunning as exc:
        print(json.dumps({"status": "skipped", "reason": str(exc)}, ensure_ascii=False))
        return 3
    except QuotaBudgetExceeded as exc:
        print(json.dumps({"status": "aborted_quota_budget", "reason": str(exc)}, ensure_ascii=False))
        return 5
    m = res["manifest"]
    print(json.dumps({"data_run_id": res["data_run_id"], "status": res["status"], "manifest": res["manifest_path"],
                      "cutoff_session": m["cutoff_session"], "requested_cutoff": m["requested_cutoff"],
                      "universe_source": m["universe"]["source"], "universe_counts": m["universe"]["counts"],
                      "summary": m["summary"], "errors": m["errors"][:10],
                      "legacy_inputs_unchanged": m["legacy_inputs"]["unchanged_after_run"],
                      "resources": m["resources"]}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
