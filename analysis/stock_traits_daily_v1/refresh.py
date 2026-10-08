#!/usr/bin/env python3
"""stock-report-refresh v1: five_traits_v2 snapshot + ledger + rendered page from a published data manifest.

Example:
  python analysis/stock_traits_daily_v1/refresh.py               # uses .cache/stock_data_v1/current.json
  python analysis/stock_traits_daily_v1/refresh.py --manifest <path> --evaluation-cutoff 2026-10-06 --mode live

The renderer never calls market/news APIs. Missing data produce `backfill_requests`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import traits as T  # noqa: E402
from common import (DEFAULT_DATA_ROOT, DEFAULT_REPORT_ROOT, REPO, SNAPSHOT_SCHEMA, AlreadyRunning, Calendar, assert_own_data_root,  # noqa: E402
                    append_jsonl, canonical_json, iso, iso_sh, make_readonly, new_id, parse_ts, publish_dir,
                    read_json, read_jsonl, rel, run_lock, sha256_bytes, sha256_file, utcnow, write_json_new)


def resolve(p: str | None, base: Path = REPO) -> Path | None:
    if not p:
        return None
    return Path(p) if Path(p).is_absolute() else base / p


def unknown_all(reason: str) -> dict:
    return {t: {"value": None, "status": "unknown", "reason": reason, "unit": T.UNITS[t], "category": None,
                "display_0_100": None} for t in T.TRAITS}


BASIS_5M = "Futu 5m NONE regular-session close (unadjusted USD)"
BASIS_QFQ = "OpenD daily K QFQ (forward-adjusted; one source per security)"


def basis_label(manifest: dict) -> tuple[str, str]:
    md = (manifest.get("requirements") or {}).get("market_daily") or {}
    if md.get("price_basis_mode") == "qfq_daily":
        return BASIS_QFQ, "1d OpenD daily K (QFQ)"
    return BASIS_5M, "1d derived from 5m"


def source_label(ds: dict | None) -> dict | None:
    src = (ds or {}).get("source")
    if not src:
        return None
    return {"type": src.get("type"), "layer": src.get("layer"), "path": src.get("path") or src.get("raw_path"),
            "rebase_detected_vs_reuse": src.get("rebase_detected_vs_reuse"),
            "rebase_checks": len(src.get("rebase_checks") or [])}


def evaluate_security(member: dict, ds: dict | None, cal: Calendar, E: str, args, eval_info_time,
                      basis: str = BASIS_5M) -> tuple[dict, dict | None]:
    sid = member["security_id"]
    base = {"security_id": sid, "ticker": member["ticker"], "name": member.get("name"), "class": member["class"],
            "issuer_id": member.get("issuer_id"), "tags": member.get("tags"), "data_source": source_label(ds),
            "news": {"status": "not_connected", "value": None, "note": "未接入：无新闻渠道，不代表无事件"},
            "valuation": {"status": "not_connected", "value": None, "note": "未接入：无版本化估值公式"},
            "business_rating": {"status": "not_connected", "value": None, "note": "未运行 ai-stock-rating；无沿用评级"},
            "composite_score": None}
    req = None
    if not member["in_scope"]:
        return {**base, "status": "unknown", "unknown_reason": "out_of_scope_market_or_identity",
                "detail": member.get("out_of_scope_reason"), "traits": unknown_all("out_of_scope"), "map": None,
                "inputs": None, "price": None}, None
    if ds is None or not ds.get("version_id"):
        why = (ds or {}).get("status", "no_dataset")
        req = {"security_id": sid, "issuer_id": member.get("issuer_id"), "source": "market_daily", "granularity": "1d",
               "needed_closes": 141, "target_cutoff": E, "existing_dataset": (ds or {}).get("dataset_id"),
               "reason": f"dataset {why}: " + "; ".join(f"{l['layer']}={l.get('state')}" for l in (ds or {}).get("layers", []))}
        return {**base, "status": "unknown", "unknown_reason": "no_price_data", "detail": req["reason"],
                "traits": unknown_all("no_price_data"), "map": None, "inputs": None, "price": None}, req
    path = resolve(ds["path"])
    inputs = {"dataset_id": ds["dataset_id"], "version_id": ds["version_id"], "content_sha256": ds["content_sha256"],
              "path": ds["path"], "file_sha256": ds.get("file_sha256"), "capture_id": ds.get("capture_id"),
              "received_at": ds.get("received_at")}
    if not path.exists() or (ds.get("file_sha256") and sha256_file(path) != ds["file_sha256"]):
        return {**base, "status": "unknown", "unknown_reason": "input_hash_mismatch", "traits": unknown_all("input_hash_mismatch"),
                "map": None, "inputs": inputs, "price": None}, None
    df = pd.read_parquet(path)
    df = df[df["session_date"] <= E]  # data after the evaluation cutoff are excluded
    if args.mode == "live":
        av = pd.to_datetime(df["available_at"].astype(str), utc=True, format="ISO8601")
        df = df[av <= pd.Timestamp(eval_info_time)]
    series = list(zip(df["session_date"], df["close"].astype(float)))
    if not series:
        return {**base, "status": "unknown", "unknown_reason": "no_price_before_cutoff",
                "traits": unknown_all("no_price_before_cutoff"), "map": None, "inputs": inputs, "price": None}, None
    asof = max(d for d, _ in series)
    lag_sessions = cal.sessions_between(asof, E)[1:]
    price = {"price_asof": asof, "evaluation_cutoff": E, "stale_sessions": lag_sessions,
             "complete_through": ds.get("complete_through"), "dataset_status": ds.get("status"),
             "price_basis": basis}
    if lag_sessions:
        req = {"security_id": sid, "issuer_id": member.get("issuer_id"), "source": "market_daily", "granularity": "1d",
               "missing_sessions": lag_sessions, "target_cutoff": E, "existing_dataset": ds["dataset_id"],
               "existing_version": ds["version_id"], "reason": "stale: " + "; ".join(
                   f"{l['layer']}={l.get('state')}" for l in ds.get("layers", []))}
        if args.stale_policy == "unknown":
            return {**base, "status": "unknown", "unknown_reason": "stale_price", "traits": unknown_all("stale_price"),
                    "map": None, "inputs": inputs, "price": price}, req
    breaks = (ds.get("anomalies") or {}).get("return_breaks") or []
    res = T.evaluate(series, cal.dates, asof, sid, reps=args.bootstrap, breaks=breaks)
    ok = [t for t in T.TRAITS if res["traits"][t]["status"] == "ok"]
    status = "evaluated" if len(ok) == 5 else ("partial" if ok else "unknown")
    out = {**base, "status": status, "price_stale": bool(lag_sessions), "traits": res["traits"], "map": res["map"],
           "contiguous_closes": res.get("contiguous_closes"), "window_first_session": res.get("window_first_session"),
           "bootstrap_seed": res.get("bootstrap_seed"), "return_breaks": breaks, "inputs": inputs, "price": price}
    if status == "unknown":
        out["unknown_reason"] = res["traits"]["G"]["reason"] or "unknown"
    return out, req


def idempotency_key(mode, E, members_sha, inputs, params) -> str:
    return sha256_bytes(canonical_json({"mode": mode, "evaluation_cutoff": E, "members_sha256": members_sha,
                                        "inputs": inputs, "params": params}))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--manifest", help="published manifest.json (default: data-root/current.json)")
    ap.add_argument("--evaluation-cutoff", help="session date; default = manifest cutoff_session")
    ap.add_argument("--mode", choices=["live", "historical_reconstruction"], default="live")
    ap.add_argument("--stale-policy", choices=["label", "unknown"], default="label",
                    help="label: compute on last complete close and show its date; unknown: mark stale prices unknown")
    ap.add_argument("--bootstrap", type=int, default=256)
    ap.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--now", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def run(args) -> dict:
    t0 = time.monotonic()
    now = parse_ts(args.now) if args.now else utcnow()
    data_root, report_root = Path(args.data_root), Path(args.report_root)
    ledger = report_root / "ledger.jsonl"
    attempt_id = new_id("attempt", now)
    if not args.manifest:
        assert_own_data_root(data_root)
    with run_lock(report_root / "refresh.lock"):
        mpath = Path(args.manifest) if args.manifest else data_root / read_json(data_root / "current.json")["manifest_path"]
        manifest = read_json(mpath)
        msha = sha256_file(mpath)
        E = args.evaluation_cutoff or manifest["cutoff_session"]
        cal_ref = manifest["requirements"]["market_daily"]["calendar"]
        cal = Calendar(resolve(cal_ref["path"]))
        if cal.sha256 != cal_ref["sha256"]:
            raise SystemExit("calendar hash differs from manifest")
        if not cal.is_session(E):
            raise SystemExit(f"evaluation cutoff {E} is not a session")
        if args.mode == "live" and cal.close_at(E) > now:
            raise SystemExit("live evaluation cutoff in the future")
        if E > manifest["cutoff_session"]:
            note = f"evaluation cutoff {E} later than manifest cutoff {manifest['cutoff_session']}"
        else:
            note = None
        members = manifest["universe"]["members"]
        basis, granularity = basis_label(manifest)
        dsmap = {d["security_id"]: d for d in manifest["datasets"] if d.get("kind") == "market_daily"}
        inputs = sorted([d["security_id"], d.get("content_sha256"), (d.get("anomalies") or {}).get("return_breaks") or []]
                        for d in dsmap.values())
        params = {**T.PARAMS, "bootstrap_replicates_run": args.bootstrap, "stale_policy": args.stale_policy,
                  "impl": T.IMPL_VERSION}
        key = idempotency_key(args.mode, E, manifest["universe"]["members_sha256"], inputs, params)
        records = read_jsonl(ledger)
        prior = [r for r in records if r.get("idempotency_key") == key and r.get("snapshot_published")]
        if prior:
            rec = {"attempt_id": attempt_id, "kind": "refresh", "status": "reused", "idempotency_key": key,
                   "report_run_id": prior[0]["report_run_id"], "data_run_id": manifest["data_run_id"],
                   "attempted_at": iso(now), "snapshot_published": False,
                   "note": "same idempotency key: existing snapshot reused, no new point"}
            append_jsonl(ledger, rec)
            render_res = None if args.no_render else safe_render(report_root, ledger, attempt_id)
            return {"report_run_id": prior[0]["report_run_id"], "reused": True, "render": render_res,
                    "snapshot_path": prior[0]["snapshot_path"]}
        report_run_id = new_id("report", now)
        tc = time.monotonic()
        results, requests = [], []
        for m in members:
            r, req = evaluate_security(m, dsmap.get(m["security_id"]), cal, E, args, now, basis)
            results.append(r)
            if req:
                requests.append(req)
        compute_s = round(time.monotonic() - tc, 3)
        # previous comparable snapshot (same mode, earlier or same cutoff)
        published = [r for r in records if r.get("snapshot_published") and r.get("mode") == args.mode]
        same_cutoff = [r for r in published if r.get("evaluation_cutoff") == E]
        earlier = sorted([r for r in published if r.get("evaluation_cutoff") < E], key=lambda r: r["evaluation_cutoff"])
        supersedes = same_cutoff[-1]["report_run_id"] if same_cutoff else None
        changes = []
        base_ref = same_cutoff[-1] if same_cutoff else (earlier[-1] if earlier else None)
        if base_ref:
            old = {x["security_id"]: x for x in read_json(resolve(base_ref["snapshot_path"], report_root))["results"]}
            for x in results:
                o = old.get(x["security_id"])
                if not o:
                    changes.append({"security_id": x["security_id"], "change": "new_member"})
                    continue
                diff = {t: [o["traits"][t].get("category"), x["traits"][t].get("category")] for t in T.TRAITS
                        if o["traits"][t].get("category") != x["traits"][t].get("category")}
                if diff:
                    changes.append({"security_id": x["security_id"], "category_changes": diff})
        counts = {"members": len(results),
                  "evaluated_all_five": sum(x["status"] == "evaluated" for x in results),
                  "partial": sum(x["status"] == "partial" for x in results),
                  "unknown": sum(x["status"] == "unknown" for x in results),
                  "with_map_coordinates": sum(x["map"] is not None for x in results),
                  "price_stale": sum(bool(x.get("price_stale")) for x in results)}
        reasons = {}
        for x in results:
            if x["status"] == "unknown":
                reasons[x.get("unknown_reason")] = reasons.get(x.get("unknown_reason"), 0) + 1
        trait_unknown = {t: {} for t in T.TRAITS}
        for x in results:
            for t in T.TRAITS:
                if x["traits"][t]["status"] != "ok":
                    rr = x["traits"][t]["reason"]
                    trait_unknown[t][rr] = trait_unknown[t].get(rr, 0) + 1
        price_asofs = sorted({x["price"]["price_asof"] for x in results if x.get("price")})
        snapshot = {
            "schema_version": SNAPSHOT_SCHEMA, "report_run_id": report_run_id, "attempt_id": attempt_id,
            "idempotency_key": key, "observation_mode": args.mode, "evaluation_cutoff": E,
            "evaluation_cutoff_close_at": cal.close_at(E).isoformat(), "computed_at": iso(now),
            "computed_at_shanghai": iso_sh(now), "supersedes": supersedes,
            "data": {"data_run_id": manifest["data_run_id"], "manifest_path": rel(mpath), "manifest_sha256": msha,
                     "manifest_status": manifest["status"], "note": note},
            "source_cutoffs": {"market_daily_price_asof_values": price_asofs,
                               "market_daily_complete_through_values": manifest["summary"].get("complete_through_values"),
                               "news": None, "financials": next((d.get("received_at") for d in manifest["datasets"]
                                                                  if d.get("kind") == "financials"), None)},
            "universe": {k: manifest["universe"][k] for k in ("source", "members_sha256", "counts")},
            "price_basis": {"label": basis, "mode": ((manifest.get("requirements") or {}).get("market_daily") or {}).get("price_basis_mode", "futu_5m_none"),
                            "by_source": manifest["summary"].get("market_daily_by_source"),
                            "qfq_rebase_detected_vs_reuse": manifest["summary"].get("qfq_rebase_detected_vs_reuse")},
            "model": params, "results": results, "counts": counts, "unknown_reasons": reasons,
            "trait_unknown_reasons": trait_unknown, "changes": changes, "backfill_requests": requests,
            "intervals": {"data_granularity": granularity, "price_basis": basis, "backfill_interval": "per run (no scheduler)",
                          "refresh_interval": "per run (no scheduler)", "windows_returns": {"G": 120, "V/M/J": 60, "S": "20 changes / 140 returns"}},
            "resources": {"compute_seconds": compute_s, "model_calls": 0, "tokens": 0, "cost": "0 known; no paid API used"},
            "disclaimer": "价格描述统计，不是投资建议或未来概率；类别频率为重采样条件倾向，未校准。",
        }
        tmp = report_root / "snapshots" / f".tmp-{report_run_id}"
        write_json_new(tmp / "snapshot.json", snapshot)
        chk = read_json(tmp / "snapshot.json")
        assert chk["idempotency_key"] == key and len(chk["results"]) == len(members)
        final = report_root / "snapshots" / report_run_id
        publish_dir(tmp, final)
        make_readonly(final)
        spath = final / "snapshot.json"
        rec = {"attempt_id": attempt_id, "kind": "refresh", "status": "computed", "idempotency_key": key,
               "report_run_id": report_run_id, "data_run_id": manifest["data_run_id"], "mode": args.mode,
               "evaluation_cutoff": E, "attempted_at": iso(now), "snapshot_published": True,
               "snapshot_path": rel(spath, report_root), "snapshot_sha256": sha256_file(spath),
               "supersedes": supersedes, "counts": counts}
        append_jsonl(ledger, rec)
        render_res = None if args.no_render else safe_render(report_root, ledger, attempt_id)
        return {"report_run_id": report_run_id, "reused": False, "snapshot_path": str(spath), "counts": counts,
                "unknown_reasons": reasons, "render": render_res, "elapsed_seconds": round(time.monotonic() - t0, 3),
                "backfill_requests": len(requests)}


def safe_render(report_root: Path, ledger: Path, attempt_id: str) -> dict:
    from render import render
    try:
        res = render(report_root)
        append_jsonl(ledger, {"attempt_id": attempt_id, "kind": "render", "status": "published", **res,
                              "attempted_at": iso(utcnow())})
        return res
    except Exception as exc:
        append_jsonl(ledger, {"attempt_id": attempt_id, "kind": "render", "status": "failed",
                              "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1200:],
                              "attempted_at": iso(utcnow()),
                              "note": "snapshot kept; previous page/current pointer left unchanged"})
        return {"status": "failed", "error": str(exc)}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        res = run(args)
    except AlreadyRunning as exc:
        print(json.dumps({"status": "skipped", "reason": str(exc)}, ensure_ascii=False))
        return 3
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
