#!/usr/bin/env python3
"""Render the five-traits map + per-security timeline from *saved* snapshots only.

Pure rendering: reads the report ledger/snapshots, writes pages/<render_id>/index.html, then
atomically repoints `current` (symlink) and current.json. Re-rendering never adds trait points.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import traits as T  # noqa: E402
from common import (DEFAULT_REPORT_ROOT, PKG, iso, iso_sh, make_readonly, new_id, publish_dir, read_json,  # noqa: E402
                    read_jsonl, swap_symlink, utcnow, write_json_atomic, write_json_new)


def collect(report_root: Path) -> tuple[dict, list[dict], list[dict]]:
    recs = [r for r in read_jsonl(report_root / "ledger.jsonl") if r.get("kind") == "refresh" and r.get("snapshot_published")]
    if not recs:
        raise RuntimeError("no published snapshot to render")
    snaps = []
    for r in recs:
        s = read_json(report_root / r["snapshot_path"])
        snaps.append(s)
    # latest version per (mode, cutoff); older versions kept as revision history
    by_key: dict = {}
    for s in sorted(snaps, key=lambda s: s["computed_at"]):
        by_key.setdefault((s["observation_mode"], s["evaluation_cutoff"]), []).append(s)
    current_versions = [v[-1] for v in by_key.values()]
    live = [s for s in current_versions if s["observation_mode"] == "live"]
    latest = max(live or current_versions, key=lambda s: (s["evaluation_cutoff"], s["computed_at"]))
    return latest, sorted(current_versions, key=lambda s: s["evaluation_cutoff"]), [s for v in by_key.values() for s in v[:-1]]


def point_of(s: dict, x: dict) -> dict:
    tr = {}
    for t in T.TRAITS:
        v = x["traits"][t]
        u = v.get("uncertainty") or {}
        tr[t] = {"value": v.get("value"), "display": v.get("display_0_100"), "category": v.get("category"),
                 "status": v.get("status"), "reason": v.get("reason"), "unit": v.get("unit"),
                 "p10": u.get("p10"), "p90": u.get("p90"), "freq": u.get("category_frequency"),
                 "valid_replicates": u.get("valid_replicates")}
    price = x.get("price") or {}
    return {"date": s["evaluation_cutoff"], "mode": s["observation_mode"], "report_run_id": s["report_run_id"],
            "data_run_id": s["data"]["data_run_id"], "computed_at": s["computed_at_shanghai"],
            "price_asof": price.get("price_asof"), "stale_sessions": price.get("stale_sessions") or [],
            "status": x["status"], "unknown_reason": x.get("unknown_reason"), "traits": tr,
            "contiguous_closes": x.get("contiguous_closes")}


def build_data(report_root: Path, render_id: str) -> dict:
    latest, versions, superseded = collect(report_root)
    secs = []
    for x in latest["results"]:
        sid = x["security_id"]
        pts = []
        for s in versions:
            y = next((r for r in s["results"] if r["security_id"] == sid), None)
            if y is not None:
                pts.append(point_of(s, y))
        revs = [{"report_run_id": s["report_run_id"], "date": s["evaluation_cutoff"], "computed_at": s["computed_at_shanghai"]}
                for s in superseded if any(r["security_id"] == sid for r in s["results"])]
        secs.append({"id": sid, "ticker": x["ticker"], "name": x.get("name"), "class": x["class"], "status": x["status"],
                     "unknown_reason": x.get("unknown_reason"), "detail": x.get("detail"), "map": x.get("map"),
                     "price": x.get("price"), "news": x["news"]["status"], "valuation": x["valuation"]["status"],
                     "tags": x.get("tags"), "data_source": x.get("data_source"),
                     "points": pts, "superseded_versions": revs})
    return {"render_id": render_id, "rendered_at": iso_sh(utcnow()),
            "latest": {k: latest[k] for k in ("report_run_id", "evaluation_cutoff", "observation_mode", "computed_at_shanghai",
                                              "counts", "unknown_reasons", "source_cutoffs", "universe", "data")},
            "price_basis": latest.get("price_basis") or {"label": (latest.get("intervals") or {}).get("price_basis")},
            "universe_label": universe_label(latest),
            "snapshot_count": len(versions), "superseded_count": len(superseded),
            "units": T.UNITS, "categories": T.CATEGORIES, "securities": secs}


def universe_label(snap: dict) -> str:
    src = (snap.get("universe") or {}).get("source") or {}
    if src.get("label"):
        return str(src["label"])
    if src.get("group"):
        return str(src["group"])
    return {"explicit_list": "自定义列表"}.get(src.get("type"), "股票池")


TEMPLATE = (PKG / "page_template.html")


def render(report_root: Path = DEFAULT_REPORT_ROOT) -> dict:
    report_root = Path(report_root)
    render_id = new_id("render")
    data = build_data(report_root, render_id)
    page = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = page.replace("/*VIEW_LOGIC*/", (PKG / "view_logic.js").read_text(encoding="utf-8"))
    page = page.replace("__DATA_JSON__", payload)
    page = page.replace("__TITLE__", html.escape(f"{data['universe_label']} · 五标签 · 评估截止 {data['latest']['evaluation_cutoff']}"))
    tmp = report_root / "pages" / f".tmp-{render_id}"
    tmp.mkdir(parents=True, exist_ok=False)
    (tmp / "index.html").write_text(page, encoding="utf-8")
    # validate before publishing
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
    back = json.loads(m.group(1).replace("<\\/", "</"))
    assert len(back["securities"]) == len(data["securities"])
    n_points = sum(len(s["points"]) for s in back["securities"])
    write_json_new(tmp / "render.json", {"render_id": render_id, "points": n_points, "snapshots": data["snapshot_count"],
                                         "latest_report_run_id": data["latest"]["report_run_id"]})
    final = report_root / "pages" / render_id
    publish_dir(tmp, final)
    make_readonly(final)
    swap_symlink(report_root / "current", final)
    write_json_atomic(report_root / "current.json", {"render_id": render_id, "page": f"pages/{render_id}/index.html",
                                                     "latest_report_run_id": data["latest"]["report_run_id"],
                                                     "evaluation_cutoff": data["latest"]["evaluation_cutoff"],
                                                     "published_at": iso(utcnow())})
    return {"render_id": render_id, "page": str(final / "index.html"), "points": n_points,
            "snapshots": data["snapshot_count"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    print(json.dumps(render(Path(ap.parse_args().report_root)), ensure_ascii=False, indent=2))
