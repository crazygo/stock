#!/usr/bin/env python3
"""Import published historical descriptions into immutable per-security records.

No market-data requests, price-bar copies, trading signals or personal decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parents[2]
SCHEMA = "stock_decision_memory_seed_v1"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def repo_path(path):
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_verified(run_dir, relative, manifest):
    path = run_dir / relative
    raw = path.read_bytes()
    expected = manifest.get("files", {}).get(relative, {}).get("sha256")
    if not expected or sha(raw) != expected:
        raise ValueError(f"Published input hash mismatch: {relative}")
    return json.loads(raw), {"path": repo_path(path), "sha256": expected}


def project_record(record, member):
    code = record["code"]
    # Source contains raw OHLC rows. Only derived observations enter the dossier.
    folds = record.get("folds") or {}
    shares = record.get("shares") or {}
    return {
        "code": code,
        "name": member.get("name", code),
        "kind": member.get("kind", "unknown"),
        "membership_groups": member.get("groups", []),
        "identity_status": "member_snapshot_not_product_terms_verification",
        "evidence_status": "historical_description",
        "requested_range": [record["start"], record["end"]],
        "observed_range": [record.get("observedFirst"), record.get("observedLast")],
        "settings": record["settings"],
        "coverage": {
            "status": record.get("status"),
            "observed_daily_bars": len(record.get("rows", [])),
            "calendar_known": record.get("calendarKnown"),
            "missing": record.get("missing", []),
        },
        "upward_windows": {
            "fraction": shares.get("up"),
            "qualifying": shares.get("upN"),
            "complete_windows": shares.get("n"),
            "interpretation": "overlapping_historical_windows_not_future_probability",
        },
        "reversals": {
            "frequency": folds.get("frequency"),
            "frequency_unit": "confirmations_per_20_trading_day_changes",
            "confirmations": folds.get("turnCount"),
            "trading_day_changes": folds.get("returns"),
            "amplitude_median": folds.get("amplitudeMedian"),
            "amplitude_unit": "high_close_divided_by_low_close_minus_one",
            "complete_segments": folds.get("amplitudeN"),
            "turns": folds.get("turns", []),
            "completed": folds.get("completed", []),
            "excluded": folds.get("excluded", []),
            "unfinished": folds.get("unfinished"),
            "interpretation": "retrospective_endpoints_with_separate_confirmation_dates",
        },
        "conditional_exit_statistics": {"status": "not_computed"},
        "validated_trading_rules": [],
        "personal_decisions_status": "not_connected",
    }


def verify_snapshot(path):
    manifest = json.loads((path / "manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        if sha((path / relative).read_bytes()) != digest:
            raise ValueError(f"Existing snapshot modified: {path.name}/{relative}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path,
                        default=REPO / ".cache/stock_review_pair_v1/current")
    parser.add_argument("--store", type=Path,
                        default=REPO / ".cache/stock_decision_memory_v1")
    args = parser.parse_args()
    run_dir = args.from_run.resolve(strict=True)  # Pin current once.
    source_manifest_path = run_dir / "manifest.json"
    source_manifest_raw = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_raw)
    run, run_ref = read_verified(run_dir, "run.json", source_manifest)
    behavior, behavior_ref = read_verified(run_dir, "results/behavior.json", source_manifest)
    members, members_ref = read_verified(run_dir, "inputs/members.json", source_manifest)
    if run.get("run_id") != source_manifest.get("run_id") or behavior.get("run_id") != run["run_id"]:
        raise ValueError("Source batch IDs disagree")
    member_map = {item["code"]: item for item in members["records"]}
    source_records = behavior["records"]
    if not source_records or len({r["code"] for r in source_records}) != len(source_records):
        raise ValueError("Empty or duplicate source securities")
    records = [project_record(r, member_map[r["code"]]) for r in source_records]
    records.sort(key=lambda item: item["code"])
    # Ignore render timestamps and batch IDs: identical descriptions are one memory.
    semantic_hash = sha(encoded({"schema_version": SCHEMA, "records": records}))
    snapshot_id = "seed-" + semantic_hash[:24]
    store = args.store.resolve()
    snapshots = store / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    destination = snapshots / snapshot_id
    reused = destination.exists()
    if reused:
        existing = verify_snapshot(destination)
        if existing["semantic_sha256"] != semantic_hash:
            raise ValueError("Snapshot identifier conflict; preserve both inputs for review")
    else:
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=snapshots))
        try:
            (staging / "profiles").mkdir()
            output_hashes = {}
            for item in records:
                filename = item["code"] + ".json"
                if Path(filename).name != filename or "\\" in filename:
                    raise ValueError("Invalid security identifier")
                profile = {"schema_version": SCHEMA, "snapshot_id": snapshot_id,
                           "observation": item}
                raw = encoded(profile)
                relative = "profiles/" + filename
                (staging / relative).write_bytes(raw)
                output_hashes[relative] = sha(raw)
            manifest = {
                "schema_version": SCHEMA,
                "snapshot_id": snapshot_id,
                "semantic_sha256": semantic_hash,
                "imported_at": datetime.now(timezone.utc).isoformat(),
                "source_run_id": run["run_id"],
                "source_data_run_id": run.get("data_run_id"),
                "source_information_cutoff": run.get("information_cutoff"),
                "source_market_cutoffs": run.get("market_cutoffs"),
                "source_status": run.get("status"),
                "sources": {
                    "manifest": {"path": repo_path(source_manifest_path),
                                 "sha256": sha(source_manifest_raw)},
                    "run": run_ref, "behavior": behavior_ref, "members": members_ref,
                },
                "member_snapshot_not_PIT": members.get("membership_not_PIT"),
                "securities": len(records),
                "data_kind": "derived_descriptions_no_raw_bars",
                "files": output_hashes,
                "scope": {"historical_descriptions": True,
                          "conditional_exit_results": False,
                          "validated_trading_rules": False,
                          "live_quotes": False, "personal_decisions": False},
            }
            (staging / "manifest.json").write_bytes(encoded(manifest))
            os.rename(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    catalog = {}
    for snapshot in sorted(snapshots.iterdir()):
        if snapshot.name.startswith(".") or not snapshot.is_dir():
            continue
        meta = verify_snapshot(snapshot)
        for relative in sorted(meta["files"]):
            code = Path(relative).stem
            catalog.setdefault(code, []).append({
                "snapshot_id": meta["snapshot_id"],
                "profile": str(Path("snapshots") / snapshot.name / relative),
                "source_information_cutoff": meta["source_information_cutoff"],
                "imported_at": meta["imported_at"],
            })
    index_raw = encoded({"schema_version": SCHEMA, "securities": catalog})
    index_path = store / "index.json"
    if not index_path.exists() or index_path.read_bytes() != index_raw:
        fd, pending = tempfile.mkstemp(prefix=".index-", dir=store)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(index_raw)
            os.replace(pending, index_path)
        finally:
            if os.path.exists(pending):
                os.unlink(pending)
    print(json.dumps({"snapshot_id": snapshot_id, "reused": reused,
                      "securities": len(records), "source_run_id": run["run_id"],
                      "index": str(index_path), "snapshot": str(destination)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
