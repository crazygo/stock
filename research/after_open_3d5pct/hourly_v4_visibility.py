"""Knowledge-time market version selection for v4 inference.

Raw full files remain untouched. This module writes a derived visible view
inside each run, allowing the frozen v3 feature schema to consume an honest
as-of prefix without asking the caller to delete future/late source rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .terminal_risk_v4 import REPO, sha256


def visible_versions_v4(frame: pd.DataFrame, cutoff: pd.Timestamp,
                        knowledge_at: pd.Timestamp) -> pd.DataFrame:
    f = frame.copy()
    for col in ("start_at", "end_at", "available_at"):
        f[col] = pd.to_datetime(f[col], utc=True)
    if "received_at" not in f:
        f["received_at"] = pd.NaT
    f["received_at"] = pd.to_datetime(f.received_at, utc=True)
    eligible = ((f.end_at <= cutoff) & (f.available_at <= knowledge_at) &
                (f.received_at.isna() | (f.received_at <= knowledge_at)))
    f = f.loc[eligible].copy()
    if f.empty:
        return f.drop(columns=["_observed_receipt", "_version_at"], errors="ignore")
    f["_observed_receipt"] = f.received_at.notna().astype(int)
    f["_version_at"] = f.received_at.fillna(f.available_at)
    repeated = f.duplicated(["start_at", "_observed_receipt", "_version_at"], keep=False)
    if repeated.any():
        for _, versions in f.loc[repeated].groupby(["start_at", "_observed_receipt", "_version_at"], dropna=False):
            if len(versions[["end_at", "open", "high", "low", "close", "volume", "price_basis"]].drop_duplicates()) > 1:
                raise ValueError("conflicting same-time market source versions")
    f = f.sort_values(["start_at", "_observed_receipt", "_version_at", "available_at"],
                      kind="stable").drop_duplicates("start_at", keep="last")
    return f.drop(columns=["_observed_receipt", "_version_at"])


def source_watermark_v4(frame: pd.DataFrame, day: str, cutoff: pd.Timestamp,
                        knowledge_at: pd.Timestamp) -> dict:
    visible = visible_versions_v4(frame, cutoff, knowledge_at)
    today = visible[visible.session_date == day].sort_values("end_at")
    regular = today[today.session_type == "regular"]
    latest_any = today.iloc[-1] if len(today) else None
    latest_regular = regular.iloc[-1] if len(regular) else None
    end = pd.Timestamp(latest_regular.end_at).isoformat() if latest_regular is not None else None
    received = (pd.Timestamp(latest_regular.received_at).isoformat()
                if latest_regular is not None and pd.notna(latest_regular.received_at) else None)
    if end == cutoff.isoformat():
        status = "complete_observed_receipt" if received else "complete_assumed_availability"
    else:
        status = "stale_or_missing"
    return {"status": status, "expected_end_at": cutoff.isoformat(),
            "latest_closed_end_at": end, "latest_received_at": received,
            "regular_bar_count": len(regular),
            "latest_visible_source_end_at": pd.Timestamp(latest_any.end_at).isoformat() if latest_any is not None else None,
            "latest_visible_source_session_type": latest_any.session_type if latest_any is not None else None,
            "latest_visible_source_received_at": pd.Timestamp(latest_any.received_at).isoformat()
            if latest_any is not None and pd.notna(latest_any.received_at) else None,
            "version_priority": "observed_receipt_before_assumed_null_after_asof_filter"}


def materialize_visible_view(snapshot: dict, output: Path, cutoff: pd.Timestamp,
                             knowledge_at: pd.Timestamp) -> dict:
    raw_root = Path(snapshot["snapshot_dir"])
    visible_root = output / "visible_sources"
    visible_root.mkdir()
    file_hashes = {}
    for symbol, info in snapshot["watermarks"].items():
        sources = info.get("snapshot_files", [])
        if not sources:
            continue
        by_year = {}
        for rel in sources:
            source = REPO / rel
            if sha256(source) != info["snapshot_sha256"][rel]:
                raise ValueError(f"raw snapshot source hash mismatch: {rel}")
            by_year.setdefault(source.name, []).append(pd.read_parquet(source))
        for filename, frames in by_year.items():
            visible = visible_versions_v4(pd.concat(frames, ignore_index=True), cutoff, knowledge_at)
            target = visible_root / symbol / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            visible.to_parquet(target, index=False)
            file_hashes[str(target.relative_to(REPO))] = sha256(target)
    return {"version": "hourly_v4_visible_view",
            "raw_snapshot_dir": str(raw_root), "view_dir": str(visible_root),
            "cutoff_at": cutoff.isoformat(), "knowledge_at": knowledge_at.isoformat(),
            "visible_files_sha256": file_hashes,
            "raw_files_preserved": True}
