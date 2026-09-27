"""Immutable run-local market snapshots and per-source as-of watermarks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time

import pandas as pd

from .hourly_v3_time import aware

UTC = timezone.utc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def visible_versions(frame: pd.DataFrame, cutoff: datetime, deadline: datetime) -> pd.DataFrame:
    """Filter complete, known versions before choosing a duplicate bar revision."""
    f = frame.copy()
    for col in ("start_at", "end_at", "available_at"):
        f[col] = pd.to_datetime(f[col], utc=True)
    mask = (f.end_at <= pd.Timestamp(cutoff)) & (f.available_at <= pd.Timestamp(deadline))
    if "received_at" in f:
        f["received_at"] = pd.to_datetime(f["received_at"], utc=True)
        mask &= f.received_at.isna() | (f.received_at <= pd.Timestamp(deadline))
        sort = ["start_at", "received_at"]
    else:
        sort = ["start_at", "available_at"]
    return f.loc[mask].sort_values(sort).drop_duplicates("start_at", keep="last")


def source_watermark(frame: pd.DataFrame, day: str, cutoff: datetime,
                     deadline: datetime) -> dict:
    visible = visible_versions(frame, cutoff, deadline)
    regular = visible[(visible.session_date == day) & (visible.session_type == "regular")]
    latest = regular.sort_values("end_at").iloc[-1] if len(regular) else None
    end = pd.Timestamp(latest.end_at).isoformat() if latest is not None else None
    received = None
    if latest is not None and "received_at" in regular and pd.notna(latest.received_at):
        received = pd.Timestamp(latest.received_at).isoformat()
    return {"latest_closed_end_at": end, "latest_received_at": received,
            "expected_end_at": cutoff.isoformat(), "bar_count": len(regular),
            "status": "complete_observed_receipt" if end == cutoff.isoformat() and received else
                      "complete_assumed_availability" if end == cutoff.isoformat() else "stale_or_missing"}


def _merge_versions(base: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return incoming
    # Retain both versions: inference chooses the last version actually
    # available at its information deadline, not a later corrected value.
    all_columns = list(dict.fromkeys([*base.columns, *incoming.columns]))
    return pd.concat([base.reindex(columns=all_columns), incoming.reindex(columns=all_columns)],
                     ignore_index=True)


def _fetch_new_version(symbol: str, day: str, year: int, target: Path,
                       audit: list[dict], r2_client, quote_ctx) -> pd.DataFrame:
    """Try R2 then paced OpenD. Writes only under the new run directory."""
    from scripts.fetch_research_data import normalize_kline_dataframe

    existing = pd.read_parquet(target) if target.exists() else pd.DataFrame()
    if r2_client is not None:
        key = f"us_5m/{symbol}/{year}.parquet"
        download = target.with_name(f"{year}.r2.parquet")
        try:
            if r2_client.head_object(key) is not None:
                r2_client.get_object(key, download)
                remote = pd.read_parquet(download)
                observed = datetime.now(UTC).isoformat()
                remote["received_at"] = observed
                existing = _merge_versions(existing, remote)
                audit.append({"symbol": symbol, "tier": "R2", "status": "downloaded",
                              "rows": len(remote), "received_at": observed})
        except Exception as exc:
            audit.append({"symbol": symbol, "tier": "R2", "status": "failed", "error": str(exc)})
        finally:
            download.unlink(missing_ok=True)
    if quote_ctx is None:
        return existing
    import futu as ft
    pages, page_key = [], None
    time.sleep(1.2)
    try:
        while True:
            ret, frame, page_key_next = quote_ctx.request_history_kline(
                code=f"US.{symbol}", start=day, end=day,
                ktype=ft.KLType.K_5M, autype=ft.AuType.NONE,
                fields=[ft.KL_FIELD.ALL], max_count=1000,
                page_req_key=page_key, extended_time=True)
            if ret != ft.RET_OK:
                raise RuntimeError(str(frame))
            if frame is not None and len(frame):
                pages.append(frame)
            if page_key_next is None:
                break
            page_key = page_key_next
            time.sleep(0.3)
        if pages:
            observed = datetime.now(UTC).isoformat()
            normalized = normalize_kline_dataframe(pd.concat(pages, ignore_index=True), symbol, "5m", "NONE")
            normalized["received_at"] = observed
            existing = _merge_versions(existing, normalized)
            audit.append({"symbol": symbol, "tier": "OpenD", "status": "received",
                          "rows": len(normalized), "received_at": observed})
        else:
            audit.append({"symbol": symbol, "tier": "OpenD", "status": "empty"})
    except Exception as exc:
        audit.append({"symbol": symbol, "tier": "OpenD", "status": "failed", "error": str(exc)})
    return existing


def make_snapshot(repo: Path, output: Path, config: dict, cutoff: datetime,
                  deadline: datetime, *, refresh: bool = False,
                  symbols: set[str] | None = None) -> dict:
    """Copy raw source versions into a new run, including future rows to test filtering."""
    calendar = json.loads((repo / config["calendar"]).read_text())
    universe = json.loads((repo / config["universe"]).read_text())
    day = cutoff.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date().isoformat()
    sessions = [s["session_date"] for s in calendar["sessions"]]
    ix = sessions.index(day)
    if ix < config["history_sessions"]:
        raise ValueError("calendar lacks ten previous regular sessions for H")
    first = sessions[ix - config["history_sessions"]]
    candidates = [m for m in universe["members"] if m["role"] == "candidate" and
                  (symbols is None or m["symbol"] in symbols)]
    required = sorted({m["symbol"] for m in candidates} |
                      {m["industry_proxy"] for m in candidates} | {"QQQ"})
    years = list(range(int(first[:4]), int(day[:4]) + 1))
    source_dir = output / "snapshot_sources"
    source_dir.mkdir(parents=True)
    audit: list[dict] = []
    hashes: dict[str, str] = {}
    watermarks: dict[str, dict] = {}
    r2, quote = None, None
    if refresh:
        try:
            from scripts.r2_client import R2Client
            r2 = R2Client()
        except Exception as exc:
            audit.append({"tier": "R2", "status": "unavailable", "error": str(exc)})
        try:
            import futu as ft
            ft.SysConfig.enable_proto_encrypt(False)
            quote = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
        except Exception as exc:
            audit.append({"tier": "OpenD", "status": "unavailable", "error": str(exc)})
    try:
        for symbol in required:
            combined = []
            for year in years:
                local = repo / config["source_dir"] / symbol / f"{year}.parquet"
                target = source_dir / symbol / f"{year}.parquet"
                target.parent.mkdir(parents=True, exist_ok=True)
                if local.exists():
                    f = pd.read_parquet(local)
                    f = f[(f.session_date >= first) & (f.session_date <= day)]
                    f.to_parquet(target, index=False)
                    hashes[str(local.relative_to(repo))] = sha256(local)
                    audit.append({"symbol": symbol, "tier": "Local", "status": "copied",
                                  "year": year, "rows": len(f)})
                else:
                    f = pd.DataFrame()
                    audit.append({"symbol": symbol, "tier": "Local", "status": "missing", "year": year})
                if year == int(day[:4]) and refresh:
                    f = _fetch_new_version(symbol, day, year, target, audit, r2, quote)
                    if not f.empty:
                        f = f[(f.session_date >= first) & (f.session_date <= day)]
                        f.to_parquet(target, index=False)
                if not f.empty:
                    combined.append(f)
            if combined:
                f = pd.concat(combined, ignore_index=True)
                watermarks[symbol] = source_watermark(f, day, cutoff, deadline)
                watermarks[symbol]["snapshot_files"] = [str((source_dir/symbol/f"{year}.parquet").relative_to(repo)) for year in years if (source_dir/symbol/f"{year}.parquet").exists()]
                watermarks[symbol]["snapshot_sha256"] = {p: sha256(repo/p) for p in watermarks[symbol]["snapshot_files"]}
            else:
                watermarks[symbol] = {"status": "source_missing", "expected_end_at": cutoff.isoformat()}
    finally:
        if quote is not None:
            quote.close()
    manifest = {"version": "hourly_snapshot_v3", "created_at": datetime.now(UTC).isoformat(),
                "snapshot_dir": str(source_dir),
                "session_date": day, "source_start_date": first,
                "cutoff_at": cutoff.isoformat(), "information_deadline": deadline.isoformat(),
                "universe_sha256": sha256(repo/config["universe"]),
                "calendar_sha256": sha256(repo/config["calendar"]),
                "source_hashes": hashes, "watermarks": watermarks, "acquisition_audit": audit,
                "historical_availability": config["historical_source_quality"],
                "refresh_attempted": refresh}
    (output/"snapshot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest
