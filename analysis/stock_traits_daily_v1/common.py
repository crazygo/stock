"""Shared helpers for stock_traits_daily_v1 (stock-data-backfill / stock-report-refresh).

Isolated from older experiments: this package only reads legacy inputs and writes to
its own roots (see README.md). Nothing here uploads to R2, pushes Git, trades or edits
watchlists.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import secrets
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
SH = ZoneInfo("Asia/Shanghai")
UTC = timezone.utc

DEFAULT_CALENDAR = REPO / "market_data" / "calendars" / "nasdaq_sessions_2026_v1.json"
DEFAULT_LOCAL_5M = REPO / "market_data" / "us_5m"
DEFAULT_CAPTURE_ROOT = REPO / "market_data" / "stock_data_v1" / "captures"
DEFAULT_DATA_ROOT = REPO / ".cache" / "stock_data_v1"
DEFAULT_REPORT_ROOT = REPO / ".cache" / "stock_report_v1"
DEFAULT_UNIVERSE_FILE = REPO / "analysis" / "preopen_5d5pct_v6" / "R06" / "favorites_snapshot.json"
DEFAULT_SEC_FACTS = REPO / "data" / "ai_sec_annual_facts.json"

MANIFEST_SCHEMA = "stock_data_manifest_v1"
SNAPSHOT_SCHEMA = "stock_report_snapshot_v1"


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(ts: datetime | None) -> str | None:
    return None if ts is None else ts.astimezone(UTC).isoformat(timespec="seconds")


def iso_sh(ts: datetime | None) -> str | None:
    return None if ts is None else ts.astimezone(SH).isoformat(timespec="seconds")


def parse_ts(value: str) -> datetime:
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {value}")
    return ts


def new_id(prefix: str, ts: datetime | None = None) -> str:
    ts = ts or utcnow()
    return f"{prefix}-{ts.astimezone(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path, root: Path = REPO) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON through a same-directory temp file, fsync, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_json_new(path: Path, obj: Any) -> None:
    """Write a JSON file that must not exist yet (immutable artefacts)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def append_jsonl(path: Path, record: dict) -> None:
    """Append-only ledger write with an exclusive lock and fsync."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class AlreadyRunning(RuntimeError):
    pass


def assert_own_data_root(data_root: Path) -> None:
    """Refuse to touch a data root whose current.json belongs to another pipeline (checked before any write)."""
    cur = Path(data_root) / "current.json"
    if not cur.exists():
        return
    try:
        ref = read_json(cur)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{cur} is unreadable ({exc}); refusing to use this data root") from exc
    if not isinstance(ref, dict) or "manifest_path" not in ref or not str(ref.get("data_run_id", "")).startswith("data-"):
        raise SystemExit(f"{cur} was written by a different pipeline (keys {sorted(ref)[:6]}); "
                         "pass --data-root (and --capture-root / --report-root) pointing at a stock_traits_daily_v1 root")


@contextlib.contextmanager
def run_lock(path: Path) -> Iterator[None]:
    """Non-blocking re-entrancy guard (same data/report task cannot overlap)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                raise AlreadyRunning(f"another run holds {path}") from exc
            raise
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} started={iso(utcnow())}\n")
        fh.flush()
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def publish_dir(tmp_dir: Path, final_dir: Path) -> None:
    """Atomically move a fully written temp directory into place; never overwrite."""
    final_dir = Path(final_dir)
    if final_dir.exists():
        raise FileExistsError(f"immutable target already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(tmp_dir, final_dir)


def make_readonly(root: Path) -> None:
    root = Path(root)
    paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    for p in paths:
        os.chmod(p, 0o444)


def swap_symlink(link: Path, target: Path) -> None:
    """Atomically repoint `link` to `target` (relative link, rename(2) over the old one)."""
    link = Path(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    rel_target = os.path.relpath(Path(target).resolve(), link.parent.resolve())
    tmp = link.parent / f".{link.name}.{secrets.token_hex(4)}.tmp"
    os.symlink(rel_target, tmp)
    os.replace(tmp, link)


# ---------------------------------------------------------------- calendar
class Calendar:
    def __init__(self, path: Path = DEFAULT_CALENDAR):
        self.path = Path(path)
        data = read_json(self.path)
        self.calendar_id = data.get("calendar_id", self.path.stem)
        self.start_date = data.get("start_date")
        self.end_date = data.get("end_date")
        self.sha256 = sha256_file(self.path)
        self.sessions = sorted(data["sessions"], key=lambda s: s["session_date"])
        self.by_date = {s["session_date"]: s for s in self.sessions}
        self.dates = [s["session_date"] for s in self.sessions]

    def is_session(self, d: str) -> bool:
        return d in self.by_date

    def close_at(self, d: str) -> datetime:
        return parse_ts(self.by_date[d]["close_at"])

    def open_at(self, d: str) -> datetime:
        return parse_ts(self.by_date[d]["open_at"])

    def close_time_et(self, d: str) -> str:
        return parse_ts(self.by_date[d]["close_at_et"]).strftime("%H:%M:%S")

    def expected_regular_5m_bars(self, d: str) -> int:
        return int(self.by_date[d]["duration_minutes"]) // 5

    def last_closed_session(self, now: datetime) -> str:
        """Latest session whose official close is at or before `now`."""
        done = [d for d in self.dates if self.close_at(d) <= now]
        if not done:
            raise ValueError("calendar has no closed session before now")
        last = done[-1]
        if last == self.dates[-1] and now.date().isoformat() > self.end_date:
            raise ValueError(f"calendar {self.calendar_id} ends {self.end_date}; extend it before using")
        return last

    def sessions_through(self, end: str, count: int) -> list[str]:
        if end not in self.by_date:
            raise ValueError(f"{end} is not a session in {self.calendar_id}")
        idx = self.dates.index(end)
        start = max(0, idx - count + 1)
        out = self.dates[start: idx + 1]
        return out

    def sessions_between(self, start: str, end: str) -> list[str]:
        return [d for d in self.dates if start <= d <= end]


def git_commit_of(path: Path) -> str | None:
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%H %cI", "--", rel(path)],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None
