#!/usr/bin/env python3
"""Manual synchronization tool between local market data (Parquet) and Cloudflare R2.

Commands:
    diff    - Show differences between local and R2
    push    - Manually upload local Parquet files to R2 (multi-threaded)
    pull    - Manually download remote Parquet files from R2 (multi-threaded)
    status  - Show bucket summary and storage statistics

Examples:
    # 1. Check diff for 60m Parquet data
    python3 scripts/r2_sync.py diff

    # 2. Push all local Parquet files to R2
    python3 scripts/r2_sync.py push --workers 16

    # 3. Push only specific tickers with dry-run check
    python3 scripts/r2_sync.py push --symbols AAPL NVDA --dry-run
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.r2_client import R2Client


def format_bytes(bytes_count: int) -> str:
    """Format bytes into human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(bytes_count)
    for u in units:
        if size < 1024.0:
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{size:.2f} PB"


def get_allowed_extensions(args: argparse.Namespace) -> Tuple[str, ...]:
    if not args.ext:
        return (".parquet",)
    exts = [e.strip() if e.strip().startswith(".") else f".{e.strip()}" for e in args.ext.split(",")]
    return tuple(exts)


def cmd_diff(args: argparse.Namespace, client: R2Client) -> int:
    local_dir = Path(args.dir)
    prefix = args.prefix.rstrip("/")
    exts = get_allowed_extensions(args)
    print(f"Diffing local [{local_dir}] vs R2 [{client.bucket}/{prefix}] for {exts}...")
    start_t = time.time()
    res = client.diff(local_dir, remote_prefix=prefix, allowed_extensions=exts)
    elapsed = time.time() - start_t

    only_loc = res["only_local"]
    only_rem = res["only_remote"]
    diff = res["different"]
    ident = res["identical"]

    print(f"\n--- Diff Summary (Completed in {elapsed:.2f}s) ---")
    print(f"  Identical (in sync): {len(ident)}")
    print(f"  Only on local      : {len(only_loc)}")
    print(f"  Only on R2         : {len(only_rem)}")
    print(f"  Size differs       : {len(diff)}")

    if only_loc:
        print(f"\n[Only on Local] ({len(only_loc)} files ready to push):")
        for item in only_loc[:15]:
            print(f"  + {item['key']} ({format_bytes(item['size'])})")
        if len(only_loc) > 15:
            print(f"  ... and {len(only_loc) - 15} more files.")

    if only_rem:
        print(f"\n[Only on R2] ({len(only_rem)} files ready to pull):")
        for item in only_rem[:15]:
            print(f"  - {item['key']} ({format_bytes(item['size'])})")
        if len(only_rem) > 15:
            print(f"  ... and {len(only_rem) - 15} more files.")

    if diff:
        print(f"\n[Different Size] ({len(diff)} files differ):")
        for item in diff[:10]:
            print(f"  * {item['key']}: local={format_bytes(item['local_size'])}, r2={format_bytes(item['remote_size'])}")

    return 0


def cmd_push(args: argparse.Namespace, client: R2Client) -> int:
    local_dir = Path(args.dir)
    prefix = args.prefix.rstrip("/")
    symbols = {s.upper().strip() for s in (args.symbols or [])}
    exts = get_allowed_extensions(args)

    print(f"Scanning local directory [{local_dir}] for {exts} to push to R2 [{client.bucket}/{prefix}]...")
    diff_res = client.diff(local_dir, remote_prefix=prefix, allowed_extensions=exts)
    candidates = diff_res["only_local"] + diff_res["different"]

    if args.force:
        candidates = [
            {"key": f"{prefix}/{p.relative_to(local_dir).as_posix()}", "local_path": str(p), "size": p.stat().st_size}
            for p in local_dir.rglob("*") if p.is_file() and any(p.name.endswith(e) for e in exts)
        ]

    # Filter symbols if specified
    targets = []
    for item in candidates:
        key = item["key"]
        if symbols:
            parts = key.split("/")
            if not any(part.upper() in symbols for part in parts):
                continue
        targets.append(item)

    if not targets:
        print("No files need to be pushed (everything in sync).")
        return 0

    total_bytes = sum(t["size"] for t in targets)
    print(f"Found {len(targets)} files to upload ({format_bytes(total_bytes)}) with {args.workers} workers.")

    if args.dry_run:
        print("[Dry-run] Would upload the following files:")
        for t in targets[:20]:
            print(f"  -> {t['key']} ({format_bytes(t['size'])})")
        if len(targets) > 20:
            print(f"  ... and {len(targets) - 20} more files.")
        return 0

    print("Starting multi-threaded upload to Cloudflare R2...")
    success = 0
    failed = 0
    start_t = time.time()

    def _upload_single(target: Dict[str, Any]) -> Tuple[bool, str, int, str]:
        loc_path = Path(target["local_path"])
        key = target["key"]
        try:
            client.put_object(loc_path, key)
            return True, key, target["size"], ""
        except Exception as e:
            return False, key, target["size"], str(e)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_upload_single, t): t for t in targets}
        for future in as_completed(futures):
            ok, key, size, err = future.result()
            if ok:
                success += 1
                if success % 20 == 0 or success == len(targets):
                    print(f"  [{success}/{len(targets)}] Uploaded: {key} ({format_bytes(size)})")
            else:
                failed += 1
                print(f"  [FAILED] {key} - {err}", file=sys.stderr)

    elapsed = time.time() - start_t
    speed = total_bytes / elapsed if elapsed > 0 else 0
    print(f"\nPush completed: {success}/{len(targets)} files uploaded in {elapsed:.2f}s ({format_bytes(int(speed))}/s).")
    return 0 if failed == 0 else 1


def cmd_pull(args: argparse.Namespace, client: R2Client) -> int:
    local_dir = Path(args.dir)
    prefix = args.prefix.rstrip("/")
    symbols = {s.upper().strip() for s in (args.symbols or [])}
    exts = get_allowed_extensions(args)

    print(f"Checking R2 objects under [{client.bucket}/{prefix}] for {exts}...")
    remote_objs = [o for o in client.list_objects(prefix=prefix) if any(o["key"].endswith(e) for e in exts)]
    diff_res = client.diff(local_dir, remote_prefix=prefix, allowed_extensions=exts)
    candidates = diff_res["only_remote"] + diff_res["different"]

    if args.force:
        candidates = remote_objs

    targets = []
    for item in candidates:
        key = item["key"]
        if symbols:
            parts = key.split("/")
            if not any(part.upper() in symbols for part in parts):
                continue
        targets.append(item)

    if not targets:
        print("No files need to be pulled (local is up to date).")
        return 0

    total_bytes = sum(t["size"] for t in targets)
    print(f"Found {len(targets)} files to download ({format_bytes(total_bytes)}) with {args.workers} workers.")

    if args.dry_run:
        print("[Dry-run] Would download the following files:")
        for t in targets[:20]:
            print(f"  <- {t['key']} ({format_bytes(t['size'])})")
        if len(targets) > 20:
            print(f"  ... and {len(targets) - 20} more files.")
        return 0

    print("Starting multi-threaded download from Cloudflare R2...")
    success = 0
    failed = 0
    start_t = time.time()

    def _download_single(target: Dict[str, Any]) -> Tuple[bool, str, int, str]:
        key = target["key"]
        rel_key = key.removeprefix(prefix).lstrip("/")
        dest_path = local_dir / rel_key
        try:
            bytes_dl = client.get_object(key, dest_path)
            return True, key, bytes_dl, ""
        except Exception as e:
            return False, key, 0, str(e)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_download_single, t): t for t in targets}
        for future in as_completed(futures):
            ok, key, size, err = future.result()
            if ok:
                success += 1
                if success % 20 == 0 or success == len(targets):
                    print(f"  [{success}/{len(targets)}] Downloaded: {key} ({format_bytes(size)})")
            else:
                failed += 1
                print(f"  [FAILED] {key} - {err}", file=sys.stderr)

    elapsed = time.time() - start_t
    speed = total_bytes / elapsed if elapsed > 0 else 0
    print(f"\nPull completed: {success}/{len(targets)} files downloaded in {elapsed:.2f}s ({format_bytes(int(speed))}/s).")
    return 0 if failed == 0 else 1


def cmd_status(args: argparse.Namespace, client: R2Client) -> int:
    prefix = args.prefix.rstrip("/")
    print(f"Fetching R2 bucket status for [{client.bucket}/{prefix or '(root)'}]...")
    objs = client.list_objects(prefix=prefix)
    total_size = sum(o["size"] for o in objs)
    parquet_objs = [o for o in objs if o["key"].endswith(".parquet")]
    parquet_size = sum(o["size"] for o in parquet_objs)

    print("\n--- Cloudflare R2 Bucket Status ---")
    print(f"  Bucket Name        : {client.bucket}")
    print(f"  Endpoint           : {client.endpoint}")
    print(f"  Prefix             : {prefix or '(root)'}")
    print(f"  Total Remote Files : {len(objs)} (Parquet: {len(parquet_objs)})")
    print(f"  Total Remote Size  : {format_bytes(total_size)} (Parquet: {format_bytes(parquet_size)})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p):
        p.add_argument("--dir", default=str(ROOT / "market_data" / "us_60m"), help="Local directory (default: market_data/us_60m)")
        p.add_argument("--prefix", default="market_data/us_60m", help="Remote R2 prefix (default: market_data/us_60m)")
        p.add_argument("--ext", default=".parquet", help="File extensions to sync (default: .parquet)")
        p.add_argument("--workers", type=int, default=16, help="Number of concurrent worker threads (default: 16)")

    # diff
    p_diff = subparsers.add_parser("diff", help="Diff local files against R2")
    add_common_args(p_diff)

    # push
    p_push = subparsers.add_parser("push", help="Manually push local Parquet files to R2")
    add_common_args(p_push)
    p_push.add_argument("--symbols", nargs="*", help="Filter specific ticker symbols (e.g. AAPL NVDA)")
    p_push.add_argument("--dry-run", action="store_true", help="Preview files to upload without modifying R2")
    p_push.add_argument("--force", action="store_true", help="Force upload all files even if identical")

    # pull
    p_pull = subparsers.add_parser("pull", help="Manually pull Parquet files from R2 to local")
    add_common_args(p_pull)
    p_pull.add_argument("--symbols", nargs="*", help="Filter specific ticker symbols")
    p_pull.add_argument("--dry-run", action="store_true", help="Preview files to download")
    p_pull.add_argument("--force", action="store_true", help="Force download all files")

    # status
    p_status = subparsers.add_parser("status", help="Show R2 storage summary")
    p_status.add_argument("--prefix", default="", help="Remote R2 prefix to check")

    args = parser.parse_args()
    client = R2Client()

    if args.command == "diff":
        return cmd_diff(args, client)
    elif args.command == "push":
        return cmd_push(args, client)
    elif args.command == "pull":
        return cmd_pull(args, client)
    elif args.command == "status":
        return cmd_status(args, client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
