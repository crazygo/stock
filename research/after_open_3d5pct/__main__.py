"""Run from repository root: python3 -m research.after_open_3d5pct ..."""

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .smoke import run_smoke, serialized


def main():
    parser = argparse.ArgumentParser(description="After-open research scaffold (offline, synthetic only)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-config")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "validate-config":
            print(serialized({"status": "valid", "config": config.to_dict()}))
        else:
            print(serialized(run_smoke(args.output, config)))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"contract error: {exc}\n")


if __name__ == "__main__":
    main()
