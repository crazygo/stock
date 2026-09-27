"""Freeze local inputs in an isolated namespace, then reuse the audited v6 builder."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct import v6_data as vd

ROOT = vd.ROOT
HERE = Path(__file__).resolve().parent
OPTICS = ROOT / "research/strategy_group_lab/runs/optics_3round_20260926/market_data"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def new_memberships(inputs, config, added):
    """Same preregistered market classifiers; only newly added symbols are appended."""
    group = inputs / config["group_run"]
    versions = json.loads((group / "versions.json").read_text())
    members = json.loads((group / "memberships.json").read_text())
    sessions = vd._calendar(config)
    dates = [s["session_date"] for s in sessions]
    for symbol in added:
        view = vd.load_symbol(symbol, sessions, config)
        actions = vd._split_days(symbol)
        for version in versions:
            strategy = version["strategy_id"]
            if strategy not in vd.STRATEGIES or version["membership_basis"] != "causal_weekly_reconstruction":
                continue
            cutoff = pd.Timestamp(version["feature_cutoff_at"])
            ids = [i for i, s in enumerate(sessions) if pd.Timestamp(s["close_at"]) < cutoff]
            n = int(strategy.split("_")[1]) if strategy.startswith("trend_") else 20
            need = n if strategy == "liquidity" else n + 1
            ids = ids[-need:]
            if len(ids) != need or not view.daily_full[ids].all():
                continue
            if any(dates[ids[0]] < a <= dates[ids[-1]] for a in actions):
                continue
            observed = view.effective_available[ids, 66:144].view("int64").max()
            if observed >= cutoff.value:
                continue
            daily = view.seqday[ids]
            closes = np.exp(daily[:, 12].astype(float) * 5 + daily[:, 0])
            returns = np.diff(np.log(closes))
            if strategy == "liquidity":
                dollars = [float(view.raw5[i, 66:66+int(sessions[i]["duration_minutes"])//5, 5].sum()) for i in ids]
                value = float(np.median(dollars))
                category = ["low", "medium", "high"][int(value >= 50e6) + int(value >= 200e6)]
            elif strategy == "volatility":
                value = float(np.std(returns, ddof=1) * np.sqrt(252))
                category = ["low", "medium", "high"][int(value >= .2) + int(value >= .4)]
            else:
                value = float(returns.sum() / max(np.std(returns, ddof=1) * np.sqrt(n), 1e-12))
                category = "up" if value > 1 else "down" if value < -1 else "range"
            members.append({"symbol": symbol, "issuer": symbol, "week_id": version["week_id"],
                            "strategy_id": strategy, "group_ids": [strategy + ":" + category],
                            "version_id": version["version_id"], "status": "classified",
                            "facts": {"history_end": dates[ids[-1]], "v8_classifier_value": value}})
    write(group / "memberships.json", members)


def prepare(output):
    if output.exists():
        raise FileExistsError(output)
    protocol = json.loads((HERE / "protocol.json").read_text())
    config = json.loads((ROOT / "research/after_open_3d5pct/configs/multiscale_groups_v61.json").read_text())
    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    universe = json.loads((ROOT / config["universe"]).read_text())
    original = {m["symbol"] for m in universe["members"] if m["role"] == "candidate"}
    added = sorted(set(sum(protocol["groups"].values(), [])) - original)
    universe["members"] += [{"symbol": s, "role": "candidate", "membership_basis": "user_confirmed_current_snapshot_20260927"} for s in added]
    write(inputs / config["universe"], universe)
    provenance = []
    for symbol in sorted(original | set(added) | {"QQQ"}):
        source_root = OPTICS if symbol in added else ROOT / "market_data"
        for relative in (f"us_5m/{symbol}/2026.parquet", f"corporate_actions/{symbol}.parquet"):
            source = source_root / relative
            if not source.exists():
                raise FileNotFoundError(source)
            dest = inputs / "market_data" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            provenance.append({"source": str(source.relative_to(ROOT)), "frozen": str(dest.relative_to(output)), "sha256": vd.digest(dest)})
    dest = inputs / config["calendar"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / config["calendar"], dest)
    for name in ("versions.json", "memberships.json", "groups.json"):
        dest = inputs / config["group_run"] / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / config["group_run"] / name, dest)
    config.update(protocol=protocol["id"], sample_start=protocol["sample_start"], sample_end=protocol["sample_end"])
    write(output / "source_provenance.json", {"added": added, "files": provenance,
          "builder": vd.digest(Path(vd.__file__)), "prepare": vd.digest(Path(__file__)), "protocol": vd.digest(HERE / "protocol.json")})
    # ROOT is module-local and restored; this command runs in its own process.
    # Both bars AND corporate actions resolve to frozen inputs, never shared WIP.
    old = vd.ROOT
    try:
        vd.ROOT = inputs
        new_memberships(inputs, config, added)
        write(output / "config.json", config)
        manifest = vd.build(config, output / "dataset")
    finally:
        vd.ROOT = old
    write(output / "complete.json", {"manifest": vd.digest(output / "dataset/manifest.json"), "rows": manifest["rows"]})
    print(json.dumps({"rows": manifest["rows"], "counts": manifest["counts"], "added": added}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output.resolve())
