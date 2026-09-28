"""Extend coverage without changing historical source groups or checkpoints."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .acquire import FOCUS, ROOT
from research.strategy_group_lab.data import build
from research.group_expectation_matrix.build import file_hash, write_json

OLD = ROOT/"research/strategy_group_lab/runs/20260926_v1_checked"
GROUPS = {
    "optics:chain": ("光产业链·全部12股", FOCUS),
    "optics:core": ("光产业链·光学直接环节", ["LITE", "COHR", "AAOI", "CIEN", "FN", "AXTI"]),
    "optics:components": ("光产业链·器件与模块", ["LITE", "COHR", "AAOI"]),
    "optics:transport": ("光产业链·光网络", ["CIEN"]),
    "optics:materials": ("光产业链·制造与材料", ["FN", "AXTI"]),
    "optics:chips": ("光产业链·互连芯片", ["CRDO", "MRVL", "AVGO", "ALAB"]),
    "optics:switching": ("光产业链·交换网络", ["ANET", "CSCO"]),
}
SOURCES = {
    "LITE": "https://www.lumentum.com/en/products/data-center",
    "COHR": "https://www.coherent.com/networking/transceivers/telecom",
    "AAOI": "https://ao-inc.com/products/optical-transceivers/",
    "CIEN": "https://www.ciena.com/products/6500",
    "FN": "https://fabrinet.com/markets/opticalcommunications",
    "AXTI": "https://www.axt.com/",
    "CRDO": "https://credosemi.com/products/optical-dsp/",
    "MRVL": "https://www.marvell.com/products/pam-dsp.html",
    "AVGO": "https://www.broadcom.com/products/optical-networking",
    "ALAB": "https://www.asteralabs.com/products/",
    "ANET": "https://www.arista.com/en/products/platforms",
    "CSCO": "https://www.cisco.com/site/us/en/products/networking/switches/index.html",
}

def prepare(output):
    cfg = json.loads((OLD/"manifest.json").read_text())["config"]
    source = ROOT/cfg["group_source"]
    newsource = output/"group_source"
    newsource.mkdir(exist_ok=True)
    instruments = json.loads((source/"instruments.json").read_text())
    known = {x["symbol"] for x in instruments}
    for sym in FOCUS:
        if sym not in known:
            instruments.append({"symbol": sym, "security_id": "US."+sym, "name": sym,
                                "candidate": True, "roles": ["candidate"], "instrument_type": "stock",
                                "membership_basis": "20260926_optical_chain_current_snapshot"})
    provenance = []
    for item in instruments:
        sym = item["symbol"]
        for relative in [f"us_5m/{sym}/2026.parquet", f"corporate_actions/{sym}.parquet"]:
            target = output/"market_data"/relative
            original = ROOT/"market_data"/relative
            if not target.exists() and original.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
            if target.exists():
                provenance.append({"path": str(target.relative_to(ROOT)), "sha256": file_hash(target)})
    groups = json.loads((source/"groups.json").read_text())
    groups += [{"group_id": gid, "strategy_id": "optics_expansion", "name": name,
                "symbols": syms, "membership_basis": "current_snapshot_retrospective",
                "source_observed_at": "2026-09-26", "sources": {s: SOURCES[s] for s in syms}}
               for gid, (name, syms) in GROUPS.items()]
    members = json.loads((source/"memberships.json").read_text())
    versions = json.loads((source/"versions.json").read_text())
    all_versions = [v for v in versions if v["strategy_id"] == "all"]
    for v in all_versions:
        old_id = v["version_id"]
        v["version_id"] += ":optics_coverage_v1"
        for m in members:
            if m["version_id"] == old_id:
                m["version_id"] = v["version_id"]
        for sym in sorted(set(FOCUS)-known):
            members.append({"symbol": sym, "issuer": sym, "week_id": v["week_id"], "strategy_id": "all",
                            "group_ids": ["all:all"], "version_id": v["version_id"], "status": "classified", "facts": {}})
        ov = copy.deepcopy(v)
        ov.update(strategy_id="optics_expansion", version_id="optics_expansion:"+v["week_id"],
                  membership_basis="current_snapshot_retrospective", source_observed_at="2026-09-26")
        versions.append(ov)
        for sym in FOCUS:
            members.append({"symbol": sym, "issuer": sym, "week_id": v["week_id"], "strategy_id": "optics_expansion",
                            "group_ids": [gid for gid, (_, ss) in GROUPS.items() if sym in ss],
                            "version_id": ov["version_id"], "status": "classified", "facts": {}})
    for filename, value in [("groups.json", groups), ("memberships.json", members), ("versions.json", versions), ("instruments.json", instruments)]:
        write_json(newsource/filename, value)
    write_json(output/"source_manifest.json", {"source_files": provenance, "base_group_source": str(source.relative_to(ROOT)),
               "new_members": sorted(set(FOCUS)-known), "groups": GROUPS, "membership_basis": "current_snapshot_retrospective"})
    cfg.update(group_source=str(newsource.relative_to(ROOT)), bars_dir=str((output/"market_data/us_5m").relative_to(ROOT)),
               actions_dir=str((output/"market_data/corporate_actions").relative_to(ROOT)), source_end="2026-09-25", as_of="2026-09-26T00:00:01+00:00")
    cfg["folds"] = [{"id": "july", "outer_start": "2026-07-01", "outer_end": "2026-07-31"},
                    {"id": "august", "outer_start": "2026-08-03", "outer_end": "2026-08-31"},
                    {"id": "september", "outer_start": "2026-09-01", "outer_end": "2026-09-25"}]
    write_json(output/"config.json", cfg)
    data = output/"data"
    data.mkdir(exist_ok=True)
    if not (data/"complete.json").exists():
        build(cfg, data)
        write_json(data/"complete.json", {"status": "complete", "source_files": provenance})
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.output.resolve())
