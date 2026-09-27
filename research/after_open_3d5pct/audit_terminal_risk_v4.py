"""Independent post-run integrity checks for the terminal-risk experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .terminal_risk_v4 import REPO, full_sample_es_loss, sha256
from .train_terminal_risk_v4 import fold_indices


def audit(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    config = json.loads((run / "config.json").read_text())
    provenance = json.loads((run / "provenance_at_launch.json").read_text())
    if sha256(run / "provenance_at_launch.json") != manifest["provenance_at_launch_sha256"]:
        raise ValueError("launch provenance altered")
    registered = REPO / "research/after_open_3d5pct/configs/terminal_risk_v4.json"
    if sha256(registered) != manifest["config_sha256"] or config != json.loads(registered.read_text()):
        raise ValueError("registered config mismatch")
    registration = REPO / "research/after_open_3d5pct/docs/17_terminal_risk_v4_registration.md"
    if sha256(registration) != manifest["registration_sha256"]:
        raise ValueError("training registration changed after launch")
    for rel, expected in provenance["files_sha256"].items():
        if sha256(REPO / rel) != expected:
            raise ValueError(f"launch-pinned source changed: {rel}")
    for name, expected in manifest["result_artifacts_sha256"].items():
        if sha256(run / name) != expected:
            raise ValueError(f"v4 result changed: {name}")
    for name, expected in manifest["model_artifacts_sha256"].items():
        if sha256(run / "models" / name) != expected:
            raise ValueError(f"v4 model changed: {name}")
    old = REPO / config["source_run"]
    old_manifest = json.loads((old / "manifest.json").read_text())
    if sha256(old / "manifest.json") != manifest["source"]["source_manifest_sha256"]:
        raise ValueError("frozen v3 manifest changed")
    market_sources = [rel for rel in old_manifest["sources_sha256"] if rel.startswith(config["source_dir"] + "/")]
    for rel in market_sources:
        if sha256(REPO / rel) != old_manifest["sources_sha256"][rel]:
            raise ValueError(f"frozen market source changed: {rel}")
    for rel, digest in manifest["source"]["metadata_hashes_verified"].items():
        if digest != old_manifest["sources_sha256"].get(rel) or sha256(REPO / rel) != digest:
            raise ValueError(f"metadata source not frozen: {rel}")
    terminal = pd.read_parquet(run / "terminal_outcomes.parquet")
    directly_rederived = [rel for rel in market_sources
                          if Path(rel).parent.name in set(terminal.symbol)]
    if len(directly_rederived) != manifest["source"]["source_hashes_verified_count"]:
        raise ValueError("reported directly rederived source count inaccurate")
    prior = pd.read_parquet(old / "outcomes.parquet")
    if len(terminal) != len(prior) or (terminal.sample_id != prior.sample_id).any():
        raise ValueError("terminal/prior sample mismatch")
    if (terminal.status != "mature").any() or not (terminal.target.to_numpy(int) == prior.target.to_numpy(int)).all():
        raise ValueError("full window maturity or touch disagreement")
    if not np.allclose(terminal.entry_price, prior.entry_price, rtol=0, atol=1e-6):
        raise ValueError("entry price mismatch")
    if (pd.to_datetime(terminal.label_available_at, utc=True) !=
            pd.to_datetime(prior.label_available_at, utc=True)).any():
        raise ValueError("label availability mismatch")
    rows = pd.read_parquet(old / "features.parquet").merge(
        prior[["sample_id", "label_end_at", "label_available_at"]], on="sample_id", validate="one_to_one")
    trials = [json.loads(line) for line in (run / "trials.jsonl").read_text().splitlines()]
    if len(trials) != 8 or any(t["status"] != "completed" for t in trials):
        raise ValueError("expected eight completed touch trials")
    for t in trials:
        if t["candidate"].startswith("lgbm") and list(t["training"]["early_curve"]["valid_0"]) != ["brier"]:
            raise ValueError("LightGBM did not early-stop on registered Brier alone")
    folds = {}
    for fold in config["folds"]:
        ix, counts = fold_indices(rows, fold)
        if len(np.concatenate(list(ix.values()))) != len(set(np.concatenate(list(ix.values())))):
            raise ValueError("nested temporal phases overlap")
        outer = terminal.iloc[ix["outer"]]
        folds[fold["name"]] = {"split": counts, "outer_n": len(outer),
                               "outer_touches": int(outer.target.sum()),
                               "outer_touch_rate": float(outer.target.mean()),
                               "outer_mean_net": float(outer.net_evaluation_return.mean()),
                               "outer_full_sample_es95_loss": full_sample_es_loss(
                                   outer.net_evaluation_return.to_numpy(), config["tail_fraction"]),
                               "inner_selected_candidate": manifest["folds"][fold["name"]]["selected_candidate"]}
    result = {"audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "run": str(run), "scope": "exposed_development_integrity_only",
              "frozen_market_sources_verified": len(market_sources),
              "directly_rederived_market_sources": len(directly_rederived),
              "frozen_metadata_sources_verified": len(manifest["source"]["metadata_hashes_verified"]),
              "model_artifacts_verified": len(manifest["model_artifacts_sha256"]),
              "result_artifacts_verified": len(manifest["result_artifacts_sha256"]),
              "full_window_mature_rows": len(terminal), "entry_touch_availability_mismatches": 0,
              "lgbm_early_metric": "brier_only", "folds": folds,
              "status": "integrity_passed_not_effectiveness_release"}
    (run / "integrity_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve()), ensure_ascii=False, indent=2))
