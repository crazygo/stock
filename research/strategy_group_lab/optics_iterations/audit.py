"""Verify frozen inputs, temporal purges, checkpoints and portfolio reconciliation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from research.group_expectation_matrix.build import file_hash, write_json
from research.strategy_group_lab.models import apply_calibration
from .train import temporal_split, inner_split
from .acquire import ROOT, FOCUS

def audit(output):
    rows = pd.read_parquet(output/"data/rows.parquet")
    labels = pd.read_parquet(output/"data/labels.parquet")
    manifest = json.loads((output/"source_manifest.json").read_text())
    counts = {"source_files":0,"checkpoint_predictions":0,"trials":0,"fit_labels":0,"calibration_labels":0,
              "inner_validation_models":0,"accounts":0,"trades":0,"open_positions":0}
    for source in manifest["source_files"]:
        assert file_hash(ROOT/source["path"]) == source["sha256"],source["path"]
        counts["source_files"] += 1
    for round_id in ["R0","R1","R2","R3"]:
        registration = json.loads((output/round_id/"registration.json").read_text())
        for path,digest in registration["files"].items():
            assert file_hash(ROOT/path) == digest,path
        trials = json.loads((output/round_id/"trials.json").read_text())
        for gran in [5,60]:
            source = "enhanced" if round_id in ["R2","R3"] else "features"
            data = dict(np.load(output/"data"/f"{source}_{gran}.npz"))
            for t in [x for x in trials if x["granularity"]==gran]:
                assert t["status"] in ["frozen_replay","trained"],t
                counts["trials"] += 1
                stem = output/round_id/t["model_id"]
                p = pd.read_parquet(stem.with_suffix(".parquet"))
                assert p.row_id.is_unique and np.isfinite(p.p).all() and p.p.between(0,1).all()
                expected = rows.date.between(t["outer_start"],t["outer_end"]).to_numpy().copy()
                if t["scope"] == "chain": expected &= rows.symbol.isin(FOCUS).to_numpy()
                np.testing.assert_array_equal(p.row_id,np.flatnonzero(expected))
                checkpoint = ROOT/t["old_model"] if round_id=="R0" else stem
                with checkpoint.with_suffix(".pkl").open("rb") as f: model = pickle.load(f)
                raw = model.predict(data,p.row_id.to_numpy())
                np.testing.assert_allclose(raw,p.raw_p,atol=1e-10)
                np.testing.assert_allclose(apply_calibration(raw,t["calibration"]),p.p,atol=1e-10)
                counts["checkpoint_predictions"] += len(p)
                if round_id != "R0":
                    lab = labels[labels.expectation_id.eq(t["expectation"])].sort_values("row_id").reset_index(drop=True)
                    fit,cal,cut = temporal_split(rows,lab,t["outer_start"],rows.symbol.isin(FOCUS).to_numpy() if t["scope"]=="chain" else None)
                    ready = pd.to_datetime(lab.label_available_at,utc=True)
                    assert not (fit&cal).any()
                    assert (ready[fit] < pd.Timestamp(cut,tz="UTC")).all()
                    assert (ready[cal] < pd.Timestamp(t["outer_start"],tz="UTC")).all()
                    assert fit.sum()==t["train_n"] and cal.sum()==t["calibration_n"]
                    counts["fit_labels"] += int(fit.sum()); counts["calibration_labels"] += int(cal.sum())
                    if round_id=="R3" and t["learning_curve"]["status"]=="inner_temporal_selection":
                        tr,va,inner = inner_split(rows,lab,fit)
                        assert (ready.iloc[tr] < pd.Timestamp(inner,tz="UTC")).all()
                        assert len(set(tr)&set(va))==0 and (ready.iloc[va]<pd.Timestamp(cut,tz="UTC")).all()
                        counts["inner_validation_models"] += 1
        print(f"audited {round_id}",flush=True)
    results = json.loads((output/"results.json").read_text())
    for r in results:
        if r["backtest_complete"]:
            assert abs(r["end_balance"]-r["initial_cash"]-r["realized_pnl"]-r["unrealized_pnl"]) < 1e-6
            assert abs(r["net_return"]-(r["end_balance"]/r["initial_cash"]-1)) < 1e-10
        counts["accounts"] += 1
        path = output/"details"/f"{r['id']}.json"
        if path.exists():
            detail = json.loads(path.read_text())
            for t in detail["trades"]:
                assert t["shares"]>0 and t["cost"]>=0
                assert pd.Timestamp(t["mark_at"])<=pd.Timestamp(r["valuation_end"]+" 16:00",tz="America/New_York")
                if t["closed"]:
                    assert abs(t["pnl"]-(t["exit_proceeds"]-t["entry_debit"]))<1e-6
                else:
                    assert abs(t["unrealized_pnl"]-(t["shares"]*t["mark_price"]-t["entry_debit"]))<1e-6
                    counts["open_positions"] += 1
                counts["trades"] += 1
    sources = [p for p in Path(__file__).parent.iterdir() if p.suffix in [".py",".md",".html"]]
    files = sources + [output/f for f in ["config.json","source_manifest.json","results.json","summary.json","predictions.parquet","facts.json","comparisons.json","latest.json"]]
    value = {"status":"pass","audited_at":datetime.now(timezone.utc).isoformat(),"counts":counts,
             "files":{str(p.relative_to(ROOT)):file_hash(p) for p in files},
             "limitations":["development replay; no untouched holdout", "current business/watchlist snapshots replayed historically", "historical available_at is an assumption, not received-at evidence", "OHLC fills and 5m drawdown; no order book or settlement simulation"]}
    write_json(output/"audit.json",value)
    print(json.dumps(value["counts"]),flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("output",type=Path)
    audit(p.parse_args().output.resolve())
