"""Independent checks of saved real-source outputs and accounting identities."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .data import ROOT, split_masks
from .models import apply_calibration
from research.group_expectation_matrix.build import write_json, file_hash

def audit(output):
    manifest = json.loads((output/"manifest.json").read_text())
    cfg = manifest["config"]
    rows, labels = pd.read_parquet(output/"rows.parquet"), pd.read_parquet(output/"labels.parquet")
    paths, membership = dict(np.load(output/"paths.npz")), np.load(output/"memberships.npz")["mask"]
    grid = pd.read_parquet(output/"grid.parquet")
    results = json.loads((output/"results.json").read_text())
    bindings = {r["id"]: r for r in results}
    groups = json.loads((output/"groups.json").read_text())
    group_index = {g["group_id"]: i for i, g in enumerate(groups)}
    trials = json.loads((output/"trials.json").read_text())
    assert len(results) == 3*(58+(2*2)*(58+57))
    assert len(trials) == 2*len(results)
    assert len({(r["binding_id"], r["fold"]) for r in trials}) == len(trials)
    for source in manifest["source_files"]:
        assert file_hash(ROOT/source["path"]) == source["sha256"]
    # Legacy overlapping target must agree exactly; no old result was regenerated.
    old = pd.read_parquet(ROOT/cfg["group_source"]/"outcomes.parquet")
    old = old[old.expectation_id == "d3_r5"]
    new = labels[labels.expectation_id == "d3_r5"].merge(rows[["row_id", "symbol", "date"]], on="row_id")
    common = new.merge(old, on=["symbol", "date"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(common) == len(new)
    assert (common.status_new == common.status_old).all()
    mature = common.status_new == "mature"
    np.testing.assert_allclose(common.loc[mature, "hit_new"], common.loc[mature, "hit_old"])
    np.testing.assert_allclose(common.loc[mature, "entry_price_new"], common.loc[mature, "entry_price_old"])
    np.testing.assert_allclose(common.loc[mature, "net_proxy_new"], common.loc[mature, "net_proxy_old"])
    checked_predictions, checked_fits = 0, 0
    for t in trials:
        b = bindings[t["binding_id"]]
        fold = next(f for f in cfg["folds"] if f["id"] == t["fold"])
        ls = labels[labels.expectation_id == b["expectation"]].sort_values("row_id").reset_index(drop=True)
        fit, cal, outer = split_masks(rows, ls, fold)
        gm = membership[:, group_index[b["group_id"]]]
        preds = pd.read_parquet(output/"trials"/f"{b['id']}_{fold['id']}.parquet")
        assert np.isfinite(preds.p).all() and preds.p.between(0, 1).all()
        if len(preds):
            assert np.all(outer[preds.row_id.to_numpy()])
            assert np.all(gm[preds.row_id.to_numpy()])
        if t["status"] == "trained":
            ids = np.flatnonzero(fit & (gm if b["scope"] == "group" else True))
            assert len(ids) == t["train_n"]
            assert int(ls.hit.iloc[ids].sum()) == t["positive"]
            assert len(np.flatnonzero(cal & gm)) == t["calibration_n"]
            checked_fits += 1
        checked_predictions += len(preds)
    checkpoint_checks = []
    for algo in cfg["algorithms"]:
        for gran in cfg["granularities"]:
            b = next(b for b in results if b["algorithm"] == algo and b["granularity"] == gran and b["group_id"] == "all:all")
            fold = "august"
            stem = output/"trials"/f"{b['id']}_{fold}"
            data = dict(np.load(output/f"features_{gran}.npz"))
            with stem.with_suffix(".pkl").open("rb") as f:
                model = pickle.load(f)
            pred = pd.read_parquet(stem.with_suffix(".parquet")).iloc[::97]
            raw = model.predict(data, pred.row_id.to_numpy())
            np.testing.assert_allclose(raw, pred.raw_p, rtol=2e-6, atol=1e-7)
            meta = json.loads(stem.with_suffix(".json").read_text())
            np.testing.assert_allclose(apply_calibration(raw, meta["calibration"]), pred.p, rtol=2e-6, atol=1e-7)
            checkpoint_checks.append({"algorithm": algo, "granularity": gran, "predictions": len(pred)})
    trade_checks, accounts = 0, 0
    cost = cfg["portfolio"]["cost_bps"]/10000
    details = [(r, json.loads((output/"details"/f"{r['id']}.json").read_text())) for r in results]
    router = json.loads((output/"router.json").read_text())
    details.append((router["metrics"], router))
    for result, detail in details:
        cash = float(cfg["portfolio"]["initial_cash"])
        pnl = 0.
        events, held = [], set()
        for t in detail["trades"]:
            exp = next(e for e in cfg["expectations"] if e["id"] == t["expectation_id"])
            a, z = t["entry_pos"], t["exit_pos"]
            entry = paths[t["symbol"]][a, 0]
            assert t["entry_price"] == entry
            window = paths[t["symbol"]][a:a+exp["days"]*78]
            touch = np.flatnonzero(window[:, 1] >= entry*(1+exp["target"]))
            assert bool(len(touch)) == t["hit"]
            assert z == a+(int(touch[0]) if len(touch) else exp["days"]*78-1)
            expected_exit = entry*(1+exp["target"]) if len(touch) else window[-1, 3]
            assert abs(expected_exit-t["exit_price"]) < 1e-10
            assert abs(t["pnl"]-(t["shares"]*(expected_exit*(1-cost)-entry*(1+cost)))) < 1e-7
            # Entries occur at bar start and exits at bar end, preserving same-bar order.
            events.append((2*a, "entry", t))
            events.append((2*z+1, "exit", t))
            pnl += t["pnl"]
            trade_checks += 1
        for _, kind, t in sorted(events, key=lambda x: x[0]):
            if kind == "entry":
                assert t["symbol"] not in held
                held.add(t["symbol"])
                cash -= t["entry_debit"]
                assert len(held) <= cfg["portfolio"]["max_positions"]
                assert cash >= -1e-6
            else:
                held.remove(t["symbol"])
                cash += t["exit_proceeds"]
        assert not held
        assert abs(cash-100000-pnl) < 1e-6
        if result.get("end_balance") is not None:
            assert abs(cash-result["end_balance"]) < 1e-6
        accounts += 1
    for w in router["schedule"]:
        for choice in w["selected"]:
            assert pd.Timestamp(choice["latest_label_at"]) < pd.Timestamp(w["evidence_cutoff"])
            assert choice["candidate_dates"] >= cfg["router"]["minimum_dates"]
            assert choice["candidate_n"] >= cfg["router"]["minimum_candidates"]
    evidence = {"status": "passed", "source_hashes": len(manifest["source_files"]),
                "legacy_3d5_labels_matched": len(common), "training_contracts_checked": checked_fits,
                "prediction_rows_checked": checked_predictions, "checkpoints_repredicted": checkpoint_checks,
                "accounts_reconciled": accounts, "trades_independently_replayed": trade_checks,
                "router_evidence_before_selection": True}
    write_json(output/"audit.json", evidence)
    print(json.dumps(evidence, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
