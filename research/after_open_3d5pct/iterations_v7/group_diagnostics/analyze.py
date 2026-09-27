"""Read-only v7 subgroup evaluation; no fitting or prediction changes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from research.after_open_3d5pct import v6_data as vd
from research.after_open_3d5pct.train_multiscale_v6 import _folds
from research.after_open_3d5pct.iterations_v7.collect_results import normalize, summarize, digest

ROOT = vd.ROOT
BASE = ROOT / "research/after_open_3d5pct"
DATA = BASE / "runs/multiscale_groups_v61_dataset_20260926_r2"
GROUPS = ROOT / "research/group_expectation_matrix/outputs/20260925_v5"
TARGETS = [f"{h}d_{t}pct" for h in (1, 3, 5) for t in (3, 5, 8)]
CHOICES = {"B有群": "R1", "B无群": "R2", "C有群": "R2", "C无群": "R1", "C无日级": "R1"}
INNER = {
    "B有群": "v7_b_group_R1_fixed/predictions_inner.parquet",
    "B无群": "v7_b_no_group_r2/inner.parquet",
    "C有群": "v7_c_group_R2_corrected_seed3566/inner_predictions.parquet",
    "C无群": "v7_c_no_group_R1_seed3566/inner_predictions.parquet",
    "C无日级": "v7_c_no_daily_r1_3566/inner.parquet",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def factor_stats(stock_close, market_close):
    """Exactly 21 consecutive observed closes; no bridging of missing sessions."""
    a, b = np.asarray(stock_close, float), np.asarray(market_close, float)
    if a.shape != (21,) or b.shape != (21,) or not np.isfinite([a, b]).all():
        return None
    s, m = np.diff(a), np.diff(b)
    sc, mc = s-s.mean(), m-m.mean()
    if np.dot(mc, mc) <= 1e-14 or np.dot(sc, sc) <= 1e-14:
        return None
    beta = np.dot(sc, mc) / np.dot(mc, mc)
    residual = sc-beta*mc
    rho = np.dot(sc, mc) / np.sqrt(np.dot(sc, sc)*np.dot(mc, mc))
    return {"r2": float(np.clip(rho*rho, 0, 1)), "rho": float(rho), "beta": float(beta),
            "residual_daily_sigma": float(np.sqrt(np.dot(residual, residual)/18))}


def past_window(values, at, length=21):
    """Official-calendar position at is excluded, regardless of future appends."""
    if at < length:
        return None
    return values[at-length:at]


def market_features(table, xday, config, sessions):
    q = vd.load_symbol("QQQ", sessions, config)
    assert q is not None
    dates = [s["session_date"] for s in sessions]
    date_index = {d: i for i, d in enumerate(dates)}
    qclose = q.seqday[:, 12].astype(float)*5 + q.seqday[:, 0]
    qclose[~q.daily_full] = np.nan
    qavail = []
    for i, session in enumerate(sessions):
        n = session["duration_minutes"]//5
        a = q.effective_available[i, 66:66+n]
        qavail.append(a.max() if not np.isnat(a).any() else np.datetime64("NaT", "ns"))
    qavail = np.array(qavail, dtype="datetime64[ns]")
    actions = vd._split_days("QQQ")
    records = []
    for i, row in enumerate(table.itertuples()):
        d = date_index[row.session_date]
        past = past_window(qclose, d)
        decision = pd.Timestamp(row.decision_at).tz_convert("UTC").tz_localize(None).to_datetime64()
        avail = past_window(qavail, d)
        own = xday[i, -21:]
        valid = past is not None and (own[:, 10] > 0).all()
        valid = valid and avail is not None and not np.isnat(avail).any() and (avail <= decision).all()
        valid = valid and not any(dates[d-21] < a <= dates[d-1] for a in actions)
        result = factor_stats(own[:, 12].astype(float)*5+own[:, 0], past) if valid else None
        records.append({"sample_id": row.sample_id, "history_end": dates[d-1],
                        **(result or {k: None for k in ("r2", "rho", "beta", "residual_daily_sigma")})})
    return pd.DataFrame(records)


def make_groups(table, factors, train_ids):
    catalog = json.loads((GROUPS/"groups.json").read_text())
    versions = {v["version_id"]: v for v in json.loads((GROUPS/"versions.json").read_text())}
    members = {(m["week_id"], m["symbol"], m["strategy_id"]): m
               for m in json.loads((GROUPS/"memberships.json").read_text())}
    dynamic = list(vd.STRATEGIES)
    catalog = [g for g in catalog if g["strategy_id"] != "watchlist_etf"]
    catalog[0] = {**catalog[0], "name": "v7全部预测证券"}
    catalog.extend({"group_id": f"{s}:unknown", "strategy_id": s, "name": f"{s}·未知"} for s in dynamic)
    catalog.append({"group_id": "industry:unclassified", "strategy_id": "industry", "name": "未覆盖行业标签"})
    masks = {g["group_id"]: np.zeros(len(table), bool) for g in catalog}
    masks["all:all"][:] = True
    for i, row in enumerate(table.itertuples()):
        for sid in dynamic + ["industry"]:
            item = members.get((row.group_week, row.symbol, sid))
            gids = item["group_ids"] if item else []
            if sid in dynamic and gids:
                v = versions[item["version_id"]]
                decision = pd.Timestamp(row.decision_at)
                assert v["membership_basis"] == "causal_weekly_reconstruction"
                assert pd.Timestamp(v["feature_cutoff_at"]) <= decision
                assert pd.Timestamp(v["effective_from"]) <= decision < pd.Timestamp(v["effective_to"])
                assert item["facts"].get("history_end", "0000") < row.session_date
                assert len(gids) == 1
            if not gids:
                gids = ["industry:unclassified" if sid == "industry" else f"{sid}:unknown"]
            for gid in gids:
                if gid in masks:
                    masks[gid][i] = True
    thresholds = {}
    for sid, column, name in [("market_r2", "r2", "过去20日市场解释度"),
                              ("residual_sigma", "residual_daily_sigma", "过去20日残差波动")]:
        values = factors[column].to_numpy(float)
        cuts = np.nanquantile(values[train_ids], [1/3, 2/3])
        thresholds[sid] = cuts.tolist()
        for label, label_name in [("low", "低"), ("medium", "中"), ("high", "高"), ("unknown", "未知")]:
            gid = f"{sid}:{label}"
            catalog.append({"group_id": gid, "strategy_id": sid, "name": f"{name}·{label_name}"})
            valid = np.isfinite(values)
            if label == "unknown": mask = ~valid
            elif label == "low": mask = valid & (values < cuts[0])
            elif label == "medium": mask = valid & (values >= cuts[0]) & (values < cuts[1])
            else: mask = valid & (values >= cuts[1])
            masks[gid] = mask
    gid = "market_residual:low_r2_high_residual"
    catalog.append({"group_id": gid, "strategy_id": "market_residual", "name": "低市场解释度且高残差波动"})
    masks[gid] = masks["market_r2:low"] & masks["residual_sigma:high"]
    for sid in dynamic + ["market_r2", "residual_sigma"]:
        ids = [g["group_id"] for g in catalog if g["strategy_id"] == sid]
        assert np.all(np.array([masks[k] for k in ids]).sum(0) == 1), sid
    return catalog, masks, thresholds


def shrink_rate(labels, global_rate, pseudo_n):
    return (labels.sum(axis=0) + pseudo_n*global_rate) / (len(labels)+pseudo_n)


def bootstrap_weights(table, block=5, n=1000):
    dates = sorted(table.session_date.unique())
    rng = np.random.default_rng(20260927)
    starts = rng.integers(len(dates), size=(n, int(np.ceil(len(dates)/block))))
    indices = ((starts[..., None]+np.arange(block)) % len(dates)).reshape(n, -1)[:, :len(dates)]
    counts = np.array([np.bincount(row, minlength=len(dates)) for row in indices])
    return dates, counts


def date_interval(dates, counts, frame, loss_difference):
    sums = np.array([loss_difference[frame.session_date.to_numpy() == d].sum() for d in dates])
    n = np.array([(frame.session_date == d).sum() for d in dates])
    denom = counts @ n
    estimates = (counts @ sums)[denom > 0] / denom[denom > 0]
    return np.quantile(estimates, [.025, .975]).tolist() if len(estimates) else None


def within_stock_auc(y, p, symbols):
    numerator = 0.; pairs = 0; usable = 0
    for sym in np.unique(symbols):
        ix = symbols == sym
        pos = int(y[ix].sum()); neg = int(ix.sum())-pos
        if pos and neg:
            weight = pos*neg
            numerator += roc_auc_score(y[ix], p[ix])*weight
            pairs += weight; usable += 1
    return {"auc": numerator/pairs if pairs else None, "positive_negative_pairs": pairs,
            "symbols_with_both_classes": usable}


def score_group(frame, p, y, group_prior, stock_prior, dates, counts, detailed=True):
    if not len(frame): return {"n": 0, "symbols": [], "targets": {}}
    result = {"n": len(frame), "dates": int(frame.session_date.nunique()),
              "symbols": sorted(frame.symbol.unique()), "targets": {}}
    for j, name in enumerate(TARGETS):
        met = summarize(y[:, j], p[:, j])
        group_loss = (y[:, j]-group_prior[j])**2
        stock_loss = (y[:, j]-stock_prior[:, j])**2
        met.update(group_train_reference_p=float(group_prior[j]),
                   group_reference_brier=float(group_loss.mean()),
                   stock_reference_brier=float(stock_loss.mean()),
                   skill_vs_group_history=float(1-met["brier"]/group_loss.mean()) if group_loss.mean() else None,
                   skill_vs_stock_history=float(1-met["brier"]/stock_loss.mean()) if stock_loss.mean() else None)
        if j == 4:
            difference = (p[:, j]-y[:, j])**2-group_loss
            met["delta_group_reference_5date_ci"] = date_interval(dates, counts, frame, difference)
            met["delta_stock_reference_5date_ci"] = date_interval(
                dates, counts, frame, (p[:, j]-y[:, j])**2-stock_loss)
            met["within_stock"] = within_stock_auc(y[:, j], p[:, j], frame.symbol.to_numpy())
        if not detailed: met.pop("reliability")
        result["targets"][name] = met
    return result


def align(path, expected, reference):
    frame = normalize(path, reference)
    ix = expected.set_index("sample_id").loc[frame.sample_id, "row_index"].to_numpy()
    assert len(frame) == len(expected) and set(frame.sample_id) == set(expected.sample_id)
    truth = reference.set_index("sample_id").loc[frame.sample_id].reset_index()
    for column in ["symbol", "session_date"] + [f"y_{i}_{j}" for i in range(3) for j in range(3)]:
        assert np.array_equal(frame[column].to_numpy(), truth[column].to_numpy()), (path, column)
    return frame, ix


def run(output):
    if output.exists(): raise FileExistsError(output)
    config = json.loads((BASE/"configs/multiscale_groups_v61.json").read_text())
    manifest = json.loads((DATA/"manifest.json").read_text())
    required_sources = [config["calendar"], config["universe"], "market_data/us_5m/QQQ/2026.parquet",
                        "market_data/corporate_actions/QQQ.parquet"]
    required_sources += [str((GROUPS/name).relative_to(ROOT)) for name in ("groups.json", "memberships.json", "versions.json")]
    for source in required_sources:
        if source in manifest["source_hashes"]:
            assert digest(ROOT/source) == manifest["source_hashes"][source], source
    table = pd.read_parquet(DATA/"rows.parquet")
    table["row_index"] = np.arange(len(table))
    with np.load(DATA/"features.npz") as data:
        y = data["y"].reshape(-1, 9).astype(float)
        xday = data["xday"]
    assert len(table) == len(y) == 9222 and np.isin(y, [0, 1]).all()
    folds = _folds(table, config)
    assert {k: len(v) for k, v in folds.items()} == {"train": 5606, "inner": 1521, "outer": 1140}
    reference = table[["sample_id", "symbol", "session_date"]].copy()
    for j in range(9): reference[f"y_{j//3}_{j%3}"] = y[:, j]
    sessions = vd._calendar(config)
    factors = market_features(table, xday, config, sessions)
    catalog, masks, thresholds = make_groups(table, factors, folds["train"])
    global_rate = y[folds["train"]].mean(0)
    train_symbols = table.iloc[folds["train"]].symbol.to_numpy()
    priors = {sym: shrink_rate(y[folds["train"]][train_symbols == sym], global_rate, 20)
              for sym in table.symbol.unique()}
    group_priors = {gid: (global_rate if gid == "all:all" else
                         shrink_rate(y[folds["train"]][mask[folds["train"]]], global_rate, 50))
                    for gid, mask in masks.items()}
    existing = json.loads((BASE/"iterations_v7/comparison.json").read_text())
    mapping = {(r["lane"], r["round"]): r for r in existing["records"]}
    for item in existing["records"]:
        assert digest(ROOT/item["predictions"]) == item["predictions_sha256"]
    report = {"scope": "v7_frozen_predictions_group_diagnostics_exposed_development",
              "main_versions": CHOICES, "target": "3d_5pct", "targets": TARGETS,
              "thresholds_training_tertiles": thresholds, "global_train_rates": global_rate.tolist(),
              "group_reference_pseudo_n": 50, "stock_reference_pseudo_n": 20,
              "group_catalog": catalog, "panels": [], "rounds_primary": [], "stocks_primary": [],
              "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in
                                [Path(__file__), DATA/"manifest.json", DATA/"features.npz", DATA/"rows.parquet",
                                 BASE/"iterations_v7/comparison.json", *[ROOT/s for s in required_sources]]}}
    loaded = {}
    for split in ("inner", "outer"):
        dates, weights = bootstrap_weights(table.iloc[folds[split]])
        for lane, round_name in CHOICES.items():
            path = BASE/"runs"/INNER[lane] if split == "inner" else ROOT/mapping[(lane, round_name)]["predictions"]
            frame, ix = align(path, table.iloc[folds[split]], reference)
            p = frame[[f"p_{i}_{j}" for i in range(3) for j in range(3)]].to_numpy(float)
            stock_prior = np.array([priors[s] for s in frame.symbol])
            loaded[(split, lane)] = (frame, ix, p)
            report["source_hashes"][str(path.relative_to(ROOT))] = digest(path)
            for group in catalog:
                gid = group["group_id"]; take = masks[gid][ix]
                result = score_group(frame[take], p[take], y[ix][take], group_priors[gid], stock_prior[take], dates, weights)
                result.update(lane=lane, round=round_name, split=split, group_id=gid,
                              training_rows=int(masks[gid][folds["train"]].sum()))
                report["panels"].append(result)
            if split == "outer":
                for symbol in sorted(frame.symbol.unique()):
                    take = (frame.symbol == symbol).to_numpy()
                    met = summarize(y[ix][take, 4], p[take, 4]); met.pop("reliability")
                    ref = float(np.mean((y[ix][take, 4]-priors[symbol][4])**2))
                    report["stocks_primary"].append({"symbol": symbol, "lane": lane, "round": round_name,
                                                    "metrics": met, "stock_reference_brier": ref,
                                                    "skill_vs_stock_history": 1-met["brier"]/ref if ref else None})
    for item in existing["records"]:
        frame, ix = align(ROOT/item["predictions"], table.iloc[folds["outer"]], reference)
        p = frame.p_1_1.to_numpy(float)
        for group in catalog:
            gid = group["group_id"]; take = masks[gid][ix]
            met = summarize(y[ix][take, 4], p[take]) if take.any() else {"n": 0}
            met.pop("reliability", None)
            report["rounds_primary"].append({"lane": item["lane"], "round": item["round"], "group_id": gid, **met})
    candidates = sorted(m["symbol"] for m in json.loads((ROOT/config["universe"]).read_text())["members"] if m["role"] == "candidate")
    outer = table.iloc[folds["outer"]]
    report["universe"] = {"candidate_count": len(candidates), "candidates": candidates,
                          "evaluated_symbols": sorted(outer.symbol.unique()),
                          "absent_from_outer": sorted(set(candidates)-set(outer.symbol)),
                          "date_start": outer.session_date.min(), "date_end": outer.session_date.max(),
                          "rows": len(outer), "dates": int(outer.session_date.nunique())}
    output.mkdir(parents=True)
    write_json(output/"results.json", report)
    factors.to_parquet(output/"market_factors.parquet", index=False)
    # Preserve all memberships, including unknown and overlaps, for exact reproduction.
    memberships = [{"sample_id": table.iloc[i].sample_id, "group_ids": [g for g, mask in masks.items() if mask[i]]}
                   for i in range(len(table))]
    write_json(output/"memberships.json", memberships)
    write_json(output/"audit.json", {"status": "passed", "all_prediction_source_hashes_unchanged": True,
                                     "all_ids_and_nine_labels_match": True,
                                     "exclusive_groups_recover_denominator": True,
                                     "rows": len(table), "main_panels": len(report["panels"]),
                                     "round_primary_cells": len(report["rounds_primary"]),
                                     "group_count": len(catalog), "fitted_models": 0})
    return {"output": str(output), "groups": len(catalog), "panels": len(report["panels"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), ensure_ascii=False))
