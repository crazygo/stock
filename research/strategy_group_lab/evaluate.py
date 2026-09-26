"""All registered results, date-block uncertainty and a causal weekly router."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .portfolio import simulate
from research.group_expectation_matrix.build import write_json

def metrics(frame, dates, days, cfg):
    valid = frame[frame.status == "mature"]
    r = {"prediction_n": len(frame), "mature_n": len(valid), "unresolved_labels": len(frame)-len(valid),
         "date_n": int(valid.date.nunique()), "symbol_n": int(valid.symbol.nunique())}
    if valid.empty:
        return {**r, "accuracy": None, "brier": None, "brier_improvement": None, "candidate_n": 0,
                "precision": None, "coverage": None, "ci_status": "no_mature_predictions"}
    y, p, b = valid.hit.to_numpy(float), valid.p.to_numpy(float), valid.b0.to_numpy(float)
    positive = p >= cfg["threshold"]
    diff = (b-y)**2-(p-y)**2
    r.update(accuracy=float(((p >= .5) == y).mean()), majority_accuracy=float(max(y.mean(), 1-y.mean())),
             baseline_accuracy=float(((b >= .5) == y).mean()), hit_rate=float(y.mean()),
             brier=float(np.mean((p-y)**2)), raw_brier=float(np.mean((valid.raw_p.to_numpy(float)-y)**2)),
             b0_brier=float(np.mean((b-y)**2)), brier_improvement=float(diff.mean()),
             log_loss=float(-np.mean(y*np.log(np.clip(p, 1e-8, 1))+(1-y)*np.log(np.clip(1-p, 1e-8, 1)))),
             candidate_n=int(positive.sum()), candidate_hit_n=int(y[positive].sum()),
             precision=float(y[positive].mean()) if positive.any() else None, coverage=float(positive.mean()))
    reliability = []
    bins = np.minimum((p*10).astype(int), 9)
    for index, lo in enumerate(np.arange(0, 1, .1)):
        q = bins == index
        if q.any():
            reliability.append({"lo": round(float(lo), 1), "n": int(q.sum()), "predicted": float(p[q].mean()), "observed": float(y[q].mean())})
    r["reliability"] = reliability
    block = max(5, days)
    if len(dates) < block*4 or valid.date.nunique() < block*4:
        r.update(ci_status="insufficient_calendar_blocks", improvement_ci95=None, p_value=None)
        return r
    daily = pd.DataFrame({"date": valid.date, "diff": diff}).groupby("date").agg(total=("diff", "sum"), n=("diff", "size")).reindex(dates, fill_value=0)
    sums, n = daily.total.to_numpy(float), daily.n.to_numpy(float)
    rng = np.random.default_rng(cfg["seed"])
    starts = rng.integers(0, len(dates)-block+1, size=(cfg["bootstrap_replicates"], int(np.ceil(len(dates)/block))))
    ix = (starts[:, :, None]+np.arange(block)).reshape(cfg["bootstrap_replicates"], -1)[:, :len(dates)]
    den = n[ix].sum(1)
    good = den > 0
    boot = sums[ix].sum(1)[good]/den[good]
    # One-sided paired cluster t approximation on non-overlapping date blocks.
    # Bootstrap draws are only for intervals; 400 draws cannot resolve a Holm
    # threshold across hundreds of comparisons.
    block_sums = np.asarray([sums[j:j+block].sum() for j in range(0, len(dates), block)])
    block_n = np.asarray([n[j:j+block].sum() for j in range(0, len(dates), block)])
    occupied = int((block_n > 0).sum())
    k = occupied
    if k < 4:
        r.update(ci_status="insufficient_occupied_blocks", improvement_ci95=None, p_value=None)
        return r
    se = np.sqrt(k/(k-1)*np.sum((block_sums-r["brier_improvement"]*block_n)**2))/n.sum()
    p_value = float(student_t.sf(r["brier_improvement"]/se, k-1)) if se > 1e-14 else 1.
    r.update(ci_status="development_block_bootstrap", improvement_ci95=np.quantile(boot, [.025, .975]).tolist(),
             p_value=p_value if occupied >= 4 else None, block_sessions=block,
             occupied_blocks=occupied, p_method="one_sided_paired_date_block_t_approximation")
    return r

def holm(results):
    eligible = sorted([r for r in results if r["algorithm"] != "B0" and r.get("p_value") is not None], key=lambda x: x["p_value"])
    previous = 0.
    for i, r in enumerate(eligible):
        previous = max(previous, min(1., (len(eligible)-i)*r["p_value"]))
        r["holm_p"] = previous
        r["holm_pass"] = previous < .05 and r["brier_improvement"] > 0
    return len(eligible)

def route(frames, bindings, grid, cfg):
    schedule, signals = [], []
    calendar = sorted(grid.date.unique())
    outer_dates = sorted({d for f in cfg["folds"] for d in calendar if f["outer_start"] <= d <= f["outer_end"]})
    weeks = {}
    for day in outer_dates:
        iso = pd.Timestamp(day).isocalendar()
        weeks.setdefault(f"{iso.year}-W{iso.week:02d}", []).append(day)
    for week, days in weeks.items():
        first = days[0]
        cutoff = pd.Timestamp(first+" 09:30", tz="America/New_York").tz_convert("UTC")
        i = calendar.index(first)
        earliest = calendar[max(0, i-cfg["router"]["lookback_sessions"])]
        ranking, rejected = [], {"short_evidence": 0, "no_brier_improvement": 0, "nonpositive_net": 0}
        for bid, frame in frames.items():
            binding = bindings[bid]
            if binding["algorithm"] == "B0" or frame.empty:
                continue
            ready = pd.to_datetime(frame.label_available_at, utc=True)
            prior = frame[(frame.date >= earliest) & (frame.date < first) & (ready < cutoff) & (frame.status == "mature")]
            candidates = prior[prior.p >= cfg["threshold"]]
            if len(candidates) < cfg["router"]["minimum_candidates"] or candidates.date.nunique() < cfg["router"]["minimum_dates"]:
                rejected["short_evidence"] += 1
                continue
            skill = np.mean((prior.b0-prior.hit)**2-(prior.p-prior.hit)**2)
            if skill <= 0:
                rejected["no_brier_improvement"] += 1
                continue
            net = candidates.net_proxy.mean()
            if net <= 0:
                rejected["nonpositive_net"] += 1
                continue
            score = net/(candidates.holding_minutes_proxy.mean()/390)
            ranking.append({"binding_id": bid, "score": float(score), "candidate_n": len(candidates),
                            "candidate_dates": int(candidates.date.nunique()), "brier_improvement": float(skill),
                            "mean_net_proxy": float(net), "latest_label_at": prior.label_available_at.max()})
        selected = sorted(ranking, key=lambda x: (-x["score"], x["binding_id"]))[:cfg["router"]["top_combinations"]]
        schedule.append({"week": week, "evidence_cutoff": cutoff.isoformat(), "lookback_from": earliest,
                         "selected": selected, "eligible_combinations": len(ranking), "rejected_counts": rejected,
                         "status": "selected" if selected else "cash_insufficient_evidence"})
        for rank, choice in enumerate(selected):
            bid = choice["binding_id"]
            f = frames[bid]
            s = f[f.date.isin(days)][["row_id", "p"]].copy()
            s["binding_id"], s["expectation_id"], s["priority"] = bid, bindings[bid]["expectation"], rank
            signals.append(s)
    combined = pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(columns=["row_id", "p", "binding_id", "expectation_id", "priority"])
    return combined, schedule

def evaluate(cfg, output):
    rows = pd.read_parquet(output/"rows.parquet")
    labels = pd.read_parquet(output/"labels.parquet")
    grid = pd.read_parquet(output/"grid.parquet")
    paths = dict(np.load(output/"paths.npz"))
    bindings = {r["id"]: r for r in json.loads((output/"bindings.json").read_text())}
    trials = json.loads((output/"trials.json").read_text())
    membership = np.load(output/"memberships.npz")["mask"]
    groups = json.loads((output/"groups.json").read_text())
    group_indices = {g["group_id"]: i for i, g in enumerate(groups)}
    outer_mask = np.zeros(len(rows), bool)
    for f in cfg["folds"]:
        outer_mask |= rows.date.between(f["outer_start"], f["outer_end"])
    dates = sorted({d for f in cfg["folds"] for d in grid.date.unique() if f["outer_start"] <= d <= f["outer_end"]})
    results, frames, details, prediction_parts = [], {}, {}, []
    detail_dir = output/"details"
    detail_dir.mkdir(exist_ok=True)
    for i, (bid, binding) in enumerate(bindings.items()):
        preds = pd.concat([pd.read_parquet(output/"trials"/f"{bid}_{f['id']}.parquet") for f in cfg["folds"]], ignore_index=True)
        frame = preds.merge(rows, on="row_id", validate="one_to_one").merge(labels[labels.expectation_id == binding["expectation"]], on="row_id", validate="one_to_one")
        frames[bid] = frame
        m = metrics(frame, dates, binding["days"], cfg)
        signal = preds[["row_id", "p"]].copy()
        signal["binding_id"], signal["expectation_id"] = bid, binding["expectation"]
        bt, detail = simulate(signal, rows, labels, paths, grid, cfg)
        expected = int(np.sum(outer_mask & membership[:, group_indices[binding["group_id"]]]))
        t = [r for r in trials if r["binding_id"] == bid]
        complete_folds = all(x["status"] in ["trained", "baseline"] for x in t)
        row = {**binding, **m, **bt, "expected_feature_n": expected, "prediction_coverage": len(frame)/expected if expected else None,
               "trained_folds": sum(x["status"] in ["trained", "baseline"] for x in t),
               "folds_complete": complete_folds, "trial_statuses": [x["status"] for x in t],
               "trial_reasons": [x.get("reason") for x in t if x.get("reason")]}
        if not complete_folds:
            row.update(net_return=None, end_balance=None, max_drawdown=None, simulation_status="incomplete_model_folds")
        elif not len(frame):
            row.update(net_return=None, end_balance=None, max_drawdown=None, backtest_complete=False,
                       simulation_status="no_eligible_observations")
        results.append(row)
        monthly = {f["id"]: metrics(frame[frame.fold == f["id"]], [d for d in dates if f["outer_start"] <= d <= f["outer_end"]], binding["days"], cfg) for f in cfg["folds"]}
        detail.update(trials=t, monthly=monthly, reliability=m.get("reliability", []),
                      evaluation_symbols=sorted(frame.symbol.unique()))
        write_json(detail_dir/f"{bid}.json", detail)
        details[bid] = detail
        prediction_parts.append(preds.assign(binding_id=bid, expectation_id=binding["expectation"]))
        if i % 100 == 0:
            print(f"evaluation {i+1}/{len(bindings)}", flush=True)
    comparisons = holm(results)
    signals, schedule = route(frames, bindings, grid, cfg)
    router_metrics, router_detail = simulate(signals, rows, labels, paths, grid, cfg)
    router = {"metrics": router_metrics, "schedule": schedule, **router_detail}
    write_json(output/"router.json", router)
    write_json(output/"results.json", results)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in results]).to_csv(output/"results.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_parquet(output/"predictions.parquet", index=False)
    summary = {"binding_count": len(results), "model_bindings": sum(r["algorithm"] != "B0" for r in results),
               "baseline_bindings": sum(r["algorithm"] == "B0" for r in results), "trial_count": len(trials),
               "trial_status_counts": pd.Series([t["status"] for t in trials]).value_counts().to_dict(),
               "holm_test_count": comparisons, "holm_pass_count": sum(r.get("holm_pass", False) for r in results),
               "successful_binding_count": sum(r["folds_complete"] and r["prediction_n"] > 0 for r in results),
               "router": router_metrics, "outer_dates": len(dates), "decision_from": dates[0], "decision_to": dates[-1]}
    write_json(output/"summary.json", summary)
    from .render import render
    render(output, cfg, results, details, router, summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
