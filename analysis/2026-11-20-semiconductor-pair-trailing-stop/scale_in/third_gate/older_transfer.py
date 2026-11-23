"""Earlier-time transfer with a fresh Jan-Feb model and two frozen March-July routes."""
from collections import Counter
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import coverage as coverage_rules
from study_gates import FEE, NOTIONAL, PATHS, VIRTUAL, closed_outcome, late_account, observe
import study_scale
from study_scale import Context, score

HERE = Path(__file__).resolve().parent
PERIODS = {"old_train": ("2026-01-01", "2026-02-13"), "old_cal": ("2026-02-16", "2026-02-27"),
           "old_eval": ("2026-03-02", "2026-07-24")}


def coverage():
    with patch.object(coverage_rules, "PHASES", PERIODS):
        raw = coverage_rules.build()
    result = {k: raw[k] for k in ("ranges", "calendar", "sources", "blocks", "totals")}
    result["excluded"] = {}
    for phase, excluded in raw["excluded"].items():
        reasons = {}
        for day, issues in excluded["data_or_action"].items():
            reasons[day] = {s: {"issue_counts": {k: len(v) for k,v in fault["issues"].items()},
                                "actions": fault["corporate_actions"]} if s != "calendar" else fault
                            for s, fault in issues.items()}
        result["excluded"][phase] = {"data_or_action": reasons, "valid_singletons": excluded["valid_singletons"]}
    result["day_row_hashes_sha256"] = {
        s: sha256(json.dumps({d:v["sha256"] for d,v in days.items()}, sort_keys=True).encode()).hexdigest()
        for s, days in raw["daily"].items()}
    return result


def fit_points(blocks, for_training):
    rows, labels, audit = [], [], {}
    for block in blocks:
        data = score.load(block["name"])
        assert sorted(set(data["days"])) == block["dates"]
        _, trades, features = score.original(data, "OLHC")
        used, wins = 0, 0
        for x, t in zip(features, trades, strict=True):
            if x is None or (for_training and t["exit_reason"] == "open"):
                continue
            rows.append(x)
            used += 1
            if for_training:
                labels.append(int(t["net"] > 0))
                wins += labels[-1]
        audit[block["name"]] = {"original_buys": len(trades), "scored_used": used}
        if for_training:
            audit[block["name"]]["closed_wins"] = wins
    return rows, labels, audit


def train_model(blocks):
    X, y, train = fit_points(blocks["old_train"], True)
    standard = StandardScaler().fit(X)
    scaled = standard.transform(X)
    clf = LogisticRegression(C=.3, solver="lbfgs", max_iter=1000).fit(scaled, y)
    model = {"mean": standard.mean_.tolist(), "scale": standard.scale_.tolist(),
             "weights": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}
    assert np.allclose(score.portable_scores(X, model), clf.predict_proba(scaled)[:, 1], atol=1e-12)
    cal, _, cal_blocks = fit_points(blocks["old_cal"], False)
    cal_scores = score.portable_scores(cal, model)
    thresholds = {k:float(np.quantile(cal_scores, q)) for k, q in score.QUANTILES.items()}
    return {"features":score.FEATURES, "model":model, "thresholds":thresholds, "sklearn":sklearn_version,
            "C":.3, "train_rows":len(y), "train_wins":sum(y), "cal_rows":len(cal),
            "train_blocks":train, "calibration_blocks":cal_blocks}


def empty_group():
    return {"buys": 0, "sells": 0, "win": 0, "loss": 0, "flat": 0, "open": 0,
            "win_net": 0., "loss_net": 0., "flat_net": 0., "open_net": 0., "fees": 0., "gross": 0., "net": 0.}


def empty_table():
    return dict({name: empty_group() for name in ("baseline", "progress", "retained", "vetoed", "new")},
                random_expected_net=0., random_expected_buys=0., delta=0., excess_over_random=0.)


def count_trade(group, trade):
    outcome = closed_outcome(trade)
    closed = int(outcome != "open")
    group[outcome] += 1
    group["buys"] += 1
    group["sells"] += closed
    group[outcome + "_net"] += trade["net"]
    group["net"] += trade["net"]
    group["gross"] += trade["gross"]
    group["fees"] += FEE * (1+closed)


def totals(table):
    for g in ("baseline", "progress", "retained", "vetoed", "new"):
        group = table[g]
        assert isclose(group["gross"]-group["fees"], group["net"], abs_tol=1e-8)
        assert group["buys"] == sum(group[o] for o in ("win", "loss", "flat", "open"))
    assert isclose(table["retained"]["net"] + table["vetoed"]["net"], table["baseline"]["net"], abs_tol=1e-8)
    assert isclose(table["retained"]["net"] + table["new"]["net"], table["progress"]["net"], abs_tol=1e-8)
    table["delta"] = table["progress"]["net"] - table["baseline"]["net"]
    table["excess_over_random"] = table["progress"]["net"] - table["random_expected_net"]


def replay(blocks, fitted):
    months = sorted({day[:7] for b in blocks for day in b["dates"]})
    result = {path: dict(months={m: empty_table() for m in months}, overall=empty_table(), blocks={})
              for path in PATHS}
    assert (VIRTUAL.amounts,VIRTUAL.wait,VIRTUAL.seed,VIRTUAL.mode) == ((100,200,300),(1,1),"q50","step")
    # Context resolves its runtime globals in study_scale. Every seed, stage and scoring call
    # below is inside this scope, using the earlier fitted standardizer, coefficients and cuts.
    with patch.object(study_scale, "MODEL", fitted), patch.object(study_scale, "LEVELS", fitted["thresholds"]):
        for block in blocks:
            ctx = Context(block["name"])
            assert sorted(set(ctx.data["days"])) == block["dates"]
            for path, report in result.items():
                paper = ctx.execute(VIRTUAL, path, FEE)["lots"]
                events = {e["entry_i"]:e for e in observe(ctx, paper)}
                data = dict(ctx.data, path=path)
                base = late_account(data, paper, 3, NOTIONAL)
                kept_paper = [t for t in paper if t["stage"] == 3 and events[t["entry_i"]]["progress"]]
                gated = late_account(data, kept_paper, 3, NOTIONAL)
                source = {t["entry_i"]:t for t in base["trades"]}
                actual = {t["entry_i"]:t for t in gated["trades"]}
                shared = source.keys() & actual.keys()
                eligible_but_cash_blocked = {i for i in source.keys()-actual.keys() if events[i]["progress"]}
                if eligible_but_cash_blocked:
                    raise RuntimeError("Separate gate vetoes from newly cash-blocked trades.")
                if shared != actual.keys():
                    # If any new actual buy exists, do not silently change the random comparator.
                    assert len(actual) <= len(source), "Matched draw would exceed original opportunity count."
                random_fraction = len(actual)/len(source) if source else 0.
                table = empty_table()
                ledger = []
                for i in sorted(source.keys() | actual.keys()):
                    e = events[i]
                    row = {"date": e["date"], "entry": data["times"][i].isoformat(), "symbol": e["symbol"],
                           "decision": e["decision"], "prev_close":e["close"], "paper_second_fill":e["second_entry"],
                           "progress_pass":e["progress"]}
                    trade = source.get(i) or actual[i]
                    if i in shared:
                        assert source[i] == actual[i]
                    classification = "retained" if i in shared else "vetoed" if i in source else "new"
                    row.update(reason=trade["reason"], outcome=closed_outcome(trade), net=trade["net"], status=classification)
                    ledger.append(row)
                    for out in (table, report["overall"], report["months"][e["date"][:7]]):
                        count_trade(out[classification], trade)
                        if i in source:
                            count_trade(out["baseline"], source[i])
                            out["random_expected_buys"] += random_fraction
                            out["random_expected_net"] += random_fraction * source[i]["net"]
                        if i in actual:
                            count_trade(out["progress"], actual[i])
                totals(table)
                for key, acc in (("baseline", base),("progress",gated)):
                    assert isclose(table[key]["net"], acc["end_equity"]-1000, abs_tol=1e-8)
                    assert table[key]["buys"] == acc["buys"]
                assert isclose(table["random_expected_buys"], gated["buys"], abs_tol=1e-8)
                report["blocks"][block["name"]] = {"summary":table, "opportunities":ledger,
                    "paper_thirds":len(events), "progress_threshold_pass":len(kept_paper),
                    "cash_blocked":{"baseline":base["cash_blocked"], "progress":gated["cash_blocked"]}}
    for report in result.values():
        totals(report["overall"])
        for table in report["months"].values():
            totals(table)
        for key in ("delta", "random_expected_net","excess_over_random"):
            assert isclose(sum(t[key] for t in report["months"].values()), report["overall"][key], abs_tol=1e-8)
    return result


def main():
    source = coverage()
    block_map = {b["name"]:(b["start"],b["end"]) for records in source["blocks"].values() for b in records}
    with patch.dict(score.study.BLOCKS, block_map):
        fitted = train_model(source["blocks"])
        comparison = replay(source["blocks"]["old_eval"], fitted)
    months = Counter(d[:7] for b in source["blocks"]["old_eval"] for d in b["dates"])
    document = {"study":"Post-hoc historical transfer using a different, earlier fitted model. No promotion.",
        "protocol_sha256":sha256((HERE/"older_protocol.md").read_bytes()).hexdigest(),
        "virtual":VIRTUAL.record(), "initial_cash_per_block":1000, "notional":NOTIONAL, "fee_per_fill":FEE,
        "extra_progress_threshold":2*FEE/NOTIONAL, "coverage":source, "valid_eval_dates_by_month":dict(months),
        "model":fitted, "results":comparison}
    output = HERE/"older_transfer.json"
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False, default=str)+"\n")
    for path, results in comparison.items():
        for period, table in [*results["months"].items(), ("overall", results["overall"])]:
            print(path, period, *(round(table[k]["net"],2) for k in ("baseline","progress")),
                  round(table["random_expected_net"],2), "buy", table["progress"]["buys"],
                  "veto W/L", table["vetoed"]["win"], table["vetoed"]["loss"])


if __name__ == "__main__":
    main()
