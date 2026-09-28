"""Discover and select interpretable group rules; never read confirmation outcomes."""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(os.environ.get("TOUCH_RESEARCH_RUN", str(Path(__file__).resolve().parent)))
SOURCE = ROOT / "research/group_expectation_matrix/outputs/20260925_v5"
FEATURES = json.loads((HERE / "feature_definitions.json").read_text())


def wilson(h, n):
    if not n:
        return 0.
    z = 1.96
    p = h / n
    return (p + z*z/(2*n) - z*math.sqrt((p*(1-p)+z*z/(4*n))/n)) / (1+z*z/n)


def rule_text(rule):
    terms = []
    for feature, operator, threshold in rule:
        name, unit, _ = FEATURES[feature]
        value = f"{threshold:.1%}" if unit in ("pct", "ratio") else f"{threshold:.2f}倍"
        terms.append(f"{name}{'≥' if operator == 'ge' else '≤'}{value}")
    return " 且 ".join(terms)


def mask_rule(frame, rule):
    mask = np.ones(len(frame), dtype=bool)
    for feature, operator, threshold in rule:
        values = frame[feature].to_numpy(float)
        mask &= np.isfinite(values) & ((values >= threshold) if operator == "ge" else (values <= threshold))
    return mask


def describe(frame, mask):
    selected = frame.iloc[np.flatnonzero(mask)]
    n = len(selected)
    if not n:
        return {"n": 0, "hits": 0, "rate": None, "dates": 0, "stocks": 0, "lcb": 0., "baseline": None, "lift": None}
    hits = int(selected.target.sum())
    daily_baseline = frame.groupby("date").target.mean()
    matched = float(selected.date.map(daily_baseline).mean())
    return {"n": n, "hits": hits, "rate": hits/n, "dates": selected.date.nunique(),
            "stocks": selected.symbol.nunique(), "lcb": wilson(hits, n),
            "baseline": matched, "lift": hits/n - matched}


def atom_rules(discovery):
    rules = []
    for feature, (_, unit, _) in FEATURES.items():
        values = discovery[feature].dropna().to_numpy()
        if len(values) < 100 or np.ptp(values) == 0:
            continue
        quantiles = np.quantile(values, [.1, .25, .5, .75, .9])
        decimals = 3 if unit == "pct" else 2
        fixed = ([-.05, -.03, -.02, -.01, 0., .01, .02, .03, .05] if unit == "pct"
                 else [.25, .5, .6, .7, .8, .9] if unit == "ratio"
                 else [.75, 1., 1.25, 1.5, 2., 3.])
        cuts = sorted(set([round(float(x), decimals) for x in quantiles] + fixed))
        for cut in cuts:
            if np.min(values) <= cut <= np.max(values):
                for operator in ("ge", "le"):
                    rules.append(((feature, operator, cut),))
    return rules


def search_group(discovery, selection, rules, dm, sm, group, checkpoint, min_samples=40, min_dates=15):
    """Search single rules and a bounded AND beam, preserving discovery selection."""
    records = []
    seen = set()
    def consider(rule, train_mask, select_mask):
        if int(train_mask.sum()) < min_samples:
            return None
        td = describe(discovery, train_mask)
        if td["dates"] < min_dates:
            return None
        key = tuple(sorted(rule))
        if key in seen:
            return None
        seen.add(key)
        sd = describe(selection, select_mask)
        item = {"group_id": group["group_id"], "strategy_id": group["strategy_id"], "group_name": group["name"],
                "checkpoint": checkpoint, "rule": list(rule), "text": rule_text(rule), "terms": len(rule),
                "discovery": td, "selection": sd}
        records.append(item)
        return (item, train_mask, select_mask)
    singles = []
    for j, rule in enumerate(rules):
        found = consider(rule, dm[:, j], sm[:, j])
        if found:
            singles.append(found)
    pool, features = [], set()
    for item in sorted(singles, key=lambda x: x[0]["discovery"]["lcb"], reverse=True):
        feature = item[0]["rule"][0][0]
        if feature not in features:
            pool.append(item)
            features.add(feature)
        if len(pool) == 12:
            break
    pairs = []
    for a, b in combinations(pool, 2):
        found = consider(a[0]["rule"] + b[0]["rule"], a[1] & b[1], a[2] & b[2])
        if found:
            pairs.append(found)
    for pair in sorted(pairs, key=lambda x: x[0]["discovery"]["lcb"], reverse=True)[:12]:
        used = {a[0] for a in pair[0]["rule"]}
        for single in pool:
            if single[0]["rule"][0][0] not in used:
                consider(pair[0]["rule"] + single[0]["rule"], pair[1] & single[1], pair[2] & single[2])
    return records


def main():
    if (HERE / "frozen_candidates.json").exists():
        raise FileExistsError("Frozen candidate selection exists; do not tune after confirmation")
    panel = pd.read_parquet(HERE / "features.parquet")
    # Remove confirmation rows before generating any data-dependent threshold or rule.
    development = panel[panel.split.isin(["discovery", "selection"])].copy()
    groups = json.loads((SOURCE / "groups.json").read_text())
    records, coverage, catalogs = [], [], {}
    for checkpoint in sorted(development.checkpoint.unique()):
        current = development[development.checkpoint.eq(checkpoint) & ~development.already_hit].copy()
        train_all = current[current.split.eq("discovery")]
        select_all = current[current.split.eq("selection")]
        rules = atom_rules(train_all)
        catalogs[checkpoint] = rules
        dm_all = np.column_stack([mask_rule(train_all, r) for r in rules])
        sm_all = np.column_stack([mask_rule(select_all, r) for r in rules])
        train_members = train_all.group_ids.tolist()
        select_members = select_all.group_ids.tolist()
        count = 0
        for group in groups:
            gid = group["group_id"]
            ti = np.array([gid in x for x in train_members], dtype=bool)
            si = np.array([gid in x for x in select_members], dtype=bool)
            train, select = train_all.loc[ti], select_all.loc[si]
            cv = {"group_id": gid, "checkpoint": checkpoint, "discovery_n": len(train), "selection_n": len(select),
                  "discovery_dates": train.date.nunique(), "selection_dates": select.date.nunique()}
            if len(train) < 40 or train.date.nunique() < 15:
                cv["status"] = "insufficient_discovery_history"
                cv["rules_examined"] = len(rules)
                cv["supported_rules"] = 0
            else:
                found = search_group(train, select, rules, dm_all[ti], sm_all[si], group, checkpoint)
                records.extend(found)
                count += len(found)
                cv["status"] = "searched"
                cv["rules_examined"] = len(rules)
                cv["supported_rules"] = len(found)
            coverage.append(cv)
        print(f"{checkpoint}: {len(rules)} atoms, {count} supported rules/combinations across 58 groups", flush=True)
    def eligible(item):
        s = item["selection"]
        return s["n"] >= 20 and s["dates"] >= 8 and s["lift"] is not None and s["lift"] > 0
    def order(item):
        return (item["selection"]["lcb"], item["selection"]["lift"], -item["terms"], item["discovery"]["lcb"])
    by_group, winners = {}, []
    for group in groups:
        gid = group["group_id"]
        choices = sorted((x for x in records if x["group_id"] == gid and eligible(x)), key=order, reverse=True)
        all_choices = [x for x in records if x["group_id"] == gid]
        if not choices:
            by_group[gid] = {"group": group, "status": "no_supported_positive_lift_rule", "searched_supported_rules": len(all_choices)}
            continue
        winner = dict(choices[0])
        singles = sorted((x for x in choices if x["terms"] == 1 and x["checkpoint"] == winner["checkpoint"]), key=order, reverse=True)
        best_features, used = [], set()
        for s in singles:
            feature = s["rule"][0][0]
            if feature not in used:
                used.add(feature)
                best_features.append(s)
            if len(best_features) == 3:
                break
        winner["top3_single_features"] = best_features
        by_group[gid] = {"group": group, "status": "supported_candidate", "searched_supported_rules": len(all_choices), "winner": winner}
        winners.append(winner)
    top, aliases, selected_sets = [], [], []
    for winner in sorted(winners, key=order, reverse=True):
        g = development[development.split.eq("selection") & development.checkpoint.eq(winner["checkpoint"])
                        & ~development.already_hit & development.group_ids.map(lambda ids: winner["group_id"] in ids)]
        events = set(g.loc[mask_rule(g, winner["rule"]), "sample_id"])
        duplicate = next((i for i, s in enumerate(selected_sets) if len(events & s) / len(events | s) >= .9), None)
        if duplicate is not None:
            aliases.append({"group_id": winner["group_id"], "representative": top[duplicate]["group_id"], "reason": "selection_event_jaccard_ge_0.9"})
            continue
        top.append(winner)
        selected_sets.append(events)
        if len(top) == 5:
            break
    frozen = {"selection_basis": "Development-only; no confirmation labels used for shortlist", "top5": top,
              "all_group_winners": by_group, "near_duplicate_aliases": aliases,
              "feature_panel_sha256": hashlib.sha256((HERE / "features.parquet").read_bytes()).hexdigest(),
              "protocol_sha256": hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest(),
              "search_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "supported_candidates": len(records), "group_coverage": coverage}
    (HERE / "frozen_candidates.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    (HERE / "threshold_catalog.json").write_text(json.dumps(catalogs, ensure_ascii=False, indent=2))
    flat = [{"group_id": r["group_id"], "checkpoint": r["checkpoint"], "terms": r["terms"], "text": r["text"],
             "rule_json": json.dumps(r["rule"]), **{f"{stage}_{k}": v for stage in ("discovery", "selection") for k, v in r[stage].items()}}
            for r in records]
    pd.DataFrame(flat).to_parquet(HERE / "development_search.parquet", index=False, compression="zstd")
    print(json.dumps({"supported_candidates": len(records), "group_winners": len(winners),
                      "top5": [{k: v for k, v in w.items() if k != "top3_single_features"} for w in top]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
