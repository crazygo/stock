"""Validate saved matrix identities, aggregation, time boundaries and provenance."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build import ROOT, file_hash, write_json, validate_refinement


def audit(run):
    manifest = json.loads((run/"manifest.json").read_text())
    cells = json.loads((run/"matrix.json").read_text())
    groups = json.loads((run/"groups.json").read_text())
    versions = json.loads((run/"versions.json").read_text())
    members = json.loads((run/"memberships.json").read_text())
    exps = json.loads((run/"expectations.json").read_text())
    out = pd.read_parquet(run/"outcomes.parquet")
    checks = []
    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
    check("source_hashes", all(file_hash(ROOT/p["path"]) == p["sha256"] for p in manifest["source_files"]))
    taxonomy = json.loads((run/"industry_tags.json").read_text())
    if "refinement" in taxonomy:
        try:
            validate_refinement(taxonomy)
            check("flat_industry_refinement", True)
        except (ValueError, KeyError) as error:
            check("flat_industry_refinement", False, str(error))
        check("refined_group_snapshot", {g["group_id"] for g in groups if g["strategy_id"] == "industry"} == {g["group_id"] for g in taxonomy["groups"]})
        if taxonomy.get("refinement_mode") == "parent_entities":
            check("parent_membership_equals_child_union_each_week", all(
                (r["source_group_id"] in m["group_ids"]) == bool(set(r["subgroup_ids"]) & set(m["group_ids"]))
                for m in members if m["strategy_id"] == "industry" for r in taxonomy["refinement"]))
            check("no_duplicate_group_membership", all(len(m["group_ids"]) == len(set(m["group_ids"])) for m in members))
    check("unique_observation_expectation", not out.duplicated(["sample_id", "expectation_id"]).any())
    check("unique_cells", len(cells) == len({(c["scope"],c["group_id"],c["expectation_id"]) for c in cells}))
    check("complete_matrix_shape", len(cells) == len(groups)*len(exps)*(manifest["week_count"]+1))
    check("denominators", all(c["mature_n"]+c["pending_n"]+c["missing_n"] == c["sample_count"] and 0 <= c["hit_n"] <= c["mature_n"] for c in cells))
    check("rates", all(c["mature_n"] == 0 or abs(c["hit_n"]/c["mature_n"]-c["hit_rate"]) < 1e-7 for c in cells))
    check("no_pending_negative_labels", out.loc[out.status != "mature", "hit"].isna().all())
    mature = out[out.status == "mature"]
    asof = pd.Timestamp(manifest["evaluation_as_of"])
    check("mature_window_available", (pd.to_datetime(mature.label_available_at, utc=True) <= asof).all())
    check("mature_end_before_availability", (pd.to_datetime(mature.label_end_at, utc=True) <= pd.to_datetime(mature.label_available_at, utc=True)).all())
    check("entry_after_decision", (pd.to_datetime(out.entry_at, utc=True) > pd.to_datetime(out.decision_at, utc=True)).all())
    vmap = {v["version_id"]:v for v in versions}
    check("membership_version_fk", all(m["version_id"] in vmap and m["strategy_id"] == vmap[m["version_id"]]["strategy_id"] and m["week_id"] == vmap[m["version_id"]]["week_id"] for m in members))
    check("feature_history_precedes_week", all(m["facts"].get("history_end", "") < vmap[m["version_id"]]["first_session"] for m in members))
    check("membership_unique", len(members) == len({(m["week_id"],m["strategy_id"],m["symbol"]) for m in members}))
    check("nonoverlap_within_behavior_strategy", all(len(m["group_ids"]) <= 1 for m in members if m["strategy_id"] != "industry"))
    watchlist_path = run/"watchlist_snapshot.json"
    if watchlist_path.exists():
        watch = json.loads(watchlist_path.read_text())
        etfs = {m["symbol"] for m in watch["members"]}
        instruments = {m["symbol"]: m for m in json.loads((run/"instruments.json").read_text())}
        check("etf_group_members_are_security_entities", all(instruments[s]["instrument_type"] == "etf" and instruments[s]["candidate"] and watch["group_id"] in instruments[s]["watchlist_group_ids"] for s in etfs))
        check("etf_watchlist_membership_each_week", all((watch["group_id"] in m["group_ids"]) == (m["symbol"] in etfs) for m in members if m["strategy_id"] == "watchlist_etf"))
        missing_etfs = {s for s in etfs if instruments[s]["market_data_status"] != "local_bars_and_actions_present"}
        check("missing_etf_prices_are_unknown_labels", all(out.loc[out.symbol == s, "status"].eq("missing").all() and out.loc[out.symbol == s, "hit"].isna().all() for s in missing_etfs))
        check("etf_coverage_manifest", manifest["etf_candidate_count"] == len(etfs) and manifest["etf_market_data_unavailable_count"] == len(missing_etfs))
    for e in exps:
        eid = e["expectation_id"]
        c = next(c for c in cells if c["scope"] == "all" and c["group_id"] == "all:all" and c["expectation_id"] == eid)
        r = mature[mature.expectation_id == eid]
        check(f"all_pool_counts:{eid}", c["mature_n"] == len(r) and c["hit_n"] == int(r.hit.sum()))
        check(f"baseline_identity:{eid}", c["market_comparison"].get("lift", 0) == 0)
    for e1 in exps:
        for e2 in exps:
            if e1 == e2 or e1["trading_days"] > e2["trading_days"] or e1["target_return"] < e2["target_return"]:
                continue
            p = mature[mature.expectation_id.isin([e1["expectation_id"],e2["expectation_id"]])].pivot(index="sample_id", columns="expectation_id", values="hit").dropna()
            if len(p):
                check(f"nested_target_monotonicity:{e1['expectation_id']}->{e2['expectation_id']}", (p[e1["expectation_id"]] <= p[e2["expectation_id"]]).all())
    monthly = json.loads((run/"monthly_metrics.json").read_text())
    stocks = json.loads((run/"stock_metrics.json").read_text())
    for label, rows in [("monthly",monthly),("stock",stocks)]:
        agg = pd.DataFrame(rows).groupby(["group_id","expectation_id"])[["sample_count","mature_n","hit_n","pending_n","missing_n"]].sum()
        ok = True
        for c in [x for x in cells if x["scope"] == "all"]:
            key = (c["group_id"],c["expectation_id"])
            if key in agg.index:
                ok &= all(c[k] == int(agg.loc[key,k]) for k in agg.columns)
            else:
                ok &= c["sample_count"] == 0
        check(f"{label}_aggregation", ok)
    result = {"passed": all(x["passed"] for x in checks), "check_count": len(checks), "checks": checks,
              "artifact_hashes": [{"path": p.name, "sha256": file_hash(p)} for p in sorted(run.glob("*.json")) if p.name != "audit.json"]}
    write_json(run/"audit.json", result)
    print(json.dumps({"passed":result["passed"],"checks":len(checks),"failed":[x for x in checks if not x["passed"]]}, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    audit(p.parse_args().run.resolve())
