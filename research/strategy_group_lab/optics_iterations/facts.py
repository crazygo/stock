"""Fixed-window, descriptive price paths; never fed back as model labels/features."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .acquire import FOCUS, ROOT
from .prepare import OLD, GROUPS
from research.group_expectation_matrix.build import calendar_grid, prepare_bars, corporate_dates, write_json

def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def facts(output):
    cfg = json.loads((output/"config.json").read_text())
    sessions = json.loads((ROOT/cfg["calendar"]).read_text())["sessions"]
    grid = calendar_grid(sessions)
    last_dates = [s["session_date"] for s in sessions if s["session_date"] <= cfg["source_end"]][-21:]
    rows, price_paths = [], {}
    for sym in FOCUS+["QQQ"]:
        path = output/"market_data/us_5m"/sym/"2026.parquet"
        if not path.exists():
            rows.append({"symbol": sym, "status": "missing"})
            continue
        raw = pd.read_parquet(path)
        aligned, a, valid, _, _, _ = prepare_bars(raw, grid, pd.Timestamp(cfg["as_of"]))
        actions = corporate_dates(pd.read_parquet(output/"market_data/corporate_actions"/f"{sym}.parquet"))
        if any(last_dates[0] < d <= last_dates[-1] for d in actions):
            rows.append({"symbol": sym, "status": "corporate_action_in_fact_window"})
            continue
        daily = []
        for d in last_dates:
            ids = np.flatnonzero(grid.date.to_numpy() == d)
            if not valid[ids].all():
                daily.append({"date": d, "complete": False})
            else:
                z = a[ids]
                daily.append({"date": d, "complete": True, "open": float(z[0, 0]), "high": float(z[:, 1].max()),
                              "low": float(z[:, 2].min()), "close": float(z[-1, 3])})
        if not all(x["complete"] for x in daily):
            rows.append({"symbol": sym, "status": "incomplete_fact_window", "missing_dates": [d["date"] for d in daily if not d["complete"]]})
            price_paths[sym] = daily
            continue
        anchor = daily[0]["close"]
        closes = np.asarray([d["close"] for d in daily])
        highs = np.asarray([d["high"] for d in daily[1:]])
        peak_at = int(np.argmax(highs))
        peak_return, terminal = float(highs.max()/anchor-1), float(closes[-1]/anchor-1)
        giveback = float(closes[-1]/highs.max()-1)
        if peak_return < .03:
            shape = "窄幅或下行"
        elif giveback < -.08 and terminal < .6*peak_return:
            shape = "冲高后回落"
        elif peak_at < 5 and terminal > 0:
            shape = "提前上涨后维持"
        elif peak_at >= 10 and np.mean(np.diff(closes) > 0) >= .55 and terminal > 0:
            shape = "持续推进"
        else:
            shape = "波动上行或分化"
        record = {"symbol": sym, "status": "complete", "anchor_date": last_dates[0], "from_date": last_dates[1],
                  "to_date": last_dates[-1], "anchor_price": anchor, "latest_close": float(closes[-1]),
                  "return_20": terminal, "return_5": float(closes[-1]/closes[-6]-1), "peak_return": peak_return,
                  "peak_date": last_dates[peak_at+1], "giveback_from_peak": giveback,
                  "close_max_drawdown": float(np.min(closes/np.maximum.accumulate(closes)-1)),
                  "up_day_fraction": float(np.mean(np.diff(closes) > 0)), "retrospective_shape": shape,
                  "stage": next((name for gid, (name, ss) in GROUPS.items() if gid not in ["optics:core", "optics:chain"] and sym in ss), "市场参照")}
        rows.append(record)
        price_paths[sym] = [{**d, "normalized_close": d["close"]/anchor*100,
                             "normalized_high": d["high"]/anchor*100} for d in daily]
    old_groups = json.loads((OLD/"groups.json").read_text())
    old_rows = pd.read_parquet(OLD/"rows.parquet")
    groupid = "industry:network_interconnect__optical_communications"
    mask = np.load(OLD/"memberships.npz")["mask"][:, next(i for i, g in enumerate(old_groups) if g["group_id"] == groupid)]
    old_predictions = pd.DataFrame(json.loads((OLD/"results.json").read_text()))
    router = json.loads((OLD/"router.json").read_text())
    bindings = {x["id"]: x for x in json.loads((OLD/"bindings.json").read_text())}
    audit = {"old_optical_members": sorted(old_rows.loc[mask, "symbol"].unique()),
             "missing_from_old_candidate_pool": sorted(set(FOCUS)-set(old_rows.symbol)),
             "present_but_outside_old_optical_group": sorted((set(FOCUS)&set(old_rows.symbol))-set(old_rows.loc[mask, "symbol"])),
             "old_decision_end": "2026-08-25", "new_fact_end": cfg["source_end"],
             "summary_omission": "posthoc_family_example_filter_required_two_symbols",
             "old_optical_model_rows": old_predictions[old_predictions.group_id == groupid].where(pd.notna(old_predictions), None).to_dict("records"),
             "old_optical_router_trades": [t for t in router["trades"] if bindings[t["binding_id"]]["group_id"] == groupid]}
    # DataFrame conversion can leave NaN in sparse float columns; preserve nulls.
    write_json(output/"facts.json", clean({"stocks": rows, "paths": price_paths, "coverage_audit": audit,
               "shape_labels_are_retrospective_only": True, "fact_source": "NONE 5m RTH complete official sessions"}))
    pd.DataFrame(rows).to_csv(output/"facts.csv", index=False)
    print(pd.DataFrame(rows)[["symbol", "status", "return_20", "return_5", "peak_return", "giveback_from_peak", "retrospective_shape"]].to_string(index=False), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    facts(p.parse_args().output)
