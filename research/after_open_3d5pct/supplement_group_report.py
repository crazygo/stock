"""Additive group report: frozen models, frozen original predictions, new securities.

No fitting, networking, mutation of old artifacts, or order interfaces.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from . import v6_data as vd
from .train_multiscale_v6 import CModel, _predict_c, _torch_arrays, project_monotone, tabular
from research.group_expectation_matrix.build import classify_history, corporate_dates

ROOT = vd.ROOT
NAMES = {"B_lgbm_group": "B有群", "B_lgbm_no_group": "B无群",
         "C_patch_group": "C有群", "C_patch_no_group": "C无群",
         "C_patch_no_daily": "C无日级"}
TARGETS = [{"id": f"{h}_{t}", "days": d, "gain": g, "name": f"{d}日内 +{g}%"}
           for h, d in enumerate((1, 3, 5)) for t, g in enumerate((3, 5, 8))]
SOURCES = {
    "CRDO": "https://credosemi.com/products/optical-dsp/",
    "COHR": "https://www.coherent.com/communications/datacom/datacenter",
    "VRT": "https://www.vertiv.com/en-us/about/news-and-events/corporate-news/vertiv-expects-powering-up-for-ai-digital-twins-and-adaptive-liquid-cooling-to-shape-data-center-design-and-operations/",
    "SOXL": "https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs",
    "SOXS": "https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs",
}
EXTRA_TAGS = {
    "CRDO": ["industry:compute_connectivity_chips", "industry:compute_connectivity_chips__connectivity_custom_silicon",
             "industry:network_interconnect", "industry:network_interconnect__interconnect_chips"],
    "COHR": ["industry:network_interconnect", "industry:network_interconnect__optical_communications"],
    "VRT": ["industry:datacenter_power_cooling"],
    "SOXL": ["instrument:leveraged_semiconductor", "instrument:semiconductor_long3x"],
    "SOXS": ["instrument:leveraged_semiconductor", "instrument:semiconductor_inverse3x"],
}


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def week(date):
    iso = pd.Timestamp(date).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def metrics(y, p):
    """Unknown outcomes excluded, with denominators preserved by the caller."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    n = len(y)
    if not n:
        return {"n": 0, "hit_n": 0, "accuracy50": None, "precision50": None,
                "recall50": None, "brier": None, "auc": None, "mean_p": None,
                "hit_rate": None, "positive_n": 0, "tn": 0, "fp": 0,
                "fn": 0, "tp": 0, "always_no_accuracy": None}
    hat, actual = p >= .5, y == 1
    tp = int(np.sum(hat & actual)); fp = int(np.sum(hat & ~actual))
    fn = int(np.sum(~hat & actual)); tn = int(np.sum(~hat & ~actual))
    return {"n": n, "hit_n": int(y.sum()), "accuracy50": (tp+tn)/n,
            "precision50": tp/(tp+fp) if tp+fp else None,
            "recall50": tp/(tp+fn) if tp+fn else None,
            "brier": float(np.mean((y-p)**2)),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "mean_p": float(p.mean()), "hit_rate": float(y.mean()),
            "positive_n": tp+fp, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "always_no_accuracy": float(1-y.mean())}


def rank_cells(cells, metric="accuracy50", k=3):
    eligible = [c for c in cells if c["group_id"] != "all:all" and c["metrics"].get(metric) is not None]
    sign = 1 if metric == "brier" else -1
    return sorted(eligible, key=lambda c: (sign*c["metrics"][metric], -c["metrics"]["n"], c["group_id"]))[:k]


def window_features(symbol, d, dates, sessions, views, own, peers):
    """Only source prefixes; feature construction never takes an outcome."""
    v = views[symbol]
    if not np.isfinite(v.cutoff_return[d]) or not vd.inputs_available(v, sessions[d], d, 30):
        return None
    x5 = np.zeros((8, vd.SLOTS, 14), np.float32)
    lo = max(0, d-7); x5[7-(d-lo):7] = v.seq5[lo:d]; x5[7, :90] = v.seq5[d, :90]
    x60 = np.zeros((31, len(vd.HOUR_BINS), 14), np.float32)
    lo = max(0, d-30); x60[30-(d-lo):30] = v.seq60[lo:d]
    for j, (_, b, _) in enumerate(vd.HOUR_BINS):
        if b <= 90:
            x60[30, j] = v.seq60[d, j]
    xday = np.zeros((126, 14), np.float32)
    lo = max(0, d-126); xday[126-(d-lo):] = v.seqday[lo:d]
    representative = {week(dates[d]): d}
    for old in range(d-1, -1, -1):
        representative.setdefault(week(dates[old]), old)
        if len(representative) >= 6:
            break
    group_seq = np.zeros((6, 6, 10), np.float32)
    for j, old in enumerate(sorted(representative.values())[-6:]):
        group_seq[6-len(representative)+j] = vd._group_state(symbol, old, dates, sessions, views, own, peers)
    return {"x5": x5, "x60": x60, "xday": xday, "group_seq": group_seq}


def infer(name, run, data):
    """Load only a saved checkpoint, never fit or change calibration."""
    if name.startswith("B_"):
        x = tabular(data, group=name == "B_lgbm_group")
        expected = json.loads((run/"models"/f"{name}_columns.json").read_text())
        if expected != x.columns.tolist():
            raise ValueError("frozen B schema mismatch")
        p = np.stack([lgb.Booster(model_file=str(run/"models"/f"{name}_{j}.txt")).predict(x, num_threads=1)
                      for j in range(9)], axis=-1)
    else:
        cp = torch.load(run/"models"/f"{name}.pt", map_location="cpu", weights_only=False)
        model = CModel(group=name != "C_patch_no_group", daily=name != "C_patch_no_daily",
                       input_channels=14, type_identity="member_weight" in cp["state_dict"])
        model.load_state_dict(cp["state_dict"])
        p = _predict_c(model, _torch_arrays(data), np.arange(len(data["x5"])))
    return project_monotone(p)


def add_memberships(extra, views, dates, sessions, matrix, own, report_members, actions):
    """Classify new symbols; leave the original peer membership sets untouched."""
    versions = json.loads((matrix/"versions.json").read_text())
    strategies = {s["strategy_id"]: s for s in json.loads((matrix/"strategies.json").read_text())}
    added = []
    for symbol in extra:
        v = views.get(symbol)
        if v is None:
            continue
        daily = []
        for d, s in enumerate(sessions):
            n = int(s["duration_minutes"])//5
            block = v.raw5[d, 66:66+n]
            avail = v.effective_available[d, 66:66+n]
            daily.append({"date": dates[d], "complete": bool(v.daily_full[d]),
                          "close": float(block[-1, 3]) if v.daily_full[d] else np.nan,
                          "dollar_volume": float(block[:, 5].sum()) if v.daily_full[d] else np.nan,
                          "available": pd.Timestamp(avail.max()).tz_localize("UTC") if v.daily_full[d] else pd.NaT})
        daily = pd.DataFrame(daily).set_index("date")
        for ver in versions:
            sid, wk = ver["strategy_id"], ver["week_id"]
            if sid not in vd.STRATEGIES:
                continue
            prior = [x for x in dates if x < ver["first_session"]]
            label, status, facts = classify_history(daily, prior, pd.Timestamp(ver["feature_cutoff_at"]),
                                                     strategies[sid], actions[symbol])
            gids = [sid+":"+label] if label else []
            record = {"symbol": symbol, "week_id": wk, "strategy_id": sid, "group_ids": gids,
                      "status": status, "facts": facts, "basis": "causal_reconstruction_new_symbol_frozen_rules",
                      "generated_at": datetime.now(timezone.utc).isoformat()}
            added.append(record)
            # Old ETF memberships may contain unavailable classifications: replace only this new symbol's slice.
            report_members[(wk, symbol)] = {g for g in report_members[(wk, symbol)] if not g.startswith(sid+":")}
            report_members[(wk, symbol)].update(gids)
            if label:
                own[(wk, sid, symbol)] = (gids[0], facts, ver)
    return added


def group_catalog(matrix):
    groups = json.loads((matrix/"groups.json").read_text())
    for gid, name, family in [
        ("industry:datacenter_power_cooling", "数据中心供电与散热", "industry"),
        ("instrument:leveraged_semiconductor", "半导体杠杆ETF", "instrument"),
        ("instrument:semiconductor_long3x", "半导体杠杆ETF · 多头3倍", "instrument"),
        ("instrument:semiconductor_inverse3x", "半导体杠杆ETF · 反向3倍", "instrument"),
        ("watchlist:user_added", "本次补充关注的五只证券", "watchlist")]:
        groups.append({"group_id": gid, "name": name, "strategy_id": family,
                       "classification_basis": "2026-09-27_current_snapshot_report_only"})
    return groups


def build(config_path, output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    config = json.loads(config_path.read_text())
    base_cfg = json.loads((ROOT/"research/after_open_3d5pct/configs/multiscale_groups_v61.json").read_text())
    matrix = ROOT/base_cfg["group_run"]
    original = {n: pd.read_parquet(ROOT/p/f"predictions_{n}.parquet")
                for n, p in config["five_model_sources"].items()}
    ref = original["C_patch_group"]
    original_symbols = sorted(ref.symbol.unique())
    eval_dates = sorted(ref.session_date.unique())
    for n, f in original.items():
        assert f.sample_id.equals(ref.sample_id) and f.y_1_1.equals(ref.y_1_1)
    paths = [ROOT/p/f"predictions_{n}.parquet" for n,p in config["five_model_sources"].items()]
    for n,p in config["five_model_sources"].items():
        paths.extend((ROOT/p/"models").glob(f"{n}*"))
    frozen_hashes = {str(p.relative_to(ROOT)): vd.digest(p) for p in paths}
    sessions = vd._calendar(base_cfg); dates = [s["session_date"] for s in sessions]
    candidates = sorted(m["symbol"] for m in json.loads((ROOT/base_cfg["universe"]).read_text())["members"]
                        if m["role"] == "candidate")
    extra = config["additional_symbols"]
    views, actions, source_hashes, coverage = {}, {}, {}, []
    for symbol in sorted(set(candidates+extra+["QQQ"])):
        v = vd.load_symbol(symbol, sessions, base_cfg)
        action_path = ROOT/f"market_data/corporate_actions/{symbol}.parquet"
        actions[symbol] = corporate_dates(pd.read_parquet(action_path)) if action_path.exists() else []
        if v is not None:
            views[symbol] = v; source_hashes[v.source] = vd.digest(ROOT/v.source)
        if action_path.exists():
            source_hashes[str(action_path.relative_to(ROOT))] = vd.digest(action_path)
        if symbol in extra:
            coverage.append({"symbol": symbol, "bars_available": v is not None,
                             "actions_available": action_path.exists(),
                             "capital_action_dates_2026": [a for a in actions[symbol] if a.startswith("2026")],
                             "complete_daily_sessions": int(v.daily_full.sum()) if v else 0,
                             "source_url": SOURCES[symbol], "classification_basis": "current_snapshot_report_only"})
        if len(views) % 20 == 0:
            print(f"loaded {len(views)} source series", flush=True)
    own, peers = vd._group_map(base_cfg, dates, candidates)
    report_members = defaultdict(set)
    for m in json.loads((matrix/"memberships.json").read_text()):
        report_members[(m["week_id"], m["symbol"])].update(m["group_ids"])
    added = add_memberships(extra, views, dates, sessions, matrix, own, report_members, actions)
    for wk in {week(d) for d in dates}:
        for s in extra:
            report_members[(wk, s)].update(EXTRA_TAGS[s]+["all:all", "watchlist:user_added"])
    groups = group_catalog(matrix)
    rows, tensors, exclusions = [], defaultdict(list), []
    resets = {}
    for symbol in original_symbols+extra:
        sample_dates = eval_dates+[config["snapshot_date"]] if symbol in extra else [config["snapshot_date"]]
        for date in sample_dates:
            d = dates.index(date)
            if symbol not in views or (symbol in extra and not (ROOT/f"market_data/corporate_actions/{symbol}.parquet").exists()):
                exclusions.append({"symbol":symbol,"date":date,"reason":"missing_market_or_action_source"}); continue
            raw_view = views[symbol]
            applicable = [a for a in actions[symbol] if dates[max(0,d-146)] <= a <= date]
            active_reset = None
            if applicable:
                if symbol not in extra:
                    exclusions.append({"symbol":symbol,"date":date,"reason":"capital_action_in_frozen_lookback"}); continue
                active_reset = max(applicable)
                key = (symbol,active_reset)
                if key not in resets:
                    destination = output/"split_reset_sources"/active_reset/symbol/"2026.parquet"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    frame = pd.read_parquet(ROOT/raw_view.source)
                    frame[frame.session_date >= active_reset].to_parquet(destination, compression="zstd", compression_level=7)
                    reset_cfg = dict(base_cfg,source_dir=str(destination.parents[1].relative_to(ROOT)))
                    resets[key] = vd.load_symbol(symbol,sessions,reset_cfg)
                views[symbol] = resets[key]
            features = window_features(symbol,d,dates,sessions,views,own,peers)
            feature_view = views[symbol]
            views[symbol] = raw_view
            if features is None:
                exclusions.append({"symbol":symbol,"date":date,"reason":"unavailable_input_prefix"}); continue
            for key,value in features.items():
                tensors[key].append(value)
            outcome = vd._outcomes(raw_view,sessions,d,base_cfg,set(actions[symbol])) if date in eval_dates else None
            row = {"sample_id":f"{symbol}:{date}:11:30:group_supplement_v1", "symbol":symbol,
                   "session_date":date,"scope":"evaluation" if date in eval_dates else "snapshot",
                   "cutoff_at":(pd.Timestamp(sessions[d]["open_at"])+pd.Timedelta(hours=2)).isoformat(),
                   "history_daily_n":int(feature_view.daily_full[max(0,d-126):d].sum()),
                   "split_reset_at":active_reset,"label_status":"complete" if outcome else "unscored",
                   "group_ids":sorted(report_members[(week(date),symbol)]),
                   "label_end_at":outcome[2] if outcome else None,
                   "label_available_at":outcome[3] if outcome else None}
            for h in range(3):
                for t in range(3):
                    row[f"y_{h}_{t}"] = float(outcome[0][h,t]) if outcome else None
            rows.append(row)
    data = {k:np.stack(v) for k,v in tensors.items()}
    np.savez_compressed(output/"supplement_features.npz",**data)
    all_frames = {}
    for name,run in config["five_model_sources"].items():
        p = infer(name,ROOT/run,data)
        new = pd.DataFrame(rows)
        for h in range(3):
            for t in range(3): new[f"p_{h}_{t}"] = p[:,h,t]
        old = original[name].copy()
        old["scope"] = "evaluation"; old["label_status"] = "complete"
        old["group_ids"] = [sorted(report_members[(week(r.session_date),r.symbol)]) for r in old.itertuples()]
        combined = pd.concat([old,new],ignore_index=True)
        assert not combined.duplicated(["symbol","session_date"]).any()
        assert np.array_equal(combined.iloc[:len(old)][[f"p_{h}_{t}" for h in range(3) for t in range(3)]].values,
                              old[[f"p_{h}_{t}" for h in range(3) for t in range(3)]].values)
        combined.to_parquet(output/f"predictions_{name}.parquet",index=False)
        all_frames[name] = combined
        print(f"{name}: {len(new)} new inference rows; {len(old)} original rows reused",flush=True)
    cells = []
    for name,frame in all_frames.items():
        hist = frame[frame.scope=="evaluation"]
        latest = frame[frame.scope=="snapshot"]
        for g in groups:
            gid = g["group_id"]
            selected = hist[hist.group_ids.map(lambda ids: gid in ids)]
            now = latest[latest.group_ids.map(lambda ids: gid in ids)]
            for target in TARGETS:
                k = target["id"]; y = selected[f"y_{k}"].to_numpy(float); p = selected[f"p_{k}"].to_numpy(float)
                valid = np.isfinite(y) & np.isfinite(p)
                scored = selected.loc[valid]
                cells.append({"model":name,"group_id":gid,"target":k,"metrics":metrics(y,p),
                              "unscored_n":int((~valid).sum()),"symbol_n":int(scored.symbol.nunique()),
                              "date_n":int(scored.session_date.nunique()),"symbols":sorted(scored.symbol.unique()),
                              "snapshot_mean_p":float(now[f"p_{k}"].mean()) if len(now) else None,
                              "snapshot_n":len(now),"snapshot_symbols":sorted(now.symbol.unique()),
                              "member_fingerprint":hashlib.sha256('|'.join(sorted(selected.sample_id)).encode()).hexdigest()[:16],
                              "current_membership_retrospective":gid.startswith(("industry:","watchlist:","instrument:"))})
    top3 = []
    for model in NAMES:
        for t in TARGETS:
            ranked = rank_cells([c for c in cells if c["model"]==model and c["target"]==t["id"]],config["ranking_metric"])
            ids = [c["group_id"] for c in ranked]
            current = all_frames[model]; current = current[current.scope=="snapshot"]
            chosen = current[current.group_ids.map(lambda gs:bool(set(gs)&set(ids)))]
            top3.append({"model":model,"target":t["id"],"ranking_metric":config["ranking_metric"],
                         "groups":ids,"deduplicated_symbols":sorted(chosen.symbol.unique()),
                         "selection_status":"post_selection_exposed_history_no_forward_validation"})
    records = []
    for model,f in all_frames.items():
        keep=["symbol","session_date","scope","group_ids","label_status"]+[f"{prefix}_{h}_{t}" for prefix in ("p","y") for h in range(3) for t in range(3)]
        # JSON encoder turns NaN into null here, rather than reporting absent labels as zero.
        r=json.loads(f[keep].to_json(orient="records"))
        records.extend(dict(x,model=model) for x in r)
    for item in coverage:
        s=item["symbol"]; frame=all_frames["C_patch_group"]
        item["evaluation_prediction_n"]=int(((frame.symbol==s)&(frame.scope=="evaluation")).sum())
        item["snapshot_prediction_n"]=int(((frame.symbol==s)&(frame.scope=="snapshot")).sum())
    assert all(vd.digest(ROOT/p)==sha for p,sha in frozen_hashes.items()), "old artifact changed"
    payload={"protocol":config["protocol"],"generated_at":datetime.now(timezone.utc).isoformat(),
             "evaluation_dates":eval_dates,"snapshot_date":config["snapshot_date"],"ranking_metric":config["ranking_metric"],
             "models":NAMES,"targets":TARGETS,"groups":groups,"cells":cells,"top3":top3,
             "predictions":records,"coverage":coverage,"exclusions":exclusions,
             "sources":SOURCES,"old_predictions_preserved":True,"old_models_preserved":True,
             "trained_models":0,"availability_quality":"archived_bar_end_plus_1s_assumed",
             "membership_quality":"causal_price_groups_current_universe_and_retrospective_industry",
             "scope":"offline_supplement_exposed_history_not_live_or_independent_validation"}
    write_json(output/"report.json",payload)
    write_json(output/"additional_memberships.json",added)
    write_json(output/"manifest.json",{"config":config,"config_sha256":vd.digest(config_path),
               "code_sha256":vd.digest(Path(__file__)),"source_hashes":source_hashes,
               "frozen_artifact_hashes":frozen_hashes,"groups_source_sha256":vd.digest(matrix/"memberships.json"),
               "original_prediction_rows_per_model":len(ref),"original_symbols":original_symbols,
               "additional_symbols":extra,"new_inference_rows_per_model":len(rows)})
    print(json.dumps({"output":str(output),"coverage":coverage,"group_count":len(groups)},ensure_ascii=False),flush=True)
    return payload


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        build(a.config,a.output)
