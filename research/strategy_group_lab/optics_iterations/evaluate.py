"""Exhaustive registered groups, paired comparisons, and observed-price accounts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from research.strategy_group_lab.evaluate import metrics
from research.group_expectation_matrix.build import corporate_dates, write_json
from .train import temporal_split, stable_key
from .portfolio import simulate
from .prepare import OLD, GROUPS
from .acquire import ROOT, FOCUS
from .facts import clean

WINDOWS = {"common": ("2026-07-01", "2026-08-25"), "recent": ("2026-09-01", "2026-09-25")}

def paired(frame, datum, days, cfg):
    f = frame[frame.status.eq("mature")][["row_id","date","hit","p"]].merge(datum[["row_id","p"]],on="row_id",suffixes=("","_datum"),validate="one_to_one")
    if f.empty:
        return {"n": 0, "brier_gain": None, "ci95": None, "p_value": None}
    f["gain"] = (f.p_datum-f.hit)**2-(f.p-f.hit)**2
    result = {"n":len(f),"date_n":int(f.date.nunique()),"brier_gain":float(f.gain.mean()),"ci95":None,"p_value":None}
    daily = f.groupby("date").agg(total=("gain","sum"),n=("gain","size"))
    block = max(5, days)
    if len(daily) < block*4:
        return {**result,"status":"insufficient_independent_date_blocks"}
    a,n = daily.total.to_numpy(),daily.n.to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    starts = rng.integers(0,len(daily)-block+1,size=(1000,int(np.ceil(len(daily)/block))))
    ix = (starts[:,:,None]+np.arange(block)).reshape(1000,-1)[:,:len(daily)]
    result["ci95"] = np.quantile(a[ix].sum(1)/n[ix].sum(1),[.025,.975]).tolist()
    aa = np.array([a[i:i+block].sum() for i in range(0,len(a),block)])
    nn = np.array([n[i:i+block].sum() for i in range(0,len(n),block)])
    k = len(aa)
    se = np.sqrt(k/(k-1)*np.sum((aa-result["brier_gain"]*nn)**2))/n.sum()
    result.update(p_value=float(student_t.sf(result["brier_gain"]/se,k-1)) if se>1e-14 else 1.,status="development_date_block_approximation")
    return result

def evaluate(output):
    cfg = json.loads((output/"config.json").read_text())
    data = output/"data"
    rows = pd.read_parquet(data/"rows.parquet")
    labels = pd.read_parquet(data/"labels.parquet")
    grid = pd.read_parquet(data/"grid.parquet")
    paths = dict(np.load(data/"paths.npz"))
    mem = np.load(data/"memberships.npz")["mask"]
    groups = json.loads((data/"groups.json").read_text())
    group_ix = {g["group_id"]:i for i,g in enumerate(groups)}
    old_symbols = set(pd.read_parquet(OLD/"rows.parquet").symbol)
    actions = {sym:corporate_dates(pd.read_parquet(ROOT/cfg["actions_dir"]/f"{sym}.parquet")) for sym in paths}
    label_map = {e["id"]:labels[labels.expectation_id.eq(e["id"])].sort_values("row_id").reset_index(drop=True) for e in cfg["expectations"]}
    baseline = {}
    for exp in cfg["expectations"]:
        lab = label_map[exp["id"]]
        for fold in cfg["folds"]:
            fit,cal,_ = temporal_split(rows,lab,fold["outer_start"])
            for g in groups:
                m = mem[:,group_ix[g["group_id"]]]
                ids = np.flatnonzero(cal&m)
                if not len(ids): ids = np.flatnonzero(fit&m)
                baseline[(exp["id"],fold["id"],g["group_id"])] = float(lab.hit.iloc[ids].mean()) if len(ids) else .5
    trial_records,frames = [],{}
    for round_id in ["R0","R1","R2","R3"]:
        trials = json.loads((output/round_id/"trials.json").read_text())
        if any(t["status"] not in ["trained","frozen_replay"] for t in trials):
            raise ValueError(f"{round_id}: incomplete models; inspect trials before evaluation")
        trial_records.extend(trials)
        for t in trials:
            key = (round_id,t["algorithm"],t["granularity"],t["expectation"],t["scope"])
            p = pd.read_parquet(output/round_id/f"{t['model_id']}.parquet").assign(fold=t["fold"])
            frames.setdefault(key,[]).append(p)
    for key,parts in frames.items():
        frames[key] = pd.concat(parts,ignore_index=True).merge(rows,on="row_id",validate="one_to_one").merge(label_map[key[3]],on="row_id",validate="one_to_one")
    results,comparisons,latest = [],[],[]
    details = output/"details"; details.mkdir(exist_ok=True)
    for ki,(key,whole) in enumerate(frames.items()):
        round_id,algo,gran,expectation,scope = key
        exp = next(e for e in cfg["expectations"] if e["id"] == expectation)
        for window,(start,end) in WINDOWS.items():
            dates = sorted(d for d in grid.date.unique() if start<=d<=end)
            for g in groups:
                gid = g["group_id"]
                if (scope == "chain" or window == "recent") and gid not in GROUPS: continue
                selected = mem[whole.row_id.to_numpy(),group_ix[gid]] & whole.date.between(start,end).to_numpy()
                frame = whole[selected].copy()
                frame["b0"] = [baseline[(expectation,f,gid)] for f in frame.fold]
                bid = stable_key(*key,window,gid)
                m = metrics(frame,dates,exp["days"],cfg)
                mature = frame[frame.status.eq("mature")]
                true_n = int(mature.hit.sum())
                m["recall"] = float(mature.loc[mature.p>=cfg["threshold"],"hit"].sum()/true_n) if true_n else None
                signal = frame[["row_id","p"]].assign(binding_id=bid,expectation_id=expectation)
                bt,detail = simulate(signal,rows,paths,grid,cfg,start,end,cfg["source_end"],actions)
                if frame.empty:
                    bt.update(backtest_complete=False,end_balance=None,net_return=None,max_drawdown=None,simulation_status="no_eligible_observations")
                result = {"id":bid,"round":round_id,"algorithm":algo,"granularity":gran,"expectation":expectation,
                          "scope":scope,"window":window,"group_id":gid,"group_name":g["name"],**m,**bt,
                          "available_symbols":sorted(frame.symbol.unique()),"decision_dates":len(dates),
                          "mature_opportunities":true_n,"below_threshold_opportunities":int(mature[(mature.hit==1)&(mature.p<cfg["threshold"])].shape[0])}
                if gid == "optics:chain" and round_id != "R0":
                    datum = frames[("R0",algo,gran,expectation,"pooled")]
                    datum = datum[datum.date.between(start,end)]
                    c = {k:result[k] for k in ["id","round","algorithm","granularity","expectation","scope","window"]}
                    c.update(paired(frame,datum,exp["days"],cfg))
                    common = frame[frame.symbol.isin(old_symbols)]
                    c["old_coverage_only"] = paired(common,datum,exp["days"],cfg)
                    c["newly_covered_only"] = paired(frame[~frame.symbol.isin(old_symbols)],datum,exp["days"],cfg)
                    comparisons.append(c)
                    result["vs_R0"] = c
                if gid in GROUPS or gid == "all:all":
                    detail["reliability"] = m.get("reliability",[])
                    write_json(details/f"{bid}.json",clean(detail))
                results.append(result)
        f = whole[(whole.date == "2026-09-25") & whole.symbol.isin(FOCUS)]
        latest.extend([{"round":round_id,"algorithm":algo,"granularity":gran,"expectation":expectation,"scope":scope,**r}
                       for r in f[["symbol","date","cutoff_at","entry_at","p","raw_p","status"]].to_dict("records")])
        print(f"evaluate {ki+1}/{len(frames)} {key}",flush=True)
    # Prespecified comparison family: all rounds/routes/scopes, whole chain, common window.
    eligible = sorted([c for c in comparisons if c["window"]=="common" and c["p_value"] is not None],key=lambda c:c["p_value"])
    last = 0.
    for i,c in enumerate(eligible):
        last = max(last,min(1.,c["p_value"]*(len(eligible)-i)))
        c["holm_p"] = last
        c["holm_pass"] = last<.05 and c["brier_gain"]>0
    curves = [t for t in trial_records if t["round"]=="R3"]
    training = {}
    for algo in cfg["algorithms"]:
        a = [t["learning_curve"] for t in curves if t["algorithm"]==algo]
        budgets = [t["budget"] for t in a]
        original = cfg["lgbm"]["n_estimators"] if algo=="lgbm" else cfg["tcn"]["epochs"]
        training[algo] = {"model_n":len(a),"min":min(budgets),"median":float(np.median(budgets)),"max":max(budgets),
                          "above_original":sum(b>original for b in budgets),"below_original":sum(b<original for b in budgets),
                          "original_budget":original,"selected_n":sum(x["status"]=="inner_temporal_selection" for x in a)}
    summary = {"row_n":len(rows),"available_symbol_n":int(rows.symbol.nunique()),"group_n":len(groups),
               "trial_n":len(trial_records),"new_models":sum(t["status"]=="trained" for t in trial_records),
               "result_n":len(results),"paired_holm_tests":len(eligible),"paired_holm_pass":sum(c.get("holm_pass",False) for c in eligible),
               "training_budgets":training,"windows":WINDOWS,"as_of":cfg["as_of"],"status":"development_replay_not_independent_validation"}
    for filename,value in [("results.json",results),("comparisons.json",comparisons),("latest.json",latest),
                           ("learning_curves.json",curves),("summary.json",summary)]:
        write_json(output/filename,clean(value))
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in results]).to_csv(output/"results.csv",index=False)
    pd.DataFrame(latest).to_csv(output/"latest.csv",index=False)
    pd.concat([f.assign(round=k[0],algorithm=k[1],granularity=k[2],expectation=k[3],scope=k[4]) for k,f in frames.items()],ignore_index=True).to_parquet(output/"predictions.parquet",index=False)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("output",type=Path)
    evaluate(p.parse_args().output.resolve())
