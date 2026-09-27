"""One-time outer reveal after R1/R2/R3 completion; saved-model replay and paired dates."""
from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.after_open_3d5pct.train_multiscale_v6 import CModel,_torch_arrays,_predict_c,project_monotone,violations
from .v7_c_group import (BASE,BASELINE,DATASET,CONFIG,TARGET_NAMES,ShapeCModel,
                         ContrastCModel,buckets,measure,read_data,save_predictions,sha)


def metrics(frame):
    result={}
    for name in TARGET_NAMES:
        y=frame[f"y_{name}"].to_numpy();raw=frame[f"raw_{name}"].to_numpy();p=frame[f"p_{name}"].to_numpy()
        rel=buckets(y,p)
        result[name]={"raw":measure(y,raw),"final":measure(y,p),"reliability10":rel,
                      "ece10":sum(v["n"]*abs(v["mean_p"]-v["rate"]) for v in rel if v["n"])/len(y)}
    return result


def block_ci(frame, ref, name, seed=62027, draws=1000, length=5):
    """Moving contiguous ET-date block bootstrap of paired Brier difference."""
    assert frame.sample_id.equals(ref.sample_id)
    dates=sorted(frame.session_date.unique())
    if len(dates)<length: return None
    by_date={d:np.flatnonzero(frame.session_date.to_numpy()==d) for d in dates}
    y=frame[f"y_{name}"].to_numpy();p=frame[f"p_{name}"].to_numpy();q=ref[f"p_{name}"].to_numpy()
    error=(y-p)**2-(y-q)**2
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(draws):
        chosen=[]
        while len(chosen)<len(dates):
            start=int(rng.integers(0,len(dates)-length+1))
            chosen.extend(dates[start:start+length])
        ix=np.concatenate([by_date[d] for d in chosen[:len(dates)]])
        samples.append(float(error[ix].mean()))
    return {"delta_brier":float(error.mean()),"ci95":[float(v) for v in np.quantile(samples,[.025,.975])],
            "unit":"ET date", "block_dates":length,"draws":draws,"dates":len(dates)}


def main():
    out=BASE/"runs/v7_c_group_final_20260927"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    table,data,folds,config=read_data()
    torch.set_num_threads(1)
    arrays=_torch_arrays(data)
    trials={
        "R0":{"path":BASELINE/"models/C_patch_group.pt","cls":CModel,"run":BASELINE},
        "R1":{"path":BASE/"runs/v7_c_group_R1_seed3566/model.pt","cls":CModel,"run":BASE/"runs/v7_c_group_R1_seed3566"},
        "R2":{"path":BASE/"runs/v7_c_group_R2_corrected_seed3566/model.pt","cls":ShapeCModel,"run":BASE/"runs/v7_c_group_R2_corrected_seed3566"},
        "R3_seed3566":{"path":BASE/"runs/v7_c_group_R3_seed3566/model.pt","cls":ContrastCModel,"run":BASE/"runs/v7_c_group_R3_seed3566"},
        "R3_seed81173":{"path":BASE/"runs/v7_c_group_R3_seed81173/model.pt","cls":ContrastCModel,"run":BASE/"runs/v7_c_group_R3_seed81173"},
    }
    results={"protocol":"v7_c_group_exposed_development","dataset_manifest_sha256":sha(DATASET/"manifest.json"),
             "features_sha256":sha(DATASET/"features.npz"),"rows_sha256":sha(DATASET/"rows.parquet"),
             "base_config_sha256":sha(CONFIG),"final_eval_code_sha256":sha(Path(__file__)),
             "model_code_sha256_at_final_replay":sha(BASE/"iterations_v7/c_group/v7_c_group.py"),
             "python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,
             "splits":{k:{"rows":len(v),"dates":int(table.iloc[v].session_date.nunique()),
                            "min_date":str(table.iloc[v].session_date.min()),"max_date":str(table.iloc[v].session_date.max()),
                            "full126":int(table.iloc[v].full_126_prior_days.sum()),
                            "median_daily_available":float(table.iloc[v].available_daily_days.median())}
                        for k,v in folds.items()},
             "calibration":"none; final is deterministic nine-target monotonic projection, not learned calibration",
             "invalid_attempt":"runs/v7_c_group_R2_seed3566/INVALID.json",
             "trials":{},"paired_ci_vs_R0":{}}
    frames={}
    for label,info in trials.items():
        started=time.monotonic()
        checkpoint=torch.load(info["path"],map_location="cpu",weights_only=False)
        if label!="R0" and checkpoint["dataset_manifest_sha256"]!=results["dataset_manifest_sha256"]:
            raise ValueError(f"{label}: checkpoint input mismatch")
        model=(CModel(group=True,daily=True,input_channels=14,type_identity=True)
               if info["cls"]==CModel else info["cls"]())
        model.load_state_dict(checkpoint["state_dict"])
        d={"checkpoint_sha256":sha(info["path"]),"parameters":sum(p.numel() for p in model.parameters()),
           "seed":3566 if label in ("R0","R1","R2","R3_seed3566") else 81173,
           "splits":{}}
        if label=="R0":
            base_report=json.loads((BASELINE/"report.json").read_text())["trials"]["C_patch_group"]
            d["fit_seconds"]=base_report["fit"]["seconds"];d["best_epoch"]=base_report["fit"]["best_epoch"]
            d["curve"]=base_report["fit"]["epochs"]
        else:
            train_report=json.loads((info["run"]/"results.json").read_text())
            d["fit_seconds"]=train_report["seconds"];d["best_epoch"]=train_report["best_epoch"]
            d["curve"]=train_report["curve"]
            d["training_source_sha256"]=train_report["source_sha256"]
        for split in ("train","inner","outer"):
            ids=folds[split];raw=_predict_c(model,arrays,ids)
            target=(out/f"{label}_{split}_predictions.parquet" if label=="R0" else info["run"]/f"{split}_predictions.parquet")
            if target.exists():
                previous=pd.read_parquet(target)
                replay=save_predictions(out/f"replay_{label}_{split}.parquet",table,ids,data["y"][ids],raw)
                cols=[f"p_{name}" for name in TARGET_NAMES]
                if not replay.sample_id.equals(previous.sample_id): raise ValueError(f"{label}/{split}: id mismatch")
                difference=float(np.max(np.abs(replay[cols].to_numpy()-previous[cols].to_numpy())))
                if difference>1e-6: raise ValueError(f"{label}/{split}: replay mismatch {difference}")
                (out/f"replay_{label}_{split}.parquet").unlink()
                frame=previous
            else:
                frame=save_predictions(target,table,ids,data["y"][ids],raw)
                difference=0.0
            if label=="R0" and split=="outer":
                historical=pd.read_parquet(BASELINE/"predictions_C_patch_group.parquet")
                assert frame.sample_id.equals(historical.sample_id)
                max_historical=max(float(np.max(np.abs(frame[f"p_{TARGET_NAMES[j]}"].to_numpy()-historical[f"p_{j//3}_{j%3}"].to_numpy()))) for j in range(9))
                if max_historical>1e-6: raise ValueError(f"R0 vs frozen outer {max_historical}")
                d["frozen_outer_max_abs_delta"]=max_historical
            frames[(label,split)]=frame
            d["splits"][split]={"targets":metrics(frame),"prediction_sha256":sha(target),
                                 "replay_max_abs_delta":difference,
                                 "raw_order_violation_fraction":violations(raw)}
        d["replay_seconds"]=time.monotonic()-started
        results["trials"][label]=d
    for split in ("inner","outer"):
        ref=frames[("R0",split)]
        results["paired_ci_vs_R0"][split]={}
        for label in trials:
            if label=="R0": continue
            frame=frames[(label,split)]
            results["paired_ci_vs_R0"][split][label]={name:block_ci(frame,ref,name,seed=62027+j)
                                                       for j,name in enumerate(TARGET_NAMES)}
    (out/"results.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    print(json.dumps({label:{"inner":results["trials"][label]["splits"]["inner"]["targets"]["3d_5pct"]["final"]["brier"],
                             "outer":results["trials"][label]["splits"]["outer"]["targets"]["3d_5pct"]["final"]["brier"]}
                      for label in trials},indent=2))


if __name__=="__main__": main()
