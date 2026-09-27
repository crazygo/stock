"""One-time outer scoring after all C-no-group rounds are frozen."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.after_open_3d5pct.train_multiscale_v6 import CModel, violations
from .runner import (BASE, BASELINE, CONFIG, DATASET, TARGETS, arrays_of, buckets,
                     make_model, metrics, predict, read_data, save_predictions, sha)


def target_metrics(frame):
    out={}
    for name in TARGETS:
        y=frame[f"y_{name}"].to_numpy();raw=frame[f"raw_{name}"].to_numpy();p=frame[f"p_{name}"].to_numpy()
        reliability=buckets(y,p)
        out[name]={"raw":metrics(y,raw),"final":metrics(y,p),"reliability10":reliability,
                   "ece10":sum(z["n"]*abs(z["mean_p"]-z["rate"]) for z in reliability if z["n"])/len(y)}
    return out


def block_ci(frame,reference,name,seed=62027,draws=1000,block=5):
    assert frame.sample_id.equals(reference.sample_id)
    dates=sorted(frame.session_date.unique())
    by_date={d:np.flatnonzero(frame.session_date.to_numpy()==d) for d in dates}
    y=frame[f"y_{name}"].to_numpy()
    error=(y-frame[f"p_{name}"].to_numpy())**2-(y-reference[f"p_{name}"].to_numpy())**2
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(draws):
        selected=[]
        while len(selected)<len(dates):
            start=int(rng.integers(0,len(dates)-block+1))
            selected.extend(dates[start:start+block])
        ix=np.concatenate([by_date[d] for d in selected[:len(dates)]])
        samples.append(float(error[ix].mean()))
    return {"delta_brier":float(error.mean()),"ci95":[float(v) for v in np.quantile(samples,[.025,.975])],
            "unit":"ET date","block_dates":block,"draws":draws,"dates":len(dates)}


def main(output: Path, r1_run: Path, r2_run: Path, r3a_run: Path, r3b_run: Path):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    table,data,folds=read_data();torch.set_num_threads(1);arrays=arrays_of(data)
    trials={
        "R0":(BASELINE/"models/C_patch_no_group.pt","R1",BASELINE,3566),
        "R1":(r1_run/"model.pt","R1",r1_run,3566),
        "R2":(r2_run/"model.pt","R2",r2_run,3566),
        "R3_seed3566":(r3a_run/"model.pt","R3",r3a_run,3566),
        "R3_seed81173":(r3b_run/"model.pt","R3",r3b_run,81173),
    }
    result={"protocol":"v7_c_no_group_exposed_development","dataset_manifest_sha256":sha(DATASET/"manifest.json"),
            "features_sha256":sha(DATASET/"features.npz"),"rows_sha256":sha(DATASET/"rows.parquet"),
            "base_config_sha256":sha(CONFIG),"final_eval_code_sha256":sha(Path(__file__)),
            "model_code_sha256_at_final_replay":sha(BASE/"iterations_v7/c_no_group/runner.py"),
            "python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,
            "calibration":"none; final is fixed nine-target monotonic projection, not learned calibration",
            "splits":{k:{"rows":len(v),"dates":int(table.iloc[v].session_date.nunique()),
                          "min_date":str(table.iloc[v].session_date.min()),
                          "max_date":str(table.iloc[v].session_date.max()),
                          "full126":int(table.iloc[v].full_126_prior_days.sum()),
                          "median_daily_available":float(table.iloc[v].available_daily_days.median())}
                      for k,v in folds.items()},"trials":{},"paired_ci_vs_R0":{},"paired_ci_vs_R1":{}}
    frames={}
    for label,(checkpoint_path,model_round,run_dir,seed) in trials.items():
        start=time.monotonic()
        checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
        if label!="R0" and checkpoint["dataset_manifest_sha256"]!=result["dataset_manifest_sha256"]:
            raise ValueError(f"{label} checkpoint dataset changed")
        model=make_model(model_round)
        model.load_state_dict(checkpoint["state_dict"])
        info={"checkpoint_sha256":sha(checkpoint_path),"parameters":sum(p.numel() for p in model.parameters()),
              "seed":seed,"splits":{}}
        if label=="R0":
            fit=json.loads((BASELINE/"report.json").read_text())["trials"]["C_patch_no_group"]["fit"]
            info.update(fit_seconds=fit["seconds"],best_epoch=fit["best_epoch"],curve=fit["epochs"])
        else:
            fit=json.loads((run_dir/"results.json").read_text())
            if sha(run_dir/"source_at_fit.py")!=fit["source_sha256"]:
                raise ValueError(f"{label} source snapshot mismatch")
            info.update(fit_seconds=fit["seconds"],best_epoch=fit["best_epoch"],
                        curve=fit["curve"],training_source_sha256=fit["source_sha256"],
                        task_weights=fit.get("task_weights",[1]*9))
        for split in ("train","inner","outer"):
            ids=folds[split];raw=predict(model,arrays,ids)
            path=(output/f"{label}_{split}_predictions.parquet" if label=="R0" else run_dir/f"{split}_predictions.parquet")
            replay_path=output/f"replay_{label}_{split}.parquet"
            frame=save_predictions(replay_path,table,ids,data["y"][ids],raw)
            if path.exists():
                previous=pd.read_parquet(path)
                cols=[f"p_{name}" for name in TARGETS]
                if not frame.sample_id.equals(previous.sample_id):
                    raise ValueError(f"{label}/{split} sample ID replay mismatch")
                difference=float(np.max(np.abs(frame[cols].to_numpy()-previous[cols].to_numpy())))
                if difference>1e-6:raise ValueError(f"{label}/{split} prediction replay mismatch: {difference}")
                replay_path.unlink();frame=previous
            else:
                replay_path.rename(path);difference=0.0
            if label=="R0" and split=="outer":
                historical=pd.read_parquet(BASELINE/"predictions_C_patch_no_group.parquet")
                assert frame.sample_id.equals(historical.sample_id)
                historical_delta=max(float(np.max(np.abs(frame[f"p_{TARGETS[j]}"].to_numpy()
                    -historical[f"p_{j//3}_{j%3}"].to_numpy()))) for j in range(9))
                if historical_delta>1e-6:raise ValueError(f"R0 historical replay mismatch: {historical_delta}")
                info["frozen_outer_max_abs_delta"]=historical_delta
            frames[(label,split)]=frame
            info["splits"][split]={"targets":target_metrics(frame),"prediction_sha256":sha(path),
                                    "replay_max_abs_delta":difference,
                                    "raw_order_violation_fraction":violations(raw)}
        info["replay_seconds"]=time.monotonic()-start
        result["trials"][label]=info
    for split in ("inner","outer"):
        for ref_label,slot in (("R0","paired_ci_vs_R0"),("R1","paired_ci_vs_R1")):
            reference=frames[(ref_label,split)]
            result[slot][split]={}
            for label in trials:
                if label==ref_label:continue
                frame=frames[(label,split)]
                result[slot][split][label]={name:block_ci(frame,reference,name,seed=62027+j)
                                             for j,name in enumerate(TARGETS)}
    (output/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({label:{split:result["trials"][label]["splits"][split]["targets"]["3d_5pct"]["final"]["brier"]
                             for split in ("inner","outer")} for label in trials},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--r1-run",type=Path,default=BASE/"runs/v7_c_no_group_R1_seed3566")
    parser.add_argument("--r2-run",type=Path,default=BASE/"runs/v7_c_no_group_R2_seed3566")
    parser.add_argument("--r3a-run",type=Path,default=BASE/"runs/v7_c_no_group_R3_seed3566")
    parser.add_argument("--r3b-run",type=Path,default=BASE/"runs/v7_c_no_group_R3_seed81173")
    args=parser.parse_args()
    main(args.output,args.r1_run,args.r2_run,args.r3a_run,args.r3b_run)
