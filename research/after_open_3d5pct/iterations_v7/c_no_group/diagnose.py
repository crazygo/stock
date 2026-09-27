"""Read-only coverage, scale, and saved-model component-sensitivity diagnostics."""
from __future__ import annotations

import json

import numpy as np
import torch

from research.after_open_3d5pct.train_multiscale_v6 import project_monotone
from . import runner


def quantiles(x):
    x=np.asarray(x,float)
    return {"n":int(x.size),"p05":float(np.quantile(x,.05)),"median":float(np.median(x)),
            "p95":float(np.quantile(x,.95))}


def main():
    table,data,folds=runner.read_data()
    output={"dataset_manifest_sha256":runner.MANIFEST_SHA,"splits":{},"shape_zero_perturbation":{}}
    for split in ("train","inner","outer"):
        ix=folds[split]
        record={"rows":len(ix),"dates":int(table.iloc[ix].session_date.nunique()),
                "full126_rows":int(table.iloc[ix].full_126_prior_days.sum())}
        windows={"five_current":data["x5"][ix,-1],"hour_current":data["x60"][ix,-1],
                 "day_126":data["xday"][ix],"day_recent21":data["xday"][ix,-21:]}
        record["coverage"]={};record["scale"]={}
        for name,x in windows.items():
            mask=x[...,10]>0
            record["coverage"][name]={"valid_count":quantiles(mask.sum(1)),
                                       "valid_fraction_mean":float(mask.mean()),
                                       "no_valid_rows":int((mask.sum(1)==0).sum())}
            record["scale"][name]={
                "bar_log_close_open":quantiles(x[...,0][mask]),
                "gap_log_open_prev_close":quantiles(x[...,3][mask]),
                "log_open_div5":quantiles(x[...,12][mask]),
                "log_share_volume":quantiles(x[...,13][mask])}
        output["splits"][split]=record
    torch.set_num_threads(1)
    arrays=runner.arrays_of(data)
    checkpoint=torch.load(runner.BASE/"runs/v7_c_no_group_R2_seed3566/model.pt",map_location="cpu",weights_only=False)
    model=runner.make_model("R2");model.load_state_dict(checkpoint["state_dict"])
    original=runner.shape_features
    for split in ("inner","outer"):
        ix=folds[split]
        with_shape=runner.predict(model,arrays,ix)
        try:
            runner.shape_features=lambda x:torch.zeros((x.shape[0],8),dtype=x.dtype,device=x.device)
            without_shape=runner.predict(model,arrays,ix)
        finally:
            runner.shape_features=original
        y=data["y"][ix,1,1]
        p=project_monotone(with_shape)[:,1,1]
        q=project_monotone(without_shape)[:,1,1]
        output["shape_zero_perturbation"][split]={
            "n":len(ix),"mean_abs_probability_change":float(np.mean(np.abs(p-q))),
            "p95_abs_probability_change":float(np.quantile(np.abs(p-q),.95)),
            "brier_with_shape":runner.metrics(y,p)["brier"],
            "brier_zero_shape":runner.metrics(y,q)["brier"],
            "interpretation":"Out-of-distribution inference perturbation; tests model reliance, not retrained ablation value."}
    path=runner.BASE/"iterations_v7/c_no_group/diagnosis.json"
    path.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps({"coverage":{s:{k:round(v["valid_fraction_mean"],3) for k,v in x["coverage"].items()}
                                 for s,x in output["splits"].items()},
                      "shape_zero_perturbation":output["shape_zero_perturbation"]},indent=2))


if __name__=="__main__":main()
