"""Offline C-no-group experiments. Outer scores are emitted only by finalize.py."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

from research.after_open_3d5pct.train_multiscale_v6 import CModel, _folds, project_monotone


BASE = Path(__file__).resolve().parents[2]
DATASET = BASE / "runs/multiscale_groups_v61_dataset_20260926_r2"
BASELINE = BASE / "runs/multiscale_groups_v61_fit_20260926"
CONFIG = BASE / "configs/multiscale_groups_v61.json"
MANIFEST_SHA = "56b3d04def58b72e388f98e41c0d72df1b996aef3ecf7c8ee6325349082a143b"
TARGETS = [f"{h}d_{t}pct" for h in (1, 3, 5) for t in (3, 5, 8)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_data():
    if sha(DATASET / "manifest.json") != MANIFEST_SHA:
        raise ValueError("Frozen manifest changed")
    table = pd.read_parquet(DATASET / "rows.parquet")
    # Intentionally no group/group_seq array read, materialized, or passed to the model.
    with np.load(DATASET / "features.npz") as z:
        data = {k: z[k] for k in ("x5", "x60", "xday", "y")}
    folds = _folds(table, json.loads(CONFIG.read_text()))
    assert len(table) == len(data["y"]) == 9222
    assert {k: len(v) for k, v in folds.items()} == {"train": 5606, "inner": 1521, "outer": 1140}
    return table, data, folds


def arrays_of(data):
    arrays = tuple(torch.from_numpy(np.nan_to_num(data[k]).astype(np.float32))
                   for k in ("x5", "x60", "xday"))
    dummy = torch.empty((len(data["y"]), 0), dtype=torch.float32)
    return (*arrays, dummy)


def metrics(y, p):
    y = np.asarray(y).ravel(); p = np.clip(np.asarray(p).ravel(), 1e-6, 1-1e-6)
    return {"n": int(len(y)), "rate": float(y.mean()), "mean_p": float(p.mean()),
            "brier": float(np.mean((y-p)**2)), "logloss": float(log_loss(y,p,labels=[0,1])),
            "auc": float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
            "p05": float(np.quantile(p,.05)), "p50": float(np.quantile(p,.5)),
            "p95": float(np.quantile(p,.95))}


def buckets(y, p):
    y = np.asarray(y).ravel(); p = np.asarray(p).ravel()
    edges = np.linspace(0, 1, 11)
    return [{"range": [float(edges[i]), float(edges[i+1])], "n": int(m.sum()),
             "mean_p": float(p[m].mean()) if m.any() else None,
             "rate": float(y[m].mean()) if m.any() else None}
            for i in range(10) for m in [((p >= edges[i]) & (p < edges[i+1] if i < 9 else p <= 1))]]


def predict(model, arrays, ids):
    model.eval(); chunks = []
    with torch.no_grad():
        for ix in np.array_split(ids, max(1, math.ceil(len(ids)/256))):
            if len(ix):
                index = torch.as_tensor(ix, dtype=torch.long)
                chunks.append(torch.sigmoid(model(*(a.index_select(0,index) for a in arrays)))
                              .cpu().numpy().reshape(-1,3,3))
    return np.concatenate(chunks)


def save_predictions(path, table, ids, y, raw):
    frame = table.iloc[ids][["sample_id","symbol","session_date","decision_at"]].reset_index(drop=True).copy()
    raw = raw.reshape(-1,9); projected = project_monotone(raw).reshape(-1,9)
    for j, name in enumerate(TARGETS):
        frame[f"y_{name}"] = y.reshape(-1,9)[:,j].astype(np.int8)
        frame[f"raw_{name}"] = raw[:,j]
        frame[f"p_{name}"] = projected[:,j]
    frame.to_parquet(path,index=False)
    return frame


def summarize(frame):
    out = {}
    for name in TARGETS:
        y=frame[f"y_{name}"].to_numpy(); raw=frame[f"raw_{name}"].to_numpy(); p=frame[f"p_{name}"].to_numpy()
        out[name] = {"raw": metrics(y,raw), "projected": metrics(y,p), "reliability10": buckets(y,p)}
    return out


def make_model(round_name):
    if round_name == "R1":
        return CModel(group=False,daily=True,input_channels=14)
    if round_name == "R2":
        return ShapeCModel()
    if round_name == "R3":
        return ShapeCModel()
    raise ValueError(round_name)


def shape_features(x):
    """Observed endpoint path descriptors, anchored at first valid Open.

    Channel 12*5 is log Open, channel 0 is log(Close/Open). Differences
    across missing intervals reflect observed endpoint gaps only. Gap fraction
    makes that incomplete path explicit; no unseen extrema are invented.
    """
    valid=x[...,10]>0
    open_log=5*x[...,12]
    close_log=open_log+x[...,0]
    n=valid.sum(1); safe_n=n.clamp_min(1)
    first=torch.argmax(valid.int(),dim=1)
    positions=torch.arange(x.shape[1],device=x.device)[None,:].expand_as(valid)
    last=torch.where(valid,positions,-1).max(1).values.clamp_min(0)
    anchor=open_log.gather(1,first[:,None])
    value=close_log-anchor
    # Set unknown positions to the last observed Close for a valid-only path.
    prev_ix=torch.cummax(torch.where(valid,positions,-1),dim=1).values
    observed=close_log.gather(1,prev_ix.clamp_min(0))
    observed=torch.where(prev_ix>=0,observed,anchor)
    prev=torch.cat((anchor,observed[:,:-1]),dim=1)
    change=torch.where(valid,close_log-prev,0.)
    # First bar is measured from first Open, including its intrabar return.
    first_flag=(positions==first[:,None]) & valid
    change=torch.where(first_flag,x[...,0],change)
    rank=valid.long().cumsum(1)-1
    third=(rank*3//safe_n[:,None]).clamp(0,2)
    segments=torch.stack([(change*(third==j)).sum(1) for j in range(3)],1)
    cum=change.cumsum(1)
    peak=torch.cummax(torch.cat((torch.zeros_like(cum[:,:1]),cum),1),1).values[:,1:]
    drawdown=(peak-cum).amax(1,keepdim=True)
    travel=change.abs().sum(1,keepdim=True)
    efficiency=segments.sum(1,keepdim=True).abs()/travel.clamp_min(1e-6)
    early=change.abs().mul(third==0).sum(1)/((valid&(third==0)).sum(1).clamp_min(1))
    late=change.abs().mul(third==2).sum(1)/((valid&(third==2)).sum(1).clamp_min(1))
    internal=(positions>=first[:,None]) & (positions<=last[:,None])
    gaps=(internal & ~valid).sum(1,keepdim=True)/internal.sum(1,keepdim=True).clamp_min(1)
    coverage=(n/x.shape[1])[:,None]
    return torch.cat((segments*100,drawdown*100,efficiency,(late-early)[:,None]*100,gaps,coverage),1).clamp(-30,30)


class ShapeCModel(CModel):
    def __init__(self):
        super().__init__(group=False,daily=True,input_channels=14)
        self.head=nn.Sequential(nn.Linear(48+24,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,9))

    def forward(self,x5,x60,xday,unused):
        five=self.five(x5.flatten(1,2)); hour=self.hour(x60.flatten(1,2)); day=self.day(xday)
        shape=torch.cat((shape_features(x5[:,-1]),shape_features(x60[:,-1]),
                         shape_features(xday[:,-21:])),1)
        return self.head(torch.cat((five,hour,day,shape),1))


def fit(round_name, output, seed=3566):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    (output/"source_at_fit.py").write_bytes(Path(__file__).read_bytes())
    table,data,folds=read_data()
    torch.set_num_threads(1); torch.manual_seed(seed);np.random.seed(seed)
    arrays=arrays_of(data); target=torch.from_numpy(data["y"].astype(np.float32).reshape(-1,9))
    model=make_model(round_name)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(seed)
    best=math.inf;best_epoch=0;state=None;bad=0;curve=[];started=time.monotonic()
    for epoch in range(1,41):
        model.train();losses=[]
        order=rng.permutation(folds["train"])
        for ix in np.array_split(order,max(1,math.ceil(len(order)/128))):
            index=torch.as_tensor(ix,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits=model(*(a.index_select(0,index) for a in arrays))
            if round_name == "R3":
                task_weights=torch.tensor([1,1,1,1,3,1,1,1,1],dtype=logits.dtype)
                per_target=nn.functional.binary_cross_entropy_with_logits(
                    logits,target.index_select(0,index),reduction="none")
                loss=(per_target*task_weights).sum(dim=1).div(task_weights.sum()).mean()
            else:
                loss=nn.functional.binary_cross_entropy_with_logits(logits,target.index_select(0,index))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();losses.append(float(loss.item()))
        inner_raw=predict(model,arrays,folds["inner"])
        train_raw=predict(model,arrays,folds["train"])
        inner_p=project_monotone(inner_raw)[:,1,1]
        train_p=project_monotone(train_raw)[:,1,1]
        bi=metrics(data["y"][folds["inner"],1,1],inner_p)["brier"]
        bt=metrics(data["y"][folds["train"],1,1],train_p)["brier"]
        curve.append({"epoch":epoch,"train_loss9":float(np.mean(losses)),
                      "train_primary_brier":bt,"inner_primary_brier":bi,
                      "seconds":time.monotonic()-started})
        if bi<best-1e-5:
            best=bi;best_epoch=epoch;state=copy.deepcopy(model.state_dict());bad=0
        else:bad+=1
        if bad>=6:break
    assert state is not None
    model.load_state_dict(state)
    torch.save({"state_dict":state,"round":round_name,"seed":seed,"dataset_manifest_sha256":MANIFEST_SHA,
                "model_code_sha256":sha(Path(__file__))},output/"model.pt")
    result={"round":round_name,"seed":seed,"best_epoch":best_epoch,"epochs_run":len(curve),
            "parameters":sum(p.numel() for p in model.parameters()),"seconds":time.monotonic()-started,
            "curve":curve,"split_rows":{k:len(v) for k,v in folds.items()},
            "source_sha256":sha(Path(__file__)),"config_sha256":sha(CONFIG),
            "dataset_manifest_sha256":MANIFEST_SHA,"features_sha256":sha(DATASET/"features.npz"),
            "rows_sha256":sha(DATASET/"rows.parquet"),"python":platform.python_version(),
            "torch":torch.__version__,"numpy":np.__version__,"calibration":"none",
            "task_weights":[1,1,1,1,3,1,1,1,1] if round_name=="R3" else [1]*9}
    for split in ("train","inner"):
        ids=folds[split];raw=predict(model,arrays,ids)
        frame=save_predictions(output/f"{split}_predictions.parquet",table,ids,data["y"][ids],raw)
        result[split]=summarize(frame)
    transform={"round":round_name,"seed":seed,"no_group":True,
               "input_keys":["x5","x60","xday"],
               "normalization":"frozen per-row causal v6.1 channel transform; no learned scaler",
               "shape":"none" if round_name=="R1" else "current 5m/60m and recent21 daily observed endpoint descriptors",
               "loss_weights":result["task_weights"],"calibration":"none",
               "output":"sigmoid then fixed monotonic 3x3 nesting projection",
               "source_sha256":result["source_sha256"],
               "dataset_manifest_sha256":MANIFEST_SHA}
    (output/"transform.json").write_text(json.dumps(transform,indent=2)+"\n")
    (output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"round":round_name,"seed":seed,"best_epoch":best_epoch,
                      "inner_brier":result["inner"]["3d_5pct"]["projected"]["brier"],
                      "seconds":result["seconds"]}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--round",required=True,choices=["R1","R2","R3"])
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--seed",type=int,default=3566)
    args=parser.parse_args();fit(args.round,args.output,args.seed)


if __name__=="__main__":main()
