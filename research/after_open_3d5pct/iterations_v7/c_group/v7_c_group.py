"""Offline v7 C-with-typed-groups development; outer scoring is a separate final command."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

from research.after_open_3d5pct.train_multiscale_v6 import (
    CModel, _folds, _predict_c, _torch_arrays, project_monotone,
)

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "research/after_open_3d5pct"
DATASET = BASE / "runs/multiscale_groups_v61_dataset_20260926_r2"
BASELINE = BASE / "runs/multiscale_groups_v611_typefix_fit_20260926"
CONFIG = BASE / "configs/multiscale_groups_v611_typefix.json"
TARGET_NAMES = [f"{h}d_{int(t*100)}pct" for h in (1, 3, 5) for t in (.03, .05, .08)]
PRIMARY = 4


def shape_features(x: torch.Tensor) -> torch.Tensor:
    """Six prefix descriptors from observed Open/Close, anchored at first valid Open."""
    mask = x[...,10] > 0
    log_open = 5*x[...,12]
    log_close = log_open+x[...,0]
    positions=torch.arange(x.shape[1],device=x.device)[None,:].expand(x.shape[0],-1)
    last_valid=torch.cummax(torch.where(mask,positions,-1),1).values
    previous=torch.cat((torch.full_like(last_valid[:,:1],-1),last_valid[:,:-1]),1)
    previous_close=torch.gather(log_close,1,previous.clamp_min(0))
    # For the first observed bar, include only its own close/open movement.
    # Subsequent observed bars include the gap from the previous valid close.
    r=torch.where(previous>=0,log_close-previous_close,log_close-log_open)*mask
    rank = mask.long().cumsum(1) - 1
    count = mask.sum(1).clamp_min(1)
    third = (rank * 3 // count[:,None]).clamp(0,2)
    sums = torch.stack([(r * (third==j)).sum(1) for j in range(3)],1)
    path = r.cumsum(1)
    peak = torch.cummax(path,1).values
    drawdown = (peak-path).amax(1,keepdim=True)
    activity = r.abs().sum(1,keepdim=True)
    efficiency = sums.sum(1,keepdim=True) / activity.clamp_min(1e-6)
    early = (r.abs()*(third==0)).sum(1) / ((mask&(third==0)).sum(1).clamp_min(1))
    late = (r.abs()*(third==2)).sum(1) / ((mask&(third==2)).sum(1).clamp_min(1))
    cluster = (late-early)[:,None]
    return torch.cat((sums*100,drawdown*100,efficiency,cluster*100),1).clamp(-30,30)


class ShapeCModel(CModel):
    """Same typed group relation and branches, plus causal short-path descriptors."""
    def __init__(self):
        super().__init__(group=True,daily=True,input_channels=14,type_identity=True)
        self.head = nn.Sequential(nn.Linear(76+18,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,9))

    def forward(self,x5,x60,xday,group_seq):
        five = self.five(x5.flatten(1,2))
        hour = self.hour(x60.flatten(1,2))
        day = self.day(xday)
        m = group_seq[...,9:10]
        typed = torch.einsum("bwgc,gcd->bwgd",group_seq,self.member_weight)+self.member_bias
        weekly = (torch.nn.functional.gelu(typed)*m).sum(2)/m.sum(2).clamp_min(1)
        _,h=self.week(weekly); relation=h[-1]
        shape=torch.cat((shape_features(x5[:,-1]),shape_features(x60[:,-1]),
                         shape_features(xday[:,-21:])),1)
        return self.head(torch.cat((five,hour,day,relation,
                                    five*torch.sigmoid(self.gate(relation)),shape),1))


def group_contrast(group_seq: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    """Per-type current minus observed-prior mean, with explicit comparison validity."""
    previous=group_seq[:,:-1]
    prior_mask=previous[...,9:10]>0
    current=group_seq[:,-1]
    valid=(current[...,9:10]>0)&prior_mask.any(1)
    prior_mean=(previous[...,:10]*prior_mask).sum(1)/prior_mask.sum(1).clamp_min(1)
    delta=(current-prior_mean)*valid
    return delta,valid.float()


class ContrastCModel(ShapeCModel):
    def __init__(self):
        super().__init__()
        self.delta_weight=nn.Parameter(torch.randn(6,10,4)*.02)
        self.delta_bias=nn.Parameter(torch.zeros(6,4))
        self.head=nn.Sequential(nn.Linear(76+18+24+6,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,9))

    def forward(self,x5,x60,xday,group_seq):
        five=self.five(x5.flatten(1,2))
        hour=self.hour(x60.flatten(1,2))
        day=self.day(xday)
        m=group_seq[...,9:10]
        typed=torch.einsum("bwgc,gcd->bwgd",group_seq,self.member_weight)+self.member_bias
        weekly=(torch.nn.functional.gelu(typed)*m).sum(2)/m.sum(2).clamp_min(1)
        _,h=self.week(weekly);relation=h[-1]
        shape=torch.cat((shape_features(x5[:,-1]),shape_features(x60[:,-1]),
                         shape_features(xday[:,-21:])),1)
        delta,valid=group_contrast(group_seq)
        change=torch.nn.functional.gelu(torch.einsum("bgc,gcd->bgd",delta,self.delta_weight)
                                        +self.delta_bias)*valid
        return self.head(torch.cat((five,hour,day,relation,
                                    five*torch.sigmoid(self.gate(relation)),shape,
                                    change.flatten(1),valid.flatten(1)),1))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_data():
    if sha(DATASET / "manifest.json") != "56b3d04def58b72e388f98e41c0d72df1b996aef3ecf7c8ee6325349082a143b":
        raise ValueError("Frozen dataset manifest changed")
    table = pd.read_parquet(DATASET / "rows.parquet")
    with np.load(DATASET / "features.npz") as z:
        data = {k: z[k] for k in z.files}
    config = json.loads(CONFIG.read_text())
    folds = _folds(table, config)
    assert {k: len(v) for k, v in folds.items()} == {"train": 5606, "inner": 1521, "outer": 1140}
    assert len(table) == len(data["y"]) == 9222
    return table, data, folds, config


def measure(y, p):
    y = np.asarray(y).ravel(); p = np.clip(np.asarray(p).ravel(), 1e-6, 1-1e-6)
    return {"n": int(len(y)), "rate": float(y.mean()), "mean_p": float(p.mean()),
            "brier": float(np.mean((y-p)**2)), "logloss": float(log_loss(y,p,labels=[0,1])),
            "auc": float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
            "p05": float(np.quantile(p,.05)), "p50": float(np.quantile(p,.5)),
            "p95": float(np.quantile(p,.95))}


def buckets(y, p):
    y = np.asarray(y).ravel(); p = np.asarray(p).ravel()
    edges = np.linspace(0,1,11)
    return [{"range": [float(edges[i]),float(edges[i+1])], "n": int(m.sum()),
             "mean_p": float(p[m].mean()) if m.any() else None,
             "rate": float(y[m].mean()) if m.any() else None}
            for i in range(10) for m in [((p>=edges[i]) & (p<edges[i+1] if i<9 else p<=1))]]


def save_predictions(path, table, ids, y, raw):
    projected = project_monotone(raw).reshape(-1,9)
    raw = raw.reshape(-1,9)
    frame = table.iloc[ids][["sample_id","symbol","session_date","decision_at"]].reset_index(drop=True).copy()
    for j, name in enumerate(TARGET_NAMES):
        frame[f"y_{name}"] = y.reshape(-1,9)[:,j].astype(np.int8)
        frame[f"raw_{name}"] = raw[:,j]
        frame[f"p_{name}"] = projected[:,j]
    frame.to_parquet(path,index=False)
    return frame


def summary(frame):
    out = {}
    for name in TARGET_NAMES:
        y = frame[f"y_{name}"].to_numpy(); p = frame[f"p_{name}"].to_numpy()
        out[name] = {"projected": measure(y,p), "raw": measure(y,frame[f"raw_{name}"].to_numpy()),
                     "reliability10": buckets(y,p)}
    return out


def fit(round_name: str, output: Path, seed: int, max_epochs: int = 40, patience: int = 6):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    table,data,folds,config = read_data()
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    arrays = _torch_arrays(data)
    target = torch.from_numpy(data["y"].astype(np.float32).reshape(-1,9))
    model = (CModel(group=True,daily=True,input_channels=14,type_identity=True)
             if round_name=="R1" else ShapeCModel() if round_name=="R2" else ContrastCModel())
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    rng=np.random.default_rng(seed)
    best=math.inf; best_epoch=0; state=None; bad=0; curve=[]; started=time.monotonic()
    try:
        for epoch in range(1,max_epochs+1):
            model.train(); losses=[]
            order=rng.permutation(folds["train"])
            for ix in np.array_split(order,max(1,math.ceil(len(order)/128))):
                index=torch.as_tensor(ix,dtype=torch.long)
                opt.zero_grad(set_to_none=True)
                logits=model(*(torch.index_select(a,0,index) for a in arrays))
                loss=nn.functional.binary_cross_entropy_with_logits(logits,torch.index_select(target,0,index))
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
                losses.append(float(loss.item()))
            inner_raw=_predict_c(model,arrays,folds["inner"])
            train_raw=_predict_c(model,arrays,folds["train"])
            inner_p=project_monotone(inner_raw)[:,1,1]
            train_p=project_monotone(train_raw)[:,1,1]
            bi=measure(data["y"][folds["inner"],1,1],inner_p)["brier"]
            bt=measure(data["y"][folds["train"],1,1],train_p)["brier"]
            curve.append({"epoch":epoch,"train_loss9":float(np.mean(losses)),
                          "train_primary_brier":bt,"inner_primary_brier":bi,
                          "seconds":time.monotonic()-started})
            if bi < best-1e-5:
                best=bi; best_epoch=epoch; state=copy.deepcopy(model.state_dict());bad=0
            else: bad+=1
            if bad>=patience: break
        if state is None: raise RuntimeError("No checkpoint")
        model.load_state_dict(state)
        torch.save({"state_dict":state,"round":round_name,"seed":seed,
                    "variant":{"R1":"r1_base","R2":"r2_shape_corrected","R3":"r3_typed_contrast"}[round_name],
                    "dataset_manifest_sha256":sha(DATASET/"manifest.json"),
                    "feature_npz_sha256":sha(DATASET/"features.npz")},output/"model.pt")
        stats={"round":round_name,"seed":seed,"status":"complete","best_epoch":best_epoch,
               "parameters":sum(p.numel() for p in model.parameters()),"seconds":time.monotonic()-started,
               "curve":curve,"dataset_manifest_sha256":sha(DATASET/"manifest.json"),
               "feature_npz_sha256":sha(DATASET/"features.npz"),
               "row_table_sha256":sha(DATASET/"rows.parquet"),"base_config_sha256":sha(CONFIG),
               "source_sha256":sha(Path(__file__)),"torch_version":torch.__version__,
               "numpy_version":np.__version__,"split_rows":{k:len(v) for k,v in folds.items()}}
        for split in ("train","inner"):
            ids=folds[split]; raw=_predict_c(model,arrays,ids)
            frame=save_predictions(output/f"{split}_predictions.parquet",table,ids,data["y"][ids],raw)
            stats[split]=summary(frame)
        (output/"results.json").write_text(json.dumps(stats,indent=2)+"\n")
        print(json.dumps({"round":round_name,"seed":seed,"best_epoch":best_epoch,
                          "inner_brier":stats["inner"]["3d_5pct"]["projected"]["brier"],
                          "train_brier":stats["train"]["3d_5pct"]["projected"]["brier"],
                          "seconds":stats["seconds"]}))
    except Exception as e:
        (output/"failure.json").write_text(json.dumps({"error":repr(e),"round":round_name,"seed":seed})+"\n")
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--round",required=True,choices=["R1","R2","R3"])
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=3566)
    args=parser.parse_args()
    fit(args.round,args.output,args.seed)


if __name__=="__main__": main()
