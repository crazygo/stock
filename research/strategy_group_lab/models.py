"""Fixed, CPU-bounded estimators and inner-period-only calibration."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch
from torch import nn
from lightgbm import LGBMClassifier

torch.set_num_threads(1)
BRANCHES = ["H", "post", "pre", "B"]

class Preprocessor:
    def fit(self, data, ids):
        a = data["tab"][ids]
        self.median = np.nanmedian(a, axis=0)
        self.median = np.nan_to_num(self.median)
        a = np.where(np.isfinite(a), a, self.median)
        self.mean, self.std = a.mean(0), np.maximum(a.std(0), .01)
        return self
    def transform(self, data):
        a = data["tab"]
        return np.clip((np.where(np.isfinite(a), a, self.median)-self.mean)/self.std, -10, 10).astype(np.float32)

class Branch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.project = nn.Conv1d(7, channels, 1)
        self.layers = nn.ModuleList([nn.Conv1d(channels, channels, 3, padding=2*d, dilation=d) for d in [1, 2]])
    def forward(self, a):
        mask = a[:, :, 6:7].transpose(1, 2)
        x = torch.relu(self.project(a.transpose(1, 2))) * mask
        for layer in self.layers:
            y = torch.relu(layer(x)[..., :x.shape[-1]]) * mask
            x = (x+y)*mask
        return x.sum(-1)/mask.sum(-1).clamp_min(1)

class TCN(nn.Module):
    def __init__(self, static_width, channels):
        super().__init__()
        self.branches = nn.ModuleList([Branch(channels) for _ in BRANCHES])
        self.head = nn.Sequential(nn.Linear(channels*4+static_width, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, arrays, static):
        return self.head(torch.cat([b(x) for b, x in zip(self.branches, arrays)]+[static], dim=1)).squeeze(1)

class Estimator:
    def __init__(self, algorithm, config, seed):
        self.algorithm, self.config, self.seed = algorithm, config, seed
    def fit(self, data, ids, y):
        self.prep = Preprocessor().fit(data, ids)
        static = self.prep.transform(data)
        if self.algorithm == "lgbm":
            self.model = LGBMClassifier(**self.config["lgbm"], random_state=self.seed)
            self.model.fit(static[ids], y)
        else:
            torch.manual_seed(self.seed)
            self.model = TCN(static.shape[1], self.config["tcn"]["channels"])
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.config["tcn"]["learning_rate"],
                                   weight_decay=self.config["tcn"]["weight_decay"])
            rng = np.random.default_rng(self.seed)
            batch = self.config["tcn"]["batch_size"]
            target = torch.from_numpy(y.astype(np.float32))
            arrays = [torch.from_numpy(data[k][ids]) for k in BRANCHES]
            st = torch.from_numpy(static[ids])
            self.model.train()
            for _ in range(self.config["tcn"]["epochs"]):
                order = rng.permutation(len(ids))
                for start in range(0, len(ids), batch):
                    ix = order[start:start+batch]
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.binary_cross_entropy_with_logits(self.model([a[ix] for a in arrays], st[ix]), target[ix])
                    loss.backward()
                    opt.step()
        return self
    def predict(self, data, ids):
        if not len(ids):
            return np.empty(0)
        static = self.prep.transform(data)
        if self.algorithm == "lgbm":
            return self.model.predict_proba(static[ids])[:, 1]
        self.model.eval()
        results = []
        with torch.no_grad():
            for start in range(0, len(ids), 512):
                ix = ids[start:start+512]
                logits = self.model([torch.from_numpy(data[k][ix]) for k in BRANCHES], torch.from_numpy(static[ix]))
                results.extend(torch.sigmoid(logits).numpy().tolist())
        return np.asarray(results)

def calibrate(p, y, dates, fallback=None):
    if len(y) < 40 or len(set(dates)) < 10 or min(np.sum(y == 0), np.sum(y == 1)) < 5:
        return {**(fallback or {"a": 1., "b": 0.}), "status": "pooled_fallback" if fallback else "identity_insufficient_calibration", "n": len(y)}
    x = logit(np.clip(p, 1e-5, 1-1e-5))
    def objective(z):
        pred = expit(z[0]*x+z[1])
        loss = -np.mean(y*np.log(pred+1e-12)+(1-y)*np.log(1-pred+1e-12)) + .001*((z[0]-1)**2+z[1]**2)
        return loss
    result = minimize(objective, [1., 0.], bounds=[(.01, 10.), (-10., 10.)], method="L-BFGS-B")
    if not result.success:
        return {"a": 1., "b": 0., "status": "identity_optimizer_failed", "n": len(y)}
    return {"a": float(result.x[0]), "b": float(result.x[1]), "status": "inner_sigmoid", "n": len(y)}

def apply_calibration(p, calibration):
    return expit(calibration["a"]*logit(np.clip(p, 1e-5, 1-1e-5))+calibration["b"])

def train_status(rows, ids, y, cfg):
    counts = {"train_n": len(ids), "train_dates": int(rows.iloc[ids].date.nunique()),
              "train_symbols": int(rows.iloc[ids].symbol.nunique()), "positive": int(np.sum(y == 1)), "negative": int(np.sum(y == 0))}
    reason = None
    for key, minimum in [("train_n", cfg["minimum_train_rows"]), ("train_dates", cfg["minimum_train_dates"]),
                         ("train_symbols", cfg["minimum_train_symbols"]), ("positive", cfg["minimum_train_class"]),
                         ("negative", cfg["minimum_train_class"])]:
        if counts[key] < minimum:
            reason = f"{key}<{minimum}"
            break
    return counts, reason
