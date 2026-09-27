"""Causal splits, feature transforms and paired evaluation for v8."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from research.after_open_3d5pct.train_multiscale_v6 import CModel, PatchBranch, project_monotone
from research.after_open_3d5pct.iterations_v7.c_no_daily.experiment import relative_coordinates, shape_features

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
GROUPS = PROTOCOL["groups"]
TARGETS = PROTOCOL["targets"]


def splits(rows, fold):
    available = pd.to_datetime(rows.label_available_at, utc=True)
    end = pd.to_datetime(rows.label_end_at, utc=True)
    out = {}
    starts = ["1900-01-01", fold["tune"], fold["cal"], fold["eval"], fold["end"]]
    for j, name in enumerate(("fit", "tune", "cal", "eval")):
        m = (rows.session_date >= starts[j]) & (rows.session_date < starts[j+1])
        if name != "eval":
            bound = pd.Timestamp(starts[j+1], tz="America/New_York").tz_convert("UTC")
            m &= (end < bound) & (available < bound)
        out[name] = np.flatnonzero(m.to_numpy())
    return out


def focus_weights(rows):
    """Each of three groups gets equal mass, including fractional overlap mass."""
    w = np.zeros(len(rows), float)
    for symbols in GROUPS.values():
        m = rows.symbol.isin(symbols).to_numpy()
        if m.any():
            w[m] += 1 / m.sum() / 3
    return w


def mature_prior(rows, y, window=63):
    """Own-symbol labels only; no cross-stock input, even for an unseen symbol.

    A fixed Beta(1,1) startup prior is independent of future labels. At each row
    only complete nine-task outcomes available strictly before its decision count.
    """
    result = np.full((len(rows), 9), .5, np.float32)
    counts = np.zeros(len(rows), np.int32)
    available = pd.to_datetime(rows.label_available_at, utc=True).astype("int64").to_numpy()
    end = pd.to_datetime(rows.label_end_at, utc=True).astype("int64").to_numpy()
    decisions = pd.to_datetime(rows.decision_at, utc=True).astype("int64").to_numpy()
    for _, ids0 in rows.groupby("symbol").indices.items():
        ids = np.array(sorted(ids0, key=lambda i: decisions[i]))
        for i in ids:
            old = ids[(decisions[ids] < decisions[i]) & (available[ids] < decisions[i]) & (end[ids] < decisions[i])][-window:]
            counts[i] = len(old)
            result[i] = (y[old].sum(0) + 1) / (len(old) + 2)
    return result, counts


def historical_baseline(rows, y, fit, ids):
    p = y[fit].mean(0)
    per = {s: (y[js].sum(0)+20*p)/(len(js)+20) for s in rows.iloc[fit].symbol.unique()
           for js in [fit[rows.iloc[fit].symbol.to_numpy() == s]]}
    return np.stack([per.get(s, p) for s in rows.iloc[ids].symbol])


def within_auc(symbols, y, p):
    numerator, pairs = 0., 0
    for s in np.unique(symbols):
        m = symbols == s
        yy, pp = y[m], p[m]
        n = int(yy.sum()) * int((1-yy).sum())
        if n:
            numerator += roc_auc_score(yy, pp) * n
            pairs += n
    return float(numerator / pairs) if pairs else None


def metric(rows, y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    buckets = []
    for lo in np.arange(0, 1, .1):
        m = (p >= lo) & (p < lo+.100000001)
        buckets.append({"lo": float(lo), "n": int(m.sum()), "predicted": float(p[m].mean()) if m.any() else None,
                        "observed": float(y[m].mean()) if m.any() else None})
    return {"n": len(y), "dates": rows.session_date.nunique(), "stocks": rows.symbol.nunique(),
            "brier": float(np.mean((p-y)**2)), "logloss": float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
            "observed": float(y.mean()), "predicted": float(p.mean()), "mean_gap": float(abs(p.mean()-y.mean())),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "within_auc": within_auc(rows.symbol.to_numpy(), y, p), "buckets": buckets}


def metrics(rows, y, p, groups=None):
    groups = groups or GROUPS
    return {name: {t: metric(rows.loc[m].reset_index(drop=True), y[m, j], p[m, j]) for j, t in enumerate(TARGETS)}
            for name, symbols in groups.items() for m in [rows.symbol.isin(symbols).to_numpy()] if m.any()}


def score(rows, y, p):
    w = focus_weights(rows)
    return float(np.sum(w * (y[:, 4]-p[:, 4])**2))


def path_summary(data):
    """Observed path summaries across separate scales, never high-degree curve fits."""
    out = []
    for name, windows in (("x5", (24, 78, 192, 768)), ("x60", (8, 34, 170, 510)), ("xday", (10, 21, 63, 126))):
        x = data[name].reshape(len(data[name]), -1, 14)
        # Compress valid observations, so today's padding does not erase the end.
        for n in windows:
            rows = []
            for seq in x:
                z = seq[seq[:, 10] > 0][-n:].astype(float)
                if len(z) < 2:
                    rows.append([np.nan]*7)
                    continue
                path = 5*z[:, 12]+z[:, 0] - 5*z[0, 12]
                r = np.diff(np.r_[0, path])
                a, b, c = np.array_split(r, 3)
                rows.append([path[-1], np.std(r), np.max(np.maximum.accumulate(np.r_[0,path])[1:]-path),
                             path[-1]/max(abs(r).sum(), 1e-6), a.sum(), b.sum(), c.sum()])
            out.append(np.asarray(rows))
    return np.nan_to_num(np.column_stack(out)).astype(np.float32)


class MixingPatch(PatchBranch):
    """Convolution across ordered patches before masked pooling."""
    def __init__(self, patch, max_tokens, input_channels, width=16):
        super().__init__(patch, max_tokens, input_channels, width)
        self.mix = nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(width, width, 3, padding=2, dilation=2))

    def forward(self, x):
        mask = torch.nn.functional.avg_pool1d(x[..., 10].unsqueeze(1), self.patch, stride=self.patch).squeeze(1) > 0
        z = torch.nn.functional.gelu(self.proj(x.transpose(1, 2))).transpose(1, 2)
        z = (z + self.pos(torch.arange(z.shape[1], device=z.device))) * mask[..., None]
        z = self.norm(z + self.mix(z.transpose(1, 2)).transpose(1, 2)) * mask[..., None]
        score = (z*self.query).sum(-1)/4
        weight = torch.softmax(score.masked_fill(~mask, -1e4), 1)*mask
        return (z*weight[..., None]).sum(1)/weight.sum(1, keepdim=True).clamp_min(1e-6)


class FocusC(CModel):
    def __init__(self, route, recipe):
        super().__init__(group=route != "C_no_group", daily=route != "C_no_daily", type_identity=True)
        self.recipe = recipe
        if recipe.get("capacity"):
            self.five = MixingPatch(12, 128, 14)
            self.hour = MixingPatch(3, 176, 14)
            if self.daily:
                self.day = MixingPatch(6, 21, 14)
        if recipe.get("curves"):
            self.curve_head = nn.Linear(12 + (6 if self.daily else 0), 9)
            nn.init.zeros_(self.curve_head.weight)
            nn.init.zeros_(self.curve_head.bias)
        if recipe.get("regularization"):
            self.head[2] = nn.Dropout(.25)

    def forward(self, x5, x60, xday, group_seq, prior):
        a, b, c = x5, x60, xday
        if self.recipe.get("representation"):
            a, b = relative_coordinates(a), relative_coordinates(b)
            if self.daily:
                c = relative_coordinates(c)
        # No-daily never transforms or passes real xday into a branch.
        z = super().forward(a, b, c if self.daily else torch.empty(0), group_seq)
        if self.recipe.get("curves"):
            parts = [shape_features(x5[:, -1, 66:90]), shape_features(x60.flatten(1, 2))]
            if self.daily:
                parts.append(shape_features(xday))
            z = z + self.curve_head(torch.cat(parts, 1))
        if self.recipe.get("prior"):
            p = prior.clamp(.01, .99)
            z = z + torch.logit(p)
        return z
