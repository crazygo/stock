"""Causal path and peer features; no labels are accepted by these functions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .prepare import GROUPS
from research.group_expectation_matrix.build import write_json

def path_summary(x):
    mask = x[:, 6] > 0
    values = []
    for n in [3, 5, 10, 20]:
        z = x[-n:, 1]
        good = mask[-n:]
        if len(z) < n or not good.all():
            values.extend([np.nan]*9)
            continue
        delta = np.diff(z)
        slope = np.polyfit(np.arange(n), z, 1)[0]
        peak = np.maximum.accumulate(z)
        values.extend([z[-1]-z[0], float(delta.std()), float((delta > 0).mean()), float(slope),
                       float(z[-1]-z.max()), float((z-peak).min()), float(np.argmax(z)/(n-1)),
                       float(np.abs(delta).sum()), float(z.max()-z.min())])
    for lag in [1, 3, 5, 10, 19]:
        values.append(float(x[-1, 1]-x[-lag-1, 1]) if len(x) > lag and mask[-1] and mask[-lag-1] else np.nan)
    values.append(float(-x[-1, 1]) if mask[-1] else np.nan)
    return values

def prefix_summary(x):
    z = x[x[:, 6] > 0]
    if len(z) < 2:
        return [np.nan]*7
    c = z[:, 1]
    delta = np.diff(c)
    return [float(c[-1]-c[0]+z[0, 0]), float(delta.std()), float((delta > 0).mean()),
            float(c[-1]-c.max()), float((c-np.maximum.accumulate(c)).min()),
            float(np.argmax(c)/(len(c)-1)), float(np.polyfit(np.arange(len(c)), c, 1)[0])]

def peer_context(rows, own, membership, groups):
    """Peer means exclude the current security; all rows share exact cutoff/date."""
    parents = [i for i, g in enumerate(groups) if g.get("group_kind") == "parent"]
    optics = [i for i, g in enumerate(groups) if g["group_id"] in GROUPS and g["group_id"] not in ["optics:chain", "optics:core"]]
    feature_names = ["return_1", "return_3", "return_5", "return_10", "return_19", "return_today"]
    # The final six daily path fields are simple lagged returns/current reference.
    out = np.full((len(rows), 6*7+1+5), np.nan, np.float32)
    for date, positions in rows.groupby("date", sort=False).indices.items():
        ids = np.asarray(positions)
        assert rows.iloc[ids].cutoff_at.nunique() == 1
        market_ids = ids[rows.symbol.to_numpy()[ids] == "QQQ"]
        market = own[market_ids[0], -6:] if len(market_ids) else np.full(6, np.nan)
        for rid in ids:
            peers = np.zeros(len(rows), bool)
            stage_bits = [int(membership[rid, gi]) for gi in optics]
            relevant = [gi for gi in optics if membership[rid, gi]] or [gi for gi in parents if membership[rid, gi]]
            for gi in relevant:
                peers |= membership[:, gi]
            peer_ids = ids[peers[ids] & (ids != rid)]
            values = own[peer_ids, -6:]
            if len(values):
                means = np.nanmean(values, 0)
                medians = np.nanmedian(values, 0)
                std = np.nanstd(values, 0)
                finite = np.isfinite(values)
                den = finite.sum(0)
                breadth = np.divide(((values > 0) & finite).sum(0), den, out=np.full(6, np.nan), where=den > 0)
                own_value = own[rid, -6:]
                rank = np.divide(((values < own_value) & finite).sum(0), den, out=np.full(6, np.nan), where=den > 0)
                out[rid, :42] = np.r_[market, means, medians, std, breadth, own_value-means, rank]
            else:
                out[rid, :6] = market
            out[rid, 42] = len(peer_ids)
            out[rid, 43:] = stage_bits
    return out

def enhance(output):
    data = output/"data"
    rows = pd.read_parquet(data/"rows.parquet")
    membership = np.load(data/"memberships.npz")["mask"]
    groups = json.loads((data/"groups.json").read_text())
    for gran in [5, 60]:
        base = dict(np.load(data/f"features_{gran}.npz"))
        own = np.asarray([path_summary(x) for x in base["H"]], np.float32)
        intraday = np.asarray([prefix_summary(x) for x in base["B"]], np.float32)
        context = peer_context(rows, own, membership, groups)
        base["tab"] = np.column_stack([base["tab"], own, intraday, context]).astype(np.float32)
        np.savez_compressed(data/f"enhanced_{gran}.npz", **base)
        write_json(data/f"enhanced_{gran}.json", {"base_columns": 124, "path_columns": own.shape[1],
                   "intraday_columns": intraday.shape[1], "context_columns": context.shape[1],
                   "total_columns": base["tab"].shape[1], "labels_used": False, "peer_excludes_self": True})
        print(f"enhanced {gran}m: {base['tab'].shape}", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    enhance(p.parse_args().output)
