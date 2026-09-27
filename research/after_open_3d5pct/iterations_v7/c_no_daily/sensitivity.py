"""Inner-only component permutation sensitivity, not a causal ablation fit."""
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import BASE, NoDailyC, metric, predict, project_monotone, read_data


def main():
    table, data, folds, arrays = read_data()
    torch.set_num_threads(1)
    run = BASE/"runs/v7_c_no_daily_r1_3566"
    ckpt = torch.load(run/"model.pt", map_location="cpu", weights_only=False)
    model = NoDailyC(shape=False)
    model.load_state_dict(ckpt["state_dict"])
    ix = folds["inner"]
    selected = tuple(torch.index_select(a, 0, torch.as_tensor(ix)) for a in arrays)
    dates = table.iloc[ix].session_date.to_numpy()
    perm = np.arange(len(ix))
    rng = np.random.default_rng(20260927)
    for date in np.unique(dates):
        loc = np.flatnonzero(dates == date)
        perm[loc] = rng.permutation(loc)
    y = data["y"][ix, 1, 1]
    out = {}
    for label, slot in (("unchanged", -1), ("same_date_x5_permuted", 0),
                        ("same_date_x60_permuted", 1), ("same_date_group_permuted", 2)):
        arr = list(selected)
        if slot >= 0:
            arr[slot] = arr[slot][torch.as_tensor(perm)]
        p = project_monotone(predict(model, tuple(arr), np.arange(len(ix))))[:, 1, 1]
        out[label] = metric(y, p)
    path = Path(__file__).resolve().parent/"sensitivity.json"
    path.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
