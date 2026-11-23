"""Sample strict subsets of the mutually exclusive original five-bar buys."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from check import key, outcomes

HERE = Path(__file__).resolve().parent
SEED = 20261120
REPLICATES = 50000


def mean_product(factors, k):
    """Uniform sampling without replacement; expected product, no simulation."""
    e = np.zeros(k + 1)
    e[0] = 1.
    for n, factor in enumerate(factors, start=1):
        for j in range(min(k, n), 0, -1):
            e[j] = e[j]*(1-j/n) + e[j-1]*factor*j/n
    return float(e[k])


def hypergeom(n, losses, keep):
    law = np.zeros(keep + 1)
    denom = math.comb(n, keep)
    for j in range(max(0, keep - (n - losses)), min(keep, losses) + 1):
        law[j] = math.comb(losses, j) * math.comb(n - losses, keep - j) / denom
    return law


def group_indices(original, candidate_keys, stratified):
    groups = defaultdict(list)
    for i, trade in enumerate(original):
        group = (trade["symbol"], trade["entry_day"], trade["entry_session"]) if stratified else ("all",)
        groups[group].append(i)
    for indices in groups.values():
        keep = sum(key(original[i]) in candidate_keys for i in indices)
        if keep:
            yield indices, keep


def reference(original, selected, stratified):
    rng = np.random.default_rng(SEED + int(stratified))
    selected_keys = {key(t) for t in selected}
    loss_law = np.array([1.])
    log_return = np.zeros(REPLICATES)
    exact_factor, exp_closed = 1., 0.
    strata = variable = fixed = 0
    for indices, keep in group_indices(original, selected_keys, stratified):
        strata += 1
        group = [original[i] for i in indices]
        n = len(group)
        closed = [t for t in group if t["exit_reason"] != "open"]
        exp_closed += keep * len(closed) / n
        losses = sum(t["net"] < 0 for t in closed)
        loss_law = np.convolve(loss_law, hypergeom(n, losses, keep))
        factors = np.array([1+t["net"] for t in group])
        exact_factor *= mean_product(factors, keep)
        log_factors = np.log(factors)
        if keep == n:
            fixed += keep
            log_return += log_factors.sum()
            continue
        variable += 1
        for first in range(0, REPLICATES, 2048):
            draws = min(2048, REPLICATES-first)
            random_ranks = rng.random((draws, n))
            chosen = np.argpartition(random_ranks, keep-1, axis=1)[:, :keep]
            log_return[first:first+draws] += log_factors[chosen].sum(axis=1)
    actual = outcomes(selected)
    x = actual["losses"]
    assert math.isclose(loss_law.sum(), 1., abs_tol=1e-10)
    random_net = np.expm1(log_return)*100.
    pct = actual["compound_net_pct"]
    return dict(selected=actual, strata_used=strata, strata_with_choice=variable,
                forced_keeps_in_full_strata=fixed, expected_open=len(selected)-exp_closed,
                expected_losses=float(loss_law @ np.arange(len(loss_law))),
                prob_random_losses_strictly_below_actual=float(loss_law[:x].sum()),
                prob_random_losses_equal_actual=float(loss_law[x]),
                prob_random_losses_at_most_actual=float(loss_law[:x+1].sum()),
                exact_mean_compound_net_pct=100*(exact_factor-1),
                sampled_compound_net_pct=dict(median=float(np.median(random_net)),
                    p05=float(np.quantile(random_net, .05)), p95=float(np.quantile(random_net, .95)),
                    pct_strictly_below_actual=100*float(np.mean(random_net < pct-1e-9)),
                    pct_at_or_below_actual=100*float(np.mean(random_net <= pct+1e-9))))


def main():
    data = json.loads((HERE/"shadow_ledger.json").read_text())
    original = data["trades"][data["original"]]
    archived = json.loads((HERE.parent/"ledger.json").read_text())["trades"]["5"]
    assert [key(t) for t in original] == [key(t) for t in archived]
    original_by_key = {key(t): t for t in original}
    verified = {}
    for name, trades in data["trades"].items():
        actual = [key(t) for t in trades]
        observed = set(actual)
        assert len(observed) == len(actual)
        assert actual == [key(t) for t in original if key(t) in observed]
        for t in trades:
            old = original_by_key[key(t)]
            assert t["exit_reason"] == old["exit_reason"] and abs(t["net"]-old["net"]) < 1e-10
            assert old.get("exit_bar_start") == t.get("exit_bar_start")
        verified[name] = len(trades)
    selected_name = data["selected"]
    selected = data["trades"].get(selected_name, [])
    out = dict(selected=selected_name, sampled_replicates=REPLICATES, seed=SEED,
               original=outcomes(original), verified_subset_entries=verified,
               method="Uniform subsets without replacement of the original, mutually exclusive entry ledger. Losses count closed trades only; net portfolio value also marks final open trades. Loss-count probabilities and expected portfolio wealth are exact.")
    if selected_name is not None:
        out["unstratified"] = reference(original, selected, False)
        out["same_symbol_day_session"] = reference(original, selected, True)
    (HERE/"shadow_random.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
