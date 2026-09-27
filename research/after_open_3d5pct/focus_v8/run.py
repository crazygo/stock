"""One registered adaptive round at a time; final period has a separate command."""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
import torch
from torch import nn

from research.after_open_3d5pct.train_multiscale_v6 import tabular, project_monotone
from research.after_open_3d5pct.iterations_v7.b_group.experiment import curve_features, group_relative_features
from research.after_open_3d5pct.v6_data import digest
from .core import (HERE, PROTOCOL, GROUPS, TARGETS, FocusC, focus_weights, historical_baseline,
                   mature_prior, metrics, path_summary, score, splits)
from .prepare import write
from .support import build_c

MECHANISMS = {
    "balance": "三组等权训练质量占75%，全池保留25%迁移样本",
    "representation": "C用相对价格坐标放大收益信号；B补跨尺度完整路径摘要",
    "prior": "仅用当时已完整成熟的本股九任务历史概率，C/树学习logit残差",
    "recency": "训练样本按90自然日半衰期衰减，适应市场状态变化",
    "capacity": "C跨patch卷积学习时序相互作用；B增至15叶/240树并早停",
    "curves": "C增加开盘/小时/日线的分段路径、回撤和效率；B加入开盘精细路径及允许的群相对差",
    "calibration": "独立时间校准段拟合向identity收缩的非负Platt映射",
    "regularization": "更强权重惩罚与dropout/最小叶子样本，限制小样本过拟合",
    "specialize": "仅在三组股票并集拟合，沿用全池构建的允许输入",
    "bagging": "训练日期的固定随机权重扰动，检验对单个市场日期依赖",
    "support": "仅C路线：屏蔽fit中从未出现的日线位置及不足10个fit日期出现的群类型",
}


def load_data(root):
    ds = root / "dataset"
    identity = {name: digest(ds/name) for name in ("features.npz", "rows.parquet", "manifest.json")}
    identity_file = root/"dataset_identity.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError("frozen dataset contents changed")
    if not identity_file.exists():
        write(identity_file, identity)
    rows = pd.read_parquet(ds / "rows.parquet")
    with np.load(ds / "features.npz") as z:
        data = {k: z[k] for k in ("x5", "x60", "xday", "group_seq", "y")}
    y = data["y"].reshape(-1, 9)
    prior, counts = mature_prior(rows, y)
    cache = root / "path_summary.npy"
    cache_id = {"dataset": identity, "transform": digest(HERE/"core.py")}
    cache_meta = root/"path_summary_identity.json"
    if cache.exists() and cache_meta.exists():
        stored = json.loads(cache_meta.read_text())
        stored_hash = stored.pop("cache_sha256", None)
        if stored != cache_id:
            raise ValueError("path summary cache identity mismatch")
        paths = np.load(cache)
        if stored_hash is not None and digest(cache) != stored_hash:
            raise ValueError("path summary cache contents changed")
        if stored_hash is None:
            if not np.array_equal(paths, path_summary(data)):
                raise ValueError("legacy path cache contents changed")
            write(cache_meta, {**cache_id, "cache_sha256": digest(cache)})
    else:
        paths = path_summary(data)
        if cache.exists() and not np.array_equal(np.load(cache), paths):
            raise ValueError("legacy cache differs from reproducible transform")
        np.save(cache, paths)
        write(cache_meta, {**cache_id, "cache_sha256": digest(cache)})
    xbase = tabular(data, group=True).to_numpy(np.float32)
    curve = curve_features(data).to_numpy(np.float32)
    relative = group_relative_features(data).to_numpy(np.float32)
    arrays = tuple(torch.from_numpy(data[k]) for k in ("x5", "x60", "xday", "group_seq")) + (torch.from_numpy(prior),)
    return rows, data, y, prior, counts, paths, xbase, curve, relative, arrays, identity


def predict_c(model, arrays, ids):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in np.array_split(ids, max(1, math.ceil(len(ids)/256))):
            inputs = [torch.empty(0) if (j == 2 and not model.daily) or (j == 3 and not model.group)
                      else a[batch] for j, a in enumerate(arrays)]
            out.append(torch.sigmoid(model(*inputs)).numpy())
    return np.concatenate(out)


def training_weights(rows, ids, recipe, seed):
    r = rows.iloc[ids]
    w = np.ones(len(ids)) / len(ids)
    focus = focus_weights(r)
    if recipe.get("balance"):
        w = .25*w + .75*focus
    if recipe.get("specialize"):
        w *= focus > 0
    if recipe.get("recency"):
        age = (pd.to_datetime(r.session_date).max()-pd.to_datetime(r.session_date)).dt.days.to_numpy()
        w *= 2**(-age/90)
    if recipe.get("bagging"):
        rng = np.random.default_rng(seed)
        dates = sorted(r.session_date.unique())
        weights = dict(zip(dates, rng.exponential(1, len(dates))))
        w *= r.session_date.map(weights).to_numpy()
    return w / w.mean()


def fit_c(route, recipe, rows, y, arrays, fold, output, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = build_c(route, recipe)
    weights = training_weights(rows, fold["fit"], recipe, seed)
    if recipe.get("support"):
        model.fit_support(arrays, rows, fold["fit"][weights > 0])
    weight = torch.zeros(len(rows))
    weight[fold["fit"]] = torch.tensor(weights, dtype=torch.float32)
    target = torch.from_numpy(y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05 if recipe.get("regularization") else .01)
    rng = np.random.default_rng(seed)
    best, state, bad, trace = math.inf, None, 0, []
    for epoch in range(PROTOCOL["epochs"]):
        model.train()
        order = rng.permutation(fold["fit"][weights > 0])
        loss_sum = []
        for ids in np.array_split(order, max(1, math.ceil(len(order)/PROTOCOL["batch_size"]))):
            optimizer.zero_grad(set_to_none=True)
            inputs = [torch.empty(0) if (j == 2 and not model.daily) or (j == 3 and not model.group)
                      else a[ids] for j, a in enumerate(arrays)]
            logit = model(*inputs)
            losses = nn.functional.binary_cross_entropy_with_logits(logit, target[ids], reduction="none").mean(1)
            loss = (losses*weight[ids]).sum()/weight[ids].sum().clamp_min(1e-6)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss_sum.append(loss.item())
        p = project_monotone(predict_c(model, arrays, fold["tune"])).reshape(-1, 9)
        value = score(rows.iloc[fold["tune"]], y[fold["tune"]], p)
        trace.append({"epoch": epoch+1, "weighted_loss": float(np.mean(loss_sum)), "tune_score": value})
        if value < best-1e-5:
            best, state, bad = value, copy.deepcopy(model.state_dict()), 0
            best_epoch = epoch+1
        else:
            bad += 1
        if bad >= PROTOCOL["patience"]:
            break
    model.load_state_dict(state)
    torch.save({"route": route, "recipe": recipe, "state_dict": state, "seed": seed}, output / "model.pt")
    raw = {split: predict_c(model, arrays, ids) for split, ids in fold.items()}
    replay = build_c(route, recipe)
    replay.load_state_dict(torch.load(output / "model.pt", weights_only=True)["state_dict"])
    error = float(np.max(abs(predict_c(replay, arrays, fold["eval"])-raw["eval"])))
    if error > 1e-7:
        raise AssertionError("checkpoint replay mismatch")
    info = {"parameters": sum(p.numel() for p in model.parameters()), "best_epoch": best_epoch, "trace": trace, "replay_max_error": error}
    if recipe.get("support"):
        info["day_support"] = model.day_support.tolist() if model.daily else None
        info["group_support"] = model.group_support.tolist() if model.group else None
    return raw, info


def fit_b(route, recipe, rows, y, prior, xbase, paths, curve, relative, fold, output, seed):
    x = xbase if route == "B_group" else xbase[:, :585]
    if recipe.get("representation"):
        x = np.column_stack((x, paths))
    if recipe.get("curves"):
        x = np.column_stack((x, curve))
        if route == "B_group":
            x = np.column_stack((x, relative))
    weight = training_weights(rows, fold["fit"], recipe, seed)
    fitids = fold["fit"][weight > 0]
    weight = weight[weight > 0]
    tuneweight = focus_weights(rows.iloc[fold["tune"]])
    tuneweight /= tuneweight.mean()
    raw = {split: np.zeros((len(ids), 9)) for split, ids in fold.items()}
    trees, replay_error = [], 0.
    for j in range(9):
        model = lgb.LGBMClassifier(n_estimators=240 if recipe.get("capacity") else 120,
            num_leaves=15 if recipe.get("capacity") else 7, max_depth=5 if recipe.get("capacity") else 4,
            min_child_samples=150 if recipe.get("regularization") else 50,
            learning_rate=.035, reg_lambda=30 if recipe.get("regularization") else 10,
            n_jobs=1, verbosity=-1, random_state=seed)
        kwargs = {}
        if recipe.get("prior"):
            kwargs = {"init_score": scipy.special.logit(prior[fitids, j].clip(.01, .99)),
                      "eval_init_score": [scipy.special.logit(prior[fold["tune"], j].clip(.01, .99))]}
        model.fit(x[fitids], y[fitids, j], sample_weight=weight,
                  eval_set=[(x[fold["tune"]], y[fold["tune"], j])], eval_sample_weight=[tuneweight],
                  callbacks=[lgb.early_stopping(20, verbose=False)], **kwargs)
        model.booster_.save_model(str(output / f"target_{j}.txt"))
        replay = lgb.Booster(model_file=str(output / f"target_{j}.txt"))
        for split, ids in fold.items():
            z = model.booster_.predict(x[ids], raw_score=True, num_threads=1)
            zz = replay.predict(x[ids], raw_score=True, num_threads=1)
            replay_error = max(replay_error, float(np.max(abs(z-zz))))
            if recipe.get("prior"):
                z += scipy.special.logit(prior[ids, j].clip(.01, .99))
            raw[split][:, j] = scipy.special.expit(z)
        trees.append(model.booster_.num_trees())
    if replay_error > 1e-7:
        raise AssertionError("tree checkpoint replay mismatch")
    return raw, {"features": x.shape[1], "trees": trees, "replay_max_error": replay_error}


def calibrate(p, y, weights):
    result = []
    weights = weights/weights.sum()
    for j in range(9):
        x = scipy.special.logit(p[:, j].clip(1e-5, 1-1e-5))
        def objective(v):
            z = x*v[0]+v[1]
            return np.sum(weights*(np.logaddexp(0,z)-y[:, j]*z)) + .01*((v[0]-1)**2+v[1]**2)
        fit = scipy.optimize.minimize(objective, [1, 0], bounds=[(.1, 3), (-2, 2)])
        if not fit.success:
            raise RuntimeError("calibration failed: " + fit.message)
        result.append(fit.x.tolist())
    return result


def apply_cal(p, params):
    a = np.asarray(params)
    return scipy.special.expit(scipy.special.logit(p.clip(1e-5, 1-1e-5))*a[:, 0]+a[:, 1])


def fit_one(root, route, recipe, fold_config, loaded, seed, output):
    result_file = output / "result.json"
    rows, data, y, prior, counts, paths, xbase, curve, relative, arrays, identity = loaded
    fold = splits(rows, fold_config)
    deps = [*HERE.glob("*.py"), HERE.parent/"train_multiscale_v6.py", HERE.parent/"v6_data.py",
            HERE.parent/"iterations_v7/b_group/experiment.py", HERE.parent/"iterations_v7/c_no_daily/experiment.py"]
    registration = {"recipe": recipe, "route": route, "fold": fold_config, "seed": seed,
          "split_rows": {s: len(ids) for s, ids in fold.items()}, "protocol_hash": digest(HERE/"protocol.json"),
          "code_hashes": {str(p.relative_to(HERE.parent)): digest(p) for p in deps}, "dataset_identity": identity}
    if result_file.exists():
        if json.loads((output/"registration.json").read_text()) != registration:
            raise ValueError("existing fit has different code/data/recipe identity; retain it and use a new output")
        return json.loads(result_file.read_text())
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write(output / "registration.json", registration)
    if route.startswith("C"):
        raw, info = fit_c(route, recipe, rows, y, arrays, fold, output, seed)
    else:
        raw, info = fit_b(route, recipe, rows, y, prior, xbase, paths, curve, relative, fold, output, seed)
    params = [[1., 0.]]*9
    if recipe.get("calibration"):
        params = calibrate(raw["cal"], y[fold["cal"]], focus_weights(rows.iloc[fold["cal"]]))
    write(output/"calibration.json", {"method": "shrunken_nonnegative_platt" if recipe.get("calibration") else "identity_raw_estimate", "parameters": params})
    results = {}
    for split, ids in fold.items():
        p = project_monotone(apply_cal(raw[split], params)).reshape(-1, 9)
        hist = historical_baseline(rows, y, fold["fit"], ids)
        frame = rows.iloc[ids][["sample_id", "symbol", "session_date", "decision_at"]].reset_index(drop=True).copy()
        for j, name in enumerate(TARGETS):
            frame["y_"+name] = y[ids, j]
            frame["p_"+name] = p[:, j]
            frame["raw_"+name] = raw[split][:, j]
            frame["history_"+name] = hist[:, j]
        frame.to_parquet(output/f"{split}.parquet", index=False)
        results[split] = {"score": score(rows.iloc[ids], y[ids], p), "groups": metrics(rows.iloc[ids], y[ids], p),
                         "raw_groups": metrics(rows.iloc[ids], y[ids], project_monotone(raw[split]).reshape(-1,9)),
                         "history_groups": metrics(rows.iloc[ids], y[ids], hist),
                         "rows": len(ids), "dates": rows.iloc[ids].session_date.nunique(),
                         "full_daily_history_fraction": float(rows.iloc[ids].full_126_prior_days.mean()),
                         "mature_prior_count_median": float(np.median(counts[ids]))}
    result = {"route": route, "recipe": recipe, "fold": fold_config["id"], "seed": seed,
              "seconds": time.monotonic()-started, "fit": info, "results": results,
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "lightgbm": lgb.__version__}}
    write(result_file, result)
    print(json.dumps({"route": route, "fold": fold_config["id"], "recipe": recipe, "seconds": round(result["seconds"], 1), "score": results["eval"]["score"]}), flush=True)
    return result


def pugh(incumbent, remaining, prior_results):
    vals = [r["results"]["eval"]["groups"][g]["3d_5pct"] for r in prior_results for g in GROUPS]
    gap = float(np.mean([x["mean_gap"] for x in vals])) if vals else .1
    timing = float(np.mean([x["within_auc"] for x in vals if x["within_auc"] is not None])) if vals else .5
    # Five explicit axes vs incumbent: calibration, timing, stability, data risk, cost.
    matrix = {
        "balance": [1, 1, 1, 0, 1], "representation": [1, 1, 1, 0, 0],
        "prior": [1, 1, 1, 0, 0], "recency": [1, 0, 1, -1, 1],
        "capacity": [0, 1, 0, -1, -1], "curves": [0, 1, 1, 0, 0],
        "calibration": [1, 0, 1, -1, 1], "regularization": [0, 0, 1, 1, 1],
        "specialize": [1, 1, 0, -1, 1], "bagging": [0, 0, 1, 1, 0],
        "support": [1, 0, 1, 1, 1],
    }
    if gap < .05:
        for name in ("balance", "prior", "recency", "calibration", "specialize"):
            matrix[name][0] = 0
    if timing < .55:
        matrix["capacity"][2] = 1
        matrix["curves"][2] = 1
    if prior_results:
        overfit = np.mean([r["results"]["eval"]["score"]-r["results"]["fit"]["score"] for r in prior_results])
        if overfit > .04:
            matrix["regularization"][0] = 1
            matrix["bagging"][0] = 1
            matrix["capacity"][2] = -1
    weights = np.array([3, 3, 2, 2, 1])
    comparisons = [{"mechanism": m, "description": MECHANISMS[m], "signs": matrix[m], "weighted_score": int(np.dot(matrix[m], weights))} for m in remaining]
    comparisons.sort(key=lambda x: -x["weighted_score"])
    return {"datum": incumbent, "diagnosis": {"mean_absolute_group_gap": gap, "mean_within_stock_auc": timing},
            "axes": ["probability_bias", "within_stock_timing", "cross_period_stability", "sample_overfit_risk", "compute_cost"],
            "weights": weights.tolist(), "comparisons": comparisons, "chosen": comparisons[0]["mechanism"]}


def run_round(root, round_number):
    if not 0 <= round_number <= PROTOCOL["max_rounds"]:
        raise ValueError("outside registered round cap")
    rundir = root/f"round_{round_number:02d}"
    if (rundir/"summary.json").exists():
        raise FileExistsError("round already complete")
    previous = json.loads((root/f"round_{round_number-1:02d}/summary.json").read_text()) if round_number else None
    registration = {}
    for route in PROTOCOL["routes"]:
        prev = previous[route] if previous else None
        recipe = dict(prev["incumbent_recipe"]) if prev else {}
        tested = prev["tested"] if prev else []
        if round_number:
            results = [json.loads((Path(p)/"result.json").read_text()) for p in prev["incumbent_paths"]]
            remaining = [m for m in MECHANISMS if m not in tested and
                         (m != "support" or (round_number == 10 and route.startswith("C")))]
            matrix = pugh(recipe, remaining, results)
            recipe[matrix["chosen"]] = True
            tested = tested+[matrix["chosen"]]
        else:
            matrix = {"chosen": "matched_R0", "reason": "same expanded dataset and historical folds; no old future-trained checkpoints"}
        registration[route] = {"candidate_recipe": recipe, "tested": tested, "pugh": matrix}
    # Registration precedes loading/fitting; refuse silently changing a partial round.
    regpath = rundir/"registration.json"
    if regpath.exists() and json.loads(regpath.read_text()) != registration:
        raise ValueError("partial round registration mismatch")
    write(regpath, registration)
    loaded = load_data(root)
    summary = {}
    for route, reg in registration.items():
        paths = [rundir/route/f["id"] for f in PROTOCOL["folds"][:2]]
        results = [fit_one(root, route, reg["candidate_recipe"], f, loaded, PROTOCOL["seed"], p) for f, p in zip(PROTOCOL["folds"][:2], paths)]
        value = float(np.mean([r["results"]["eval"]["score"] for r in results]))
        prev = previous[route] if previous else None
        keep = prev is None or value < prev["incumbent_score"]-1e-5
        summary[route] = {**reg, "candidate_score": value, "accepted": keep,
                          "incumbent_score": value if keep else prev["incumbent_score"],
                          "incumbent_recipe": reg["candidate_recipe"] if keep else prev["incumbent_recipe"],
                          "incumbent_round": round_number if keep else prev["incumbent_round"],
                          "incumbent_paths": [str(p) for p in paths] if keep else prev["incumbent_paths"]}
    write(rundir/"summary.json", summary)
    print(json.dumps({r: {k: v[k] for k in ("candidate_score", "accepted", "incumbent_round")} for r, v in summary.items()}), flush=True)


def final(root):
    rounds = sorted(root.glob("round_*/summary.json"))
    if len(rounds) != 11:
        raise ValueError("final scoring requires all ten rounds unless a separately recorded early gate passes")
    chosen = json.loads(rounds[-1].read_text())
    frozen = {r: {"recipe": v["incumbent_recipe"], "round": v["incumbent_round"], "score": v["incumbent_score"]} for r, v in chosen.items()}
    path = root/"final_selection.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError("cannot revise final selection")
    write(path, frozen)
    loaded = load_data(root)
    fold = PROTOCOL["folds"][2]
    for route in PROTOCOL["routes"]:
        fit_one(root, route, {}, fold, loaded, PROTOCOL["seed"], root/"final"/route/"baseline")
        for seed in (PROTOCOL["seed"], PROTOCOL["confirmation_seed"]):
            fit_one(root, route, frozen[route]["recipe"], fold, loaded, seed, root/"final"/route/f"selected_{seed}")
    write(root/"final_complete.json", {"status": "complete_development_only", "selected": frozen})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--round", type=int)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.final:
        final(args.root.resolve())
    else:
        run_round(args.root.resolve(), args.round)
