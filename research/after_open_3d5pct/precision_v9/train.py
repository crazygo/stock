"""One pre-registered mechanism per route, preserving v8 source artifacts."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import torch
from torch import nn

from ..focus_v8 import run as old
from ..focus_v8.core import PROTOCOL, TARGETS, GROUPS, focus_weights, historical_baseline, metrics, score, splits
from ..focus_v8.support import SupportedC, build_c
from ..train_multiscale_v6 import project_monotone, tabular
from .evaluate import (HERE, V8, ROUTES, sha, write, calendar, enrich, choose_thresholds,
                       select_signals, summarize, block_intervals)

CHANGES = {'B_group': 'no_group_categories', 'B_no_group': 'within_stock_rank',
           'C_group': 'group_dropout', 'C_no_group': 'within_stock_rank',
           'C_no_daily': 'capacity'}


def pairs(symbols, labels, cap=8):
    """Fixed same-symbol positive-negative training pairs; no future inputs."""
    a, b = [], []
    rng = np.random.default_rng(9566)
    symbols, labels = np.asarray(symbols), np.asarray(labels)
    for s in np.unique(symbols):
        pos = np.flatnonzero((symbols == s) & (labels == 1))
        neg = np.flatnonzero((symbols == s) & (labels == 0))
        if not len(neg):
            continue
        for i in pos:
            js = rng.choice(neg, min(cap, len(neg)), replace=False)
            a.extend([i]*len(js)); b.extend(js.tolist())
    return np.asarray(a, dtype=int), np.asarray(b, dtype=int)


def rank_gradient(y, z, weights, ij, alpha=.25):
    p = expit(z)
    grad = weights*(p-y)
    hess = weights*p*(1-p)
    i, j = ij
    if len(i):
        q = expit(z[j]-z[i])
        scale = alpha*len(y)/len(i)
        np.add.at(grad, i, -scale*q); np.add.at(grad, j, scale*q)
        # LightGBM consumes a diagonal curvature approximation to pair loss.
        v = scale*q*(1-q)
        np.add.at(hess, i, v); np.add.at(hess, j, v)
    return grad, np.maximum(hess, 1e-6)


def rank_objective_value(y, z, weights, ij, alpha=.25):
    loss = np.sum(weights*(np.logaddexp(0, z)-y*z))
    i, j = ij
    if len(i):
        loss += alpha*len(y)*np.logaddexp(0, z[j]-z[i]).mean()
    return loss


def strip_group_categories(seq):
    out = seq.copy()
    out[..., :3] = 0
    return out


class GroupDropoutC(SupportedC):
    def forward(self, x5, x60, xday, group_seq, prior):
        if self.training:
            keep = (torch.rand((len(group_seq), 1, 1, 1), device=group_seq.device) >= .5)
            group_seq = group_seq * keep
        return super().forward(x5, x60, xday, group_seq, prior)


def model_c(route, recipe):
    return GroupDropoutC(route, recipe) if recipe.get('group_dropout') else build_c(route, recipe)


def fit_c(route, recipe, rows, y, arrays, fold, output, seed):
    torch.set_num_threads(1); torch.manual_seed(seed)
    model = model_c(route, recipe)
    weights = old.training_weights(rows, fold['fit'], recipe, seed)
    if recipe.get('support'):
        model.fit_support(arrays, rows, fold['fit'][weights > 0])
    weight = torch.zeros(len(rows)); weight[fold['fit']] = torch.tensor(weights, dtype=torch.float32)
    target = torch.from_numpy(y)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05 if recipe.get('regularization') else .01)
    rng = np.random.default_rng(seed)
    best, state, bad, trace = math.inf, None, 0, []
    symbols = rows.symbol.to_numpy()
    for epoch in range(PROTOCOL['epochs']):
        model.train(); losses_epoch = []; npairs = 0
        order = rng.permutation(fold['fit'][weights > 0])
        for ids in np.array_split(order, max(1, math.ceil(len(order)/PROTOCOL['batch_size']))):
            opt.zero_grad(set_to_none=True)
            inputs = [torch.empty(0) if (j == 2 and not model.daily) or (j == 3 and not model.group)
                      else a[ids] for j, a in enumerate(arrays)]
            z = model(*inputs)
            per = nn.functional.binary_cross_entropy_with_logits(z, target[ids], reduction='none').mean(1)
            loss = (per*weight[ids]).sum()/weight[ids].sum().clamp_min(1e-6)
            if recipe.get('within_stock_rank'):
                i, j = pairs(symbols[ids], y[ids, 4])
                if len(i):
                    loss = loss + .25*nn.functional.softplus(z[j, 4]-z[i, 4]).mean()
                    npairs += len(i)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
            losses_epoch.append(loss.item())
        p = project_monotone(old.predict_c(model, arrays, fold['tune'])).reshape(-1, 9)
        value = score(rows.iloc[fold['tune']], y[fold['tune']], p)
        trace.append({'epoch': epoch+1, 'loss': float(np.mean(losses_epoch)), 'tune_score': value, 'pairs': npairs})
        if value < best-1e-5:
            best, state, bad, best_epoch = value, copy.deepcopy(model.state_dict()), 0, epoch+1
        else:
            bad += 1
        if bad >= PROTOCOL['patience']:
            break
    model.load_state_dict(state)
    torch.save({'route': route, 'recipe': recipe, 'state_dict': state, 'seed': seed}, output/'model.pt')
    raw = {split: old.predict_c(model, arrays, ids) for split, ids in fold.items()}
    replay = model_c(route, recipe)
    replay.load_state_dict(torch.load(output/'model.pt', weights_only=True)['state_dict'])
    error = float(np.max(abs(old.predict_c(replay, arrays, fold['eval'])-raw['eval'])))
    if error > 1e-7:
        raise AssertionError('neural replay mismatch')
    return raw, {'parameters': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch,
                 'trace': trace, 'replay_max_error': error}


def fit_b(route, recipe, rows, y, prior, xbase, paths, curve, relative, fold, output, seed):
    if not recipe.get('within_stock_rank'):
        return old.fit_b(route, recipe, rows, y, prior, xbase, paths, curve, relative, fold, output, seed)
    x = xbase[:, :585]
    if recipe.get('representation'): x = np.column_stack((x, paths))
    if recipe.get('curves'): x = np.column_stack((x, curve))
    w = old.training_weights(rows, fold['fit'], recipe, seed)
    fitids = fold['fit'][w > 0]; w = w[w > 0]
    tw = focus_weights(rows.iloc[fold['tune']]); tw /= tw.mean()
    ij = pairs(rows.iloc[fitids].symbol.to_numpy(), y[fitids, 4])
    raw = {s: np.zeros((len(ids), 9)) for s, ids in fold.items()}
    trees, error = [], 0.
    for k in range(9):
        args = dict(n_estimators=120, num_leaves=7, max_depth=4, min_child_samples=150,
                    learning_rate=.035, reg_lambda=30, n_jobs=1, verbosity=-1, random_state=seed)
        if k == 4:
            def objective(yy, zz):
                return rank_gradient(yy, zz, w, ij)
            args.update(objective=objective, metric='None')
            def metric(yy, zz):
                return 'weighted_binary_logloss', float(np.average(np.logaddexp(0, zz)-yy*zz, weights=tw)), False
            extra = {'eval_metric': metric}
        else:
            extra = {'sample_weight': w, 'eval_sample_weight': [tw]}
        model = lgb.LGBMClassifier(**args)
        model.fit(x[fitids], y[fitids, k], eval_set=[(x[fold['tune']], y[fold['tune'], k])],
                  callbacks=[lgb.early_stopping(20, verbose=False)], **extra)
        model.booster_.save_model(str(output/f'target_{k}.txt'))
        replay = lgb.Booster(model_file=str(output/f'target_{k}.txt'))
        for split, ids in fold.items():
            z = model.booster_.predict(x[ids], raw_score=True, num_threads=1)
            zz = replay.predict(x[ids], raw_score=True, num_threads=1)
            error = max(error, float(np.max(abs(z-zz))))
            raw[split][:, k] = expit(z)
        trees.append(model.booster_.num_trees())
    if error > 1e-7: raise AssertionError('tree replay mismatch')
    return raw, {'features': x.shape[1], 'trees': trees, 'training_rank_pairs': len(ij[0]), 'replay_max_error': error}


def fit_one(route, recipe, loaded, fold_config, output):
    output.mkdir(parents=True, exist_ok=False)
    rows, data, y, prior, counts, paths, xbase, curve, relative, arrays, identity = loaded
    fold = splits(rows, fold_config)
    if recipe.get('no_group_categories'):
        changed = {**data, 'group_seq': strip_group_categories(data['group_seq'])}
        xbase = tabular(changed, group=True).to_numpy(np.float32)
    write(output/'registration.json', {'route': route, 'recipe': recipe, 'fold': fold_config,
          'seed': 3566, 'dataset': identity, 'split_counts': {k: len(v) for k,v in fold.items()},
          'code_hashes': {str(p): sha(p) for p in [*HERE.glob('*.py'), * (HERE.parent/'focus_v8').glob('*.py'),
                                                  HERE.parent/'train_multiscale_v6.py', HERE.parent/'v6_data.py']}})
    started = time.monotonic()
    if route.startswith('C'):
        raw, info = fit_c(route, recipe, rows, y, arrays, fold, output, 3566)
    else:
        raw, info = fit_b(route, recipe, rows, y, prior, xbase, paths, curve, relative, fold, output, 3566)
    params = old.calibrate(raw['cal'], y[fold['cal']], focus_weights(rows.iloc[fold['cal']])) if recipe.get('calibration') else [[1.,0.]]*9
    write(output/'calibration.json', {'parameters': params, 'method': 'v8_shrunken_platt' if recipe.get('calibration') else 'identity'})
    results = {}
    for split, ids in fold.items():
        p = project_monotone(old.apply_cal(raw[split], params)).reshape(-1, 9)
        hist = historical_baseline(rows, y, fold['fit'], ids)
        frame = rows.iloc[ids][['sample_id', 'symbol', 'session_date', 'decision_at']].reset_index(drop=True)
        for j, name in enumerate(TARGETS):
            frame['y_'+name] = y[ids, j]; frame['raw_'+name] = raw[split][:, j]
            frame['p_'+name] = p[:, j]; frame['history_'+name] = hist[:, j]
        frame.to_parquet(output/f'{split}.parquet', index=False)
        results[split] = {'score': score(rows.iloc[ids], y[ids], p), 'groups': metrics(rows.iloc[ids], y[ids], p)}
    result = {'route': route, 'recipe': recipe, 'fold': fold_config['id'], 'results': results,
              'fit': info, 'seconds': time.monotonic()-started}
    write(output/'result.json', result)
    print(json.dumps({'route': route, 'fold': fold_config['id'], 'seconds': round(result['seconds'],1),
                      'eval_brier': results['eval']['score']}), flush=True)
    return result


def evaluate_candidate(path, rows, fold):
    raw_calendar, sessions = calendar()
    cal = enrich(pd.read_parquet(path/'cal.parquet'), rows, sessions)
    ev = enrich(pd.read_parquet(path/'eval.parquet'), rows, sessions)
    cdates = [s['session_date'] for s in raw_calendar if cal.session_date.min() <= s['session_date'] <= cal.session_date.max()]
    edates = [s['session_date'] for s in raw_calendar if fold['eval'] <= s['session_date'] < fold['end']]
    thresholds, curves = choose_thresholds(cal, cdates)
    write(path/'selection.json', {'thresholds': thresholds, 'cal_curves': curves})
    signals = select_signals(ev, thresholds)
    signals.to_parquet(path/'signals.parquet', index=False)
    result = {g: {**summarize(ev, signals, g, edates), 'threshold': thresholds[g],
                  'intervals': block_intervals(signals, g, edates)} for g in GROUPS}
    write(path/'precision.json', result)
    return result


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    deps = sorted(set([*HERE.glob('*.py'), *HERE.parent.glob('*.py'),
                       *(HERE.parent/'focus_v8').glob('*.py'),
                       *(HERE.parent/'iterations_v7').glob('*/experiment.py')]))
    identity = {str(p): sha(p) for p in [* (source/'dataset').iterdir(),
                source/'final_selection.json', source/'round_10/summary.json'] if p.is_file()}
    write(output/'registration.json', {'registered_at': datetime.now().astimezone().isoformat(),
          'changes': CHANGES, 'source_identity': identity, 'code_hashes': {str(p): sha(p) for p in deps},
          'protocol': sha(HERE/'PROTOCOL.md'), 'backlog': sha(HERE/'rounds/R01/BACKLOG.md'),
          'matrix': sha(HERE/'rounds/R01/MATRIX.md'), 'independent_validation': False})
    work = output/'data_cache'; work.mkdir()
    # Copy immutable inputs/cache; load_data may write identities only in this new run.
    shutil.copytree(source/'dataset', work/'dataset')
    for name in ['dataset_identity.json', 'path_summary.npy', 'path_summary_identity.json']:
        if (source/name).exists(): shutil.copy2(source/name, work/name)
    loaded = old.load_data(work)
    previous = json.loads((source/'round_10/summary.json').read_text())
    rows = loaded[0]
    results = {}
    for route in ROUTES:
        recipe = {**previous[route]['incumbent_recipe'], CHANGES[route]: True}
        results[route] = {'recipe': recipe, 'folds': {}}
        for fold in V8['folds'][:2]:
            path = output/route/fold['id']
            fit = fit_one(route, recipe, loaded, fold, path)
            precision = evaluate_candidate(path, rows, fold)
            results[route]['folds'][fold['id']] = {'score': fit['results']['eval']['score'], 'precision': precision}
        old_paths = [Path(p) for p in previous[route]['incumbent_paths']]
        old_metrics = [json.loads((p/'result.json').read_text())['results']['eval'] for p in old_paths]
        new_metrics = [json.loads((output/route/f['id']/'result.json').read_text())['results']['eval'] for f in V8['folds'][:2]]
        old_brier = np.mean([m['score'] for m in old_metrics]); new_brier = np.mean([m['score'] for m in new_metrics])
        old_auc = np.mean([m['groups'][g][PRIMARY]['within_auc'] for m in old_metrics for g in GROUPS])
        new_auc = np.mean([m['groups'][g][PRIMARY]['within_auc'] for m in new_metrics for g in GROUPS])
        r0 = json.loads((output.parent/'R00/metrics.json').read_text())
        old_pass = sum(x['point_and_supply_pass'] for x in r0 if x['route']==route and x['method']=='cal_selected' and x['fold']!='reserved')
        new_pass = sum(m['point_and_supply_pass'] for f in results[route]['folds'].values() for m in f['precision'].values())
        keep = new_pass > old_pass or (new_pass == old_pass and new_auc-old_auc >= .02 and new_brier-old_brier <= .005)
        decision = {'keep_for_development': bool(keep), 'old_pass_cells': old_pass, 'new_pass_cells': new_pass,
                    'old_brier': float(old_brier), 'new_brier': float(new_brier),
                    'old_within_auc': float(old_auc), 'new_within_auc': float(new_auc)}
        results[route]['decision'] = decision
        write(output/route/'decision_before_diagnostic.json', decision)
        fold = V8['folds'][2]
        path = output/route/fold['id']
        fit = fit_one(route, recipe, loaded, fold, path)
        results[route]['folds'][fold['id']] = {'score': fit['results']['eval']['score'], 'precision': evaluate_candidate(path, rows, fold)}
        write(output/'progress.json', results)
    if any(sha(Path(p)) != h for p,h in identity.items()):
        raise ValueError('source modified during experiment')
    write(output/'summary.json', {'routes': results, 'source_unchanged': True, 'goal_complete': False,
          'formal_passing_routes': [], 'independent_validation': False})


PRIMARY = '3d_5pct'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=HERE.parent/'runs/focus_v8_20260927')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
