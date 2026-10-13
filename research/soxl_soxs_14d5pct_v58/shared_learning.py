"""Date-normalized, mean-regularized base models across causal decision clocks."""
from pathlib import Path
import sys
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v57'))
import short_learning as previous
from short_learning import (core, np, pd, read, dump, sha, labels, day_cutoff, restrict_market,
    development_gate, tag, registry, fit_day, issue, decide, CLOCKS, OLD_CLOCKS,
    load_sources, load_panel, PANEL_ROOT, original_fit)
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
VARIANTS = ['partial', 'independent', 'shared']
ROUTES = ['partial_band', 'partial_point', 'independent_band', 'shared_band',
          'source_v56', 'source_v55', 'premarket_only', 'cash', 'double_cost']


def columns(clock):
    return previous.columns(clock)


def old55_columns(clock):
    return previous.prior.prior.columns(clock, 'full')


def verified_source():
    source, source_sha = previous.verified_source()
    prior = core.ROOT / 'research/soxl_soxs_14d5pct_v57/evidence/delivery_manifest.json'
    m = read(prior)
    for f, h in m['delivery_files_sha256'].items():
        assert sha(prior.parents[1] / f) == h, f
    v = read(prior.parent / 'verification.json')
    assert v['passed'] and v['results_sha256'] == sha(prior.parent / 'results.json')
    return source, source_sha, sha(prior)


def base_pool(reg, at):
    cutoff = day_cutoff(at); today = pd.Timestamp(at.tz_convert(core.ET).date())
    prior = [r for r in reg if r['feature_ready'] and r['date'] < today]
    parents = []
    for row in prior:
        if row['release'] is None or pd.isna(row['release']) or row['release'] >= cutoff:
            break
        parents.append(row)
    pending = prior[len(parents)] if len(parents) < len(prior) else None
    fb = {'prior_feature_ready_dates': len(prior), 'selected_parent_dates': len(parents),
          'first_pending_date': pending['date'] if pending else None,
          'expired_unresolved_prefix': bool(pending and pending['date'] + pd.Timedelta(days=14) <= today)}
    return parents, fb


def parameter_layout(cols, kind):
    names = list(dict.fromkeys(f for fields in cols for f in fields))
    if kind == 'shared':
        indices = [[names.index(f) for f in fields] for fields in cols]
        return indices, len(names), names
    assert kind in ['partial', 'independent']
    indices = []; offset = 0
    for fields in cols:
        indices.append(list(range(offset, offset + len(fields)))); offset += len(fields)
    return indices, offset, names


def penalty(cols, eta, kind, strength):
    mapping, size, names = parameter_layout(cols, kind)
    out = np.zeros((size, size))
    for f in names:
        active = [c for c, fields in enumerate(cols) if f in fields]
        weights = np.array([eta[c] for c in active]); total = weights.sum()
        ids = [mapping[c][cols[c].index(f)] for c in active]
        if kind == 'shared':
            assert len(set(ids)) == 1
            out[ids[0], ids[0]] += strength * total
        else:
            rho = 4. if kind == 'partial' else 1.
            local = strength * (rho * np.diag(weights) + (1 - rho) * np.outer(weights, weights) / total)
            out[np.ix_(ids, ids)] += local
    return out


def prepare_arrays(panel, parents):
    n = len(parents); ids = [t for p in parents for t in p['members']]
    dw = 2. ** (-np.arange(n)[::-1] / 126.)
    if n: dw *= n / dw.sum()
    rw = np.concatenate([np.repeat(dw[j] / len(p['members']), len(p['members'])) for j, p in enumerate(parents)]) if n else np.array([])
    rows = panel.loc[ids]; counts = [int(rows.clock.eq(c).sum()) for c in CLOCKS]
    out = {'parent_weights': dw.tolist(), 'row_weights': rw.tolist(), 'members': ids,
           'clock_rows': counts, 'dates': [p['date'] for p in parents],
           'ready': bool(n >= 120 and min(counts) >= 120), 'N': n}
    if not out['ready']: return out, None
    zs = []; ws = []; cs = []; ds = []; cols = []; centers = []; scales = []; eta = []
    for clock in CLOCKS:
        mask = rows.clock.eq(clock).to_numpy(); f = columns(clock); x = np.ascontiguousarray(rows.loc[mask, f].to_numpy(float)); w = rw[mask]
        center = np.average(x, axis=0, weights=w)
        scale = np.sqrt(np.average((x - center) ** 2, axis=0, weights=w)); scale[scale < 1e-12] = 1.
        z = (x - center) / scale
        assert np.isfinite(z).all()
        zs.append(z); ws.append(w); cs.append(rows.loc[mask, 'C'].to_numpy(float)); ds.append(rows.loc[mask, 'D'].to_numpy(float))
        cols.append(f); centers.append(center.tolist()); scales.append(scale.tolist()); eta.append(float(w.sum() / n))
    out.update(columns=cols, center=centers, scale=scales, eta=eta)
    return out, {'Z': zs, 'weights': ws, 'C': cs, 'D': ds, 'columns': cols, 'eta': eta, 'N': n}


def regression(arrays, kind, target):
    cols = arrays['columns']; mapping, size, _ = parameter_layout(cols, kind)
    matrix = penalty(cols, arrays['eta'], kind, 100.); rhs = np.zeros(size); means = []
    for c, (z, w) in enumerate(zip(arrays['Z'], arrays['weights'])):
        y = arrays['C'][c] if target == 'C' else abs(arrays['D'][c])
        zm = np.average(z, axis=0, weights=w); ym = float(np.average(y, weights=w)); a = z - zm
        ids = mapping[c]; matrix[np.ix_(ids, ids)] += a.T @ (w[:, None] * a); rhs[ids] += a.T @ (w * (y - ym))
        means.append((zm, ym))
    coef = np.linalg.solve(matrix, rhs)
    intercept = [float(ym - zm @ coef[mapping[c]]) for c, (zm, ym) in enumerate(means)]
    return {'coefficients': coef.tolist(), 'intercepts': intercept}


def classification_arrays(arrays):
    weights = []; active = []; fallback = []
    for d, w in zip(arrays['D'], arrays['weights']):
        amount = w * abs(d)
        if amount.sum() > 0: amount *= w.sum() / amount.sum()
        q = float(np.average(d > 0, weights=amount)) if amount.sum() else .5
        mixed = bool(amount.sum() > 0 and len(np.unique((d[amount > 0] > 0).astype(int))) == 2)
        weights.append(amount); active.append(mixed); fallback.append(q)
    return weights, active, fallback


def logistic_objective(theta, arrays, kind, quadratic, mapping, weights, active, need_hessian=True):
    size = len(quadratic); coef = theta[:size]; active_clocks = [c for c, enabled in enumerate(active) if enabled]
    gradient = np.zeros(len(theta)); gradient[:size] = quadratic @ coef
    hessian = np.zeros((len(theta), len(theta))) if need_hessian else None
    if need_hessian: hessian[:size, :size] = quadratic
    value = .5 * float(coef @ quadratic @ coef)
    for i, c in enumerate(active_clocks):
        z = arrays['Z'][c]; w = weights[c]; y = (arrays['D'][c] > 0).astype(float); ids = mapping[c]; bias = size + i
        linear = z @ coef[ids] + theta[bias]; prob = expit(linear)
        value += float(w @ (np.logaddexp(0, linear) - y * linear))
        residual = w * (prob - y); gradient[ids] += z.T @ residual; gradient[bias] = residual.sum()
        if need_hessian:
            curvature = w * prob * (1 - prob); cross = z.T @ curvature
            hessian[np.ix_(ids, ids)] += z.T @ (curvature[:, None] * z)
            hessian[ids, bias] += cross; hessian[bias, ids] += cross; hessian[bias, bias] = curvature.sum()
    return value, gradient, hessian


def logistic(arrays, kind):
    mapping, size, _ = parameter_layout(arrays['columns'], kind)
    quadratic = penalty(arrays['columns'], arrays['eta'], kind, 10.)
    weights, active, fallback = classification_arrays(arrays)
    theta = np.r_[np.zeros(size), [np.log(q / (1 - q)) for q, mixed in zip(fallback, active) if mixed]]
    tolerance = 1e-9 * max(1, arrays['N'])
    for iteration in range(81):
        value, gradient, hessian = logistic_objective(theta, arrays, kind, quadratic, mapping, weights, active)
        if np.max(abs(gradient)) <= tolerance: break
        assert iteration < 80, 'Convex logistic Newton fit did not converge'
        step = np.linalg.solve(hessian, gradient); descent = float(gradient @ step); fraction = 1.
        assert descent > 0 and np.isfinite(step).all()
        for attempt in range(41):
            candidate = theta - fraction * step
            trial = logistic_objective(candidate, arrays, kind, quadratic, mapping, weights, active, False)[0]
            if trial <= value - 1e-4 * fraction * descent + 1e-12: break
            fraction *= .5
        else: raise AssertionError('Logistic line search failed')
        theta = candidate
    intercepts = np.zeros(len(active)); intercepts[np.array(active, bool)] = theta[size:]
    return {'coefficients': theta[:size].tolist(), 'intercepts': intercepts.tolist(),
            'mixed_classes': active, 'fallback_q': fallback, 'classification_weights': [x.tolist() for x in weights],
            'objective': value, 'gradient_max_abs': float(np.max(abs(gradient))), 'iterations': iteration}


def fit_variant(arrays, kind):
    return {'kind': kind, 'C': regression(arrays, kind, 'C'), 'M': regression(arrays, kind, 'M'), 'q': logistic(arrays, kind)}


def fit_base_day(panel, reg, at):
    parents, feedback = base_pool(reg, at); prepared, arrays = prepare_arrays(panel, parents)
    out = {'date': panel.loc[at, 'date'], 'cutoff': day_cutoff(at), 'parents': parents,
           'feedback': feedback, 'prepared': prepared, 'models': {}}
    if prepared['ready']: out['models'] = {k: fit_variant(arrays, k) for k in VARIANTS}
    return out


def predict_base(panel, at, daily, kind):
    prepared = daily['prepared']
    if not prepared['ready'] or not panel.loc[at, 'feature_ready']: return {}, None
    c = CLOCKS.index(panel.loc[at, 'clock']); model = daily['models'][kind]
    mapping, _, _ = parameter_layout(prepared['columns'], kind); ids = mapping[c]
    z = (panel.loc[at, prepared['columns'][c]].to_numpy(float) - prepared['center'][c]) / prepared['scale'][c]
    c_hat = float(z @ np.array(model['C']['coefficients'])[ids] + model['C']['intercepts'][c])
    m_hat = max(0., float(z @ np.array(model['M']['coefficients'])[ids] + model['M']['intercepts'][c]))
    q_model = model['q']
    q = float(expit(z @ np.array(q_model['coefficients'])[ids] + q_model['intercepts'][c])) if q_model['mixed_classes'][c] else q_model['fallback_q'][c]
    direction = m_hat * (2 * q - 1)
    raw = {'C': c_hat, 'D': direction, 'M': m_hat, 'q': q, 'SOXL': c_hat + direction, 'SOXS': c_hat - direction}
    return raw, np.array([raw[s] for s in core.ETFS])


def action_feedback(daily, base):
    return {**daily['feedback'], 'expired_unresolved_prefix': bool(daily['feedback']['expired_unresolved_prefix'] or base['feedback']['expired_unresolved_prefix'])}


def events(at, date, clock, estimates, daily, base, old56, old55):
    fb = {k: action_feedback(daily[k], base) for k in VARIANTS}
    primary = decide(estimates['partial'], fb['partial'], True)
    cash = {'ready': True, 'action': 'CASH', 'allocation': 0.}
    choices = {'partial_band': primary, 'partial_point': decide(estimates['partial'], fb['partial'], False),
               'independent_band': decide(estimates['independent'], fb['independent'], True),
               'shared_band': decide(estimates['shared'], fb['shared'], True),
               'source_v56': old56, 'source_v55': old55,
               'premarket_only': primary if clock == CLOCKS[0] else cash, 'cash': cash, 'double_cost': primary}
    return {k: {**e, 'at': at, 'date': date} for k, e in choices.items()}
