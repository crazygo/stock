"""Local-only descriptive opening study. No fitting, advice, or broker imports."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prefix_features(f, historical_volumes):
    """Receives exactly two closed bars and past-only volumes, never outcomes."""
    if len(f) != 2:
        raise ValueError("exactly two prefix bars required")
    o, c1, c2 = float(f.iloc[0].open), float(f.iloc[0].close), float(f.iloc[1].close)
    r1, r2 = np.log(c1 / o), np.log(c2 / c1)
    high, low = float(f.high.max()), float(f.low.min())
    vol = float(f.volume.sum())
    return {
        "r10": c2 / o - 1, "r1_log": r1, "r2_log": r2,
        "acceleration_log": r2 - r1, "both_up": bool(r1 > 0 and r2 > 0),
        "clv": (c2 - low) / (high - low) if high > low else 0.5,
        "range10": (high - low) / o, "volume10": vol,
        "upper_wick2": (float(f.iloc[1].high) - max(float(f.iloc[1].open), c2)) / o,
        "rvol": vol / np.median(historical_volumes) if len(historical_volumes) >= 10 and np.median(historical_volumes) > 0 else np.nan,
        "rvol_history_days": len(historical_volumes),
        "reference_price": c2,
    }


def path_metrics(f):
    """5m-sampled path; extremes are touch evidence, not guaranteed executions."""
    entry = float(f.iloc[0].open)
    close = f.close.to_numpy(float)
    z = np.log(np.r_[entry, close] / entry)
    total_variation = np.abs(np.diff(z)).sum()
    efficiency = abs(z[-1]) / total_variation if total_variation else 0.0
    late = z[-1] - z[len(f) - len(f)//2]
    running_high = np.maximum.accumulate(np.r_[entry, close])
    return {
        "entry_price": entry, "terminal": close[-1] / entry - 1,
        "mfe": max(0., float(f.high.max()) / entry - 1),
        "mae": min(0., float(f.low.min()) / entry - 1),
        "close_drawdown": float(np.min(np.r_[entry, close] / running_high - 1)),
        "efficiency": efficiency,
        "late_share": late / z[-1] if z[-1] > 0 else np.nan,
    }


def rule_masks(f):
    band = f.r10.ge(.005) & f.r10.lt(.02)
    shape = band & f.both_up & f.clv.ge(.7)
    return {
        "all": pd.Series(True, index=f.index),
        "r10_ge_0.5pct": f.r10.ge(.005),
        "r10_0.5_to_2pct": band,
        "band_shape": shape,
        "band_shape_rvol1.5": shape & f.rvol.ge(1.5),
        "band_shape_rvol2": shape & f.rvol.ge(2),
        "band_shape_rvol3": shape & f.rvol.ge(3),
        "band_shape_acceleration": shape & f.acceleration_log.ge(0),
        "r10_ge_1pct": f.r10.ge(.01),
        "r10_ge_2pct": f.r10.ge(.02),
    }


def stats(x, target, config, denominator):
    n = len(x)
    if not n:
        return {"n": 0}
    sustained = x.terminal.ge(target) & x.efficiency.ge(config['efficiency_min']) & x.late_share.ge(config['late_contribution_min'])
    out = {"n": n, "dates": x.session_date.nunique(), "symbols": x.symbol.nunique(),
           "coverage": n / denominator, "touch_rate": x.mfe.ge(target).mean(),
           "terminal_rate": x.terminal.ge(target).mean(), "sustain_rate": sustained.mean(),
           "mean_terminal": x.terminal.mean(), "median_terminal": x.terminal.median(),
           "q10_terminal": x.terminal.quantile(.1), "median_mfe": x.mfe.median(),
           "median_mae": x.mae.median(), "q10_mae": x.mae.quantile(.1),
           "top_symbol_share": x.symbol.value_counts().iloc[0] / n}
    for cost in [10, 20, 40]:
        side = cost / 20000
        net = (1 + x.terminal) * (1-side) / (1+side) - 1
        out[f"mean_net_{cost}bp"] = net.mean()
        out[f"positive_net_{cost}bp"] = net.gt(0).mean()
    return out


def run(output):
    config = json.loads((HERE/'audit_config.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    universe_path, calendar_path = ROOT/config['universe'], ROOT/config['calendar']
    members = [m for m in json.loads(universe_path.read_text())['members'] if m['role'] == 'candidate']
    sessions = [s for s in json.loads(calendar_path.read_text())['sessions'] if config['source_start'] <= s['session_date'] <= config['source_end']]
    days = [s['session_date'] for s in sessions]
    files = [HERE/'audit.py', HERE/'audit_config.json', HERE/'STUDY_PLAN.md', universe_path, calendar_path]
    files += [ROOT/config['data_root']/m['symbol']/'2026.parquet' for m in members]
    files += [ROOT/'market_data/corporate_actions'/f"{m['symbol']}.parquet" for m in members]
    files = [p for p in files if p.exists()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'config': config,
                'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True),
                'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
                'sources': hashes, 'scope': 'descriptive_not_predictive_validation'}
    dump(output/'manifest_at_launch.json', manifest)
    dump(output/'config.json', config)
    features, outcomes, eligibility, source_audit = [], [], [], []
    as_of = pd.Timestamp.now(tz='UTC')
    for member in members:
        sym = member['symbol']; source = ROOT/config['data_root']/sym/'2026.parquet'
        if not source.exists():
            for day in days:
                eligibility.append({'symbol': sym, 'session_date': day, 'stage': 'input', 'status': 'source_missing'})
            continue
        f = pd.read_parquet(source)
        if set(f.price_basis.dropna()) != {'NONE'}:
            raise ValueError(f'mixed price basis: {sym}')
        for c in ['start_at','end_at','available_at']:
            f[c] = pd.to_datetime(f[c], utc=True)
        if f.start_at.duplicated().any():
            raise ValueError(f'duplicate source starts: {sym}')
        numeric = f[['open','high','low','close','volume']].to_numpy(float)
        f['valid'] = (np.isfinite(numeric).all(axis=1) & (f[['open','high','low','close']] > 0).all(axis=1)
                      & (f.high >= f[['open','close','low']].max(axis=1))
                      & (f.low <= f[['open','close','high']].min(axis=1)) & (f.volume > 0)
                      & (f.end_at-f.start_at).eq(pd.Timedelta(minutes=5)) & f.available_at.ge(f.end_at))
        source_audit.append({'symbol': sym, 'rows': len(f), 'invalid_or_zero_volume': int((~f.valid).sum()),
                             'start': f.start_at.min().isoformat(), 'end': f.end_at.max().isoformat(),
                             'overnight_rows': int(f.session_type.eq('overnight').sum()),
                             'has_received_at': 'received_at' in f})
        if 'received_at' in f:
            f['received_at'] = pd.to_datetime(f.received_at, utc=True)
        f = f.set_index('start_at').sort_index()
        ca = ROOT/'market_data/corporate_actions'/f'{sym}.parquet'
        split_dates = set()
        if ca.exists():
            actions = pd.read_parquet(ca)
            for _, a in actions.iterrows():
                if any(pd.notna(a.get(k)) and a.get(k) not in (0,1) for k in ['split_ratio','join_ert','spin_off_ratio']):
                    split_dates.add(str(a['ex_div_date']))
        past = {}
        for day_i, s in enumerate(sessions):
            day = s['session_date']; opening = pd.Timestamp(s['open_at'])
            cutoff = opening+pd.Timedelta(minutes=10); decision = cutoff+pd.Timedelta(seconds=30)
            sid = f"opening_continuation_v1|{sym}|{day}|09:40"
            prefix = f.reindex(pd.date_range(opening, periods=2, freq='5min'))
            status = 'eligible'
            if prefix.end_at.isna().any(): status = 'prefix_missing'
            elif not prefix.valid.fillna(False).all() or not prefix.session_type.eq('regular').all(): status = 'prefix_invalid_or_zero_volume'
            elif prefix.end_at.max() > cutoff or prefix.available_at.max() > decision: status = 'prefix_late'
            elif 'received_at' in prefix and prefix.received_at.gt(decision).any(): status = 'prefix_late_received'
            eligibility.append({'sample_id': sid, 'symbol': sym, 'session_date': day, 'stage': 'input', 'status': status})
            if status != 'eligible': continue
            hist_days = days[max(0, day_i-20):day_i]
            history = [past[d] for d in hist_days if d in past]
            if split_dates.intersection(hist_days+[day]): history = []
            feats = prefix_features(prefix, history)
            past[day] = float(prefix.volume.sum())
            features.append({'sample_id': sid,'symbol': sym,'session_date': day,'cutoff_at': cutoff.isoformat(),
                             'decision_at': decision.isoformat(),'max_source_available_at': prefix.available_at.max().isoformat(), **feats})
            for entry_min in config['entry_minutes_after_open']:
                entry = opening+pd.Timedelta(minutes=entry_min)
                for horizon in config['horizons_minutes']:
                    path = f.reindex(pd.date_range(entry, periods=horizon//5, freq='5min'))
                    end = entry+pd.Timedelta(minutes=horizon)
                    status = 'mature'
                    if end > as_of: status = 'pending'
                    elif path.end_at.isna().any(): status = 'label_missing'
                    elif not path.valid.fillna(False).all() or not path.session_type.eq('regular').all(): status = 'label_invalid_or_zero_volume'
                    elif path.available_at.max() > as_of: status = 'pending_availability'
                    eligibility.append({'sample_id':sid,'symbol':sym,'session_date':day,'stage':'label',
                                        'entry_min':entry_min,'horizon':horizon,'status':status})
                    if status != 'mature': continue
                    result = {'sample_id':sid,'symbol':sym,'session_date':day,'entry_min':entry_min,'horizon':horizon,
                              'entry_at':entry.isoformat(),'label_end_at':end.isoformat(),
                              'label_available_at':path.available_at.max().isoformat(),**path_metrics(path)}
                    for target in config['targets']:
                        hits = np.flatnonzero(path.high.to_numpy(float) >= result['entry_price']*(1+target))
                        result[f'hit_lower_min_{target}'] = int(hits[0]*5) if len(hits) else np.nan
                    outcomes.append(result)
        print(f"{sym}: {len(features)} cumulative input rows", flush=True)
    x,y,e = pd.DataFrame(features),pd.DataFrame(outcomes),pd.DataFrame(eligibility)
    x.to_parquet(output/'features.parquet', index=False)
    y.to_parquet(output/'outcomes.parquet', index=False)
    e.to_parquet(output/'eligibility.parquet', index=False)
    pd.DataFrame(source_audit).to_csv(output/'source_audit.csv', index=False)
    merged = y.merge(x.drop(columns=['symbol','session_date']), on='sample_id',validate='many_to_one')
    merged['month'] = merged.session_date.str[:7]
    rules, bins, months = [],[],[]
    bin_edges = [-np.inf,0,.0025,.005,.0075,.01,.015,.02,.03,np.inf]
    bin_names = ['negative','0_to_0.25','0.25_to_0.5','0.5_to_0.75','0.75_to_1','1_to_1.5','1.5_to_2','2_to_3','ge_3']
    for (entry,horizon), group in merged.groupby(['entry_min','horizon']):
        masks = rule_masks(group)
        for target in config['targets']:
            for name,mask in masks.items():
                rules.append({'entry_min':entry,'horizon':horizon,'target':target,'rule':name,**stats(group.loc[mask],target,config,len(group))})
            if entry == 15:
                for name,b in group.groupby(pd.cut(group.r10,bin_edges,labels=bin_names,right=False),observed=True):
                    bins.append({'entry_min':entry,'horizon':horizon,'target':target,'r10_bin':name,**stats(b,target,config,len(group))})
        if entry == 15 and horizon == 30:
            for month, subgroup in group.groupby('month'):
                for name,mask in rule_masks(subgroup).items():
                    months.append({'month':month,'rule':name,'target':.01,**stats(subgroup.loc[mask],.01,config,len(subgroup))})
    pd.DataFrame(rules).to_csv(output/'rule_grid.csv',index=False)
    pd.DataFrame(bins).to_csv(output/'return_bins.csv',index=False)
    pd.DataFrame(months).to_csv(output/'monthly.csv',index=False)
    base = merged[(merged.entry_min == 15) & (merged.horizon == 30)].copy()
    base['sustain'] = base.terminal.ge(.01) & base.efficiency.ge(.5) & base.late_share.ge(.25)
    boot = []
    for rule,mask in rule_masks(base).items():
        selected = base.loc[mask]
        sums = selected.groupby('session_date').agg(n=('terminal','size'),wins=('sustain','sum'),ret=('terminal','sum')).reindex(days,fill_value=0).to_numpy(float)
        for block in config['bootstrap_blocks']:
            rng = np.random.default_rng(config['seed'])
            starts = rng.integers(0,len(days)-block+1,size=(config['bootstrap_repeats'],int(np.ceil(len(days)/block))))
            ix = (starts[...,None]+np.arange(block)).reshape(config['bootstrap_repeats'],-1)[:,:len(days)]
            totals = sums[ix].sum(axis=1); good = totals[:,0] > 0
            rates = totals[good,1]/totals[good,0]; returns=totals[good,2]/totals[good,0]
            boot.append({'rule':rule,'block':block,'zero_denominator_replicates':int((~good).sum()),
                         'rate_lower':np.quantile(rates,.025),'rate_upper':np.quantile(rates,.975),
                         'return_lower':np.quantile(returns,.025),'return_upper':np.quantile(returns,.975)})
    pd.DataFrame(boot).to_csv(output/'bootstrap.csv',index=False)
    primary = base[rule_masks(base)['band_shape']]
    primary.groupby('symbol').agg(n=('terminal','size'),sustain=('sustain','mean'),mean_return=('terminal','mean')).sort_values('n',ascending=False).to_csv(output/'by_symbol.csv')
    primary.groupby('session_date').size().reindex(days,fill_value=0).rename('n').to_csv(output/'daily_candidates.csv')
    if any(digest(ROOT/p) != h for p,h in hashes.items()):
        raise RuntimeError('source changed during audit; refuse completion')
    summary = {'candidate_symbols':len(members),'official_dates':len(days),'potential_stock_days':len(members)*len(days),
               'eligible_input_rows':len(x),'mature_primary_rows':len(base),'source_files':len(source_audit),
               'input_status':e[e.stage.eq('input')].status.value_counts().to_dict(),
               'label_status':e[e.stage.eq('label')].status.value_counts().to_dict(),
               'rvol_available_rows':int(x.rvol.notna().sum()),'source_hashes_unchanged':True,
               'completed_at':datetime.now(timezone.utc).isoformat()}
    dump(output/'summary.json',summary)
    dump(output/'output_hashes.json',{p.name:digest(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
