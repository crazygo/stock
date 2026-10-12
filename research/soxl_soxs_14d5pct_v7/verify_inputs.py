"""Raw-bar arithmetic and representative daily-model reproduction."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from breadth import core, PANEL, load_panel, variant_columns
from learning import fit_daily
from walkforward import label_tables


def verify(run):
    run = Path(run)
    features = pd.read_parquet(run / 'features.parquet')
    market = core.load_market(); bars, daily, manifest = load_panel(market.calendar)
    original = json.loads((run / 'input_manifest.json').read_text())
    assert original['panel'] == manifest and original['market'] == market.manifest
    assert hashlib.sha256((run / 'features.parquet').read_bytes()).hexdigest() == original['feature_sha256']
    checks, model_checks = [], []
    for day in ['2026-05-01','2026-07-01','2026-09-01']:
        for clock in ['10:05:30','10:35:30','13:05:30']:
            at = pd.Timestamp(day + ' ' + clock, tz=core.ET).tz_convert('UTC')
            saved = features.loc[at]; rebuilt, aggregate = {}, {}
            date = pd.Timestamp(day); dates = market.sessions[market.sessions < date][-21:]
            for s in PANEL:
                f = bars[s]
                history_closes, history_volumes = [], []
                for d in dates:
                    h = f[(f.date == d) & f.regular]
                    assert (h.available_at < at).all()
                    history_closes.append(h.econ_close.iloc[-1]); history_volumes.append(h.econ_volume.sum())
                h = np.asarray(history_closes)
                today = f[(f.date == date) & (f.available_at <= at)]
                reg = today[today.regular]
                row = {f'ret{lag}': h[-1] / h[-1-lag] - 1 for lag in [1,5,20]}
                row['vol20'] = np.std(h[1:] / h[:-1] - 1, ddof=1)
                row['today'] = today.econ_close.iloc[-1] / h[-1] - 1
                row['relative_volume'] = today.econ_volume.sum() / np.mean(history_volumes[-20:])
                c = reg.econ_close.to_numpy(); o = reg.econ_open.iloc[0]
                w = np.average((reg.econ_high + reg.econ_low + reg.econ_close) / 3, weights=reg.econ_volume)
                row['session_return'] = c[-1] / o - 1
                row['vwap_distance'] = c[-1] / w - 1
                row['recent15'] = c[-1] / reg.econ_open.iloc[-3] - 1
                row['efficiency'] = (c[-1] - o) / np.abs(np.diff(np.r_[o,c])).sum()
                for key, value in row.items():
                    rebuilt['leader_' + s + '_' + key] = value
                    aggregate.setdefault(key,[]).append(value)
            for key, values in aggregate.items():
                a = np.asarray(values)
                rebuilt['breadth_' + key + '_mean'] = a.mean()
                rebuilt['breadth_' + key + '_std'] = a.std(ddof=0)
                rebuilt['breadth_' + key + '_positive_fraction'] = (a > 0).mean()
            for symbol in ['SOXX','QQQ']:
                f = market.bars[symbol]
                reg = f[(f.date == date) & f.session_type.eq('regular') & (f.available_at <= at)]
                h, l = reg.econ_high.iloc[:6].max(), reg.econ_low.iloc[:6].min()
                for side, outside in [('high',reg.econ_close > h),('low',reg.econ_close < l)]:
                    outside.iloc[:6] = False
                    indices = reg.index[outside]
                    age = (reg.index[-1] - indices[-1]).total_seconds()/60 if len(indices) else 390.
                    inside = reg.econ_close.iloc[-1] <= h if side == 'high' else reg.econ_close.iloc[-1] >= l
                    prefix = 'timing_' + symbol + '_' + side
                    rebuilt[prefix + '_ever'] = float(len(indices)>0)
                    rebuilt[prefix + '_age'] = min(age,390.)/390.
                    rebuilt[prefix + '_fresh_failure'] = float(len(indices)>0 and 0<age<=15 and inside)
            for key, value in rebuilt.items():
                np.testing.assert_allclose(saved[key],value,rtol=1e-11,atol=1e-12,err_msg=key)
            checks.append({'at': str(at), 'raw_fields_checked': len(rebuilt), 'passed': True})
    protocol = json.loads((run/'registration.json').read_text())['preregistration']['protocol']
    labels = json.loads((run/'labels.json').read_text()); tables = label_tables(labels)
    regular = features[features.regular_context_ready.eq(1.)]
    for day in ['2026-05-01','2026-07-01','2026-09-01']:
        sample = features[features.date == pd.Timestamp(day)]
        for variant in protocol['feature_variants']:
            X = regular[['date','clock',*variant_columns(features,variant)]]
            fit, _, _ = fit_daily(X,tables,sample.index[0],protocol)
            models, columns = fit
            for algorithm in ['ridge_log','boosted_log']:
                history = pd.read_json(run/(variant+'__'+algorithm+'_predictions.json'),orient='table')
                for at,row in sample[sample.regular_context_ready.eq(1.)].iterrows():
                    for symbol in core.ETFS:
                        raw = float(models[algorithm][symbol].predict(pd.DataFrame([row])[columns])[0])
                        np.testing.assert_allclose(raw,history.loc[at,symbol+'_raw'],rtol=1e-10,atol=1e-12)
            model_checks.append({'date':day,'variant':variant,'two_algorithms_two_etfs_reproduced':True})
    legacy_check = {'available': False}
    legacy = core.ROOT / '.cache/soxl_soxs_14d5pct_v6/runs/intraday_path_20261012'
    if (legacy / 'features.parquet').exists():
        old = pd.read_parquet(legacy / 'features.parquet')
        common = old.index.intersection(features.index)
        cols = ['date','clock',*variant_columns(features,'base')]
        pd.testing.assert_frame_equal(old.loc[common,cols],features.loc[common,cols])
        legacy_check = {'available': True, 'unchanged_base_feature_rows': len(common),
                        'identical_full_index': features.index.equals(old.index)}
        if legacy_check['identical_full_index']:
            for a in ['ridge_log','boosted_log']:
                new = pd.read_json(run/('base__'+a+'_predictions.json'),orient='table')
                before = pd.read_json(legacy/(a+'__flat_close_predictions.json'),orient='table')
                assert list(new.action) == list(before.action)
                for s in core.ETFS:
                    np.testing.assert_allclose(new[s+'_raw'],before[s+'_raw'],atol=1e-9,rtol=1e-10,equal_nan=True)
            legacy_check['two_component_forecast_routes_unchanged'] = True
    result = {'passed':True,'raw_feature_cases':checks,'daily_model_reproductions':model_checks,
              'legacy_base_reproduction':legacy_check,
              'input_hashes_unchanged':True,'no_effect_pass_inferred':True}
    core.dump(run/'input_verification.json',result);print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    with threadpool_limits(limits=1):
        verify(parser.parse_args().run)
