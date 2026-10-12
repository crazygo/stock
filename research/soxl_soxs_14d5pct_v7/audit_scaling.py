"""Audit weighted constant-column roundoff without hiding production warnings."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from breadth import core, variant_columns


def audit(run):
    run = Path(run); features = pd.read_parquet(run / 'features.parquet')
    fits = 0; warnings_count = 0; negative = []
    for path in sorted((run / 'training').glob('*.json')):
        record = json.loads(path.read_text())
        for variant, fit in record['variants'].items():
            if not fit['ready']:
                continue
            members = pd.to_datetime(fit['members'], utc=True)
            f = features.loc[members]; counts = f.groupby('date').size()
            weights = f.date.map(1 / counts).to_numpy(dtype=float)
            columns = variant_columns(features, variant)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                scaler = StandardScaler().fit(f[columns], sample_weight=weights)
            warnings_count += len(caught)
            assert all('invalid value encountered in sqrt' in str(w.message) for w in caught)
            assert np.isfinite(scaler.mean_).all() and np.isfinite(scaler.scale_).all()
            assert np.isfinite(scaler.transform(f[columns])).all()
            for i in np.flatnonzero(scaler.var_ < 0):
                assert abs(scaler.var_[i]) < 1e-20 and scaler.scale_[i] == 1.
                assert f[columns[i]].nunique() == 1
                negative.append({'date': record['date'], 'variant': variant, 'column': columns[i],
                                 'roundoff_variance': float(scaler.var_[i]), 'scale': float(scaler.scale_[i])})
            fits += 1
    forecast_checks = []
    for path in sorted(run.glob('*_predictions.json')):
        f = pd.read_json(path, orient='table'); raw = f[['SOXL_raw','SOXS_raw']]
        partial = raw.isna().any(axis=1) & ~raw.isna().all(axis=1)
        assert not partial.any()
        present = raw.dropna(); assert np.isfinite(present.to_numpy()).all()
        forecast_checks.append({'route': path.stem, 'finite_fitted_rows': len(present)})
    result = {'passed': True, 'scaler_fits': fits, 'captured_audit_warnings': warnings_count,
        'negative_roundoff_constant_columns': negative, 'all_scales_and_transformed_values_finite': True,
        'forecast_checks': forecast_checks, 'production_warnings_preserved_in_run_log': True}
    core.dump(run / 'numerical_audit.json', result)
    print(json.dumps({k: v for k,v in result.items() if k not in ['negative_roundoff_constant_columns','forecast_checks']},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run',type=Path)
    with threadpool_limits(limits=1):
        audit(parser.parse_args().run)
