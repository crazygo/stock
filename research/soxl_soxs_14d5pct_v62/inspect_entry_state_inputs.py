"""Post-result input-only audit for a future hypothesis; no new predictions."""
import argparse
from path_learning import *


def inspect(run):
    run = Path(run)
    panel = pd.read_parquet(run / 'activity_panel.parquet')
    features = pd.read_parquet(run / 'features.parquet')
    manifest = read(run / 'input_manifest.json')
    for filename in ['activity_panel.parquet', 'features.parquet']:
        assert sha(run / filename) == manifest['derived_files_sha256'][filename]
    assert panel.index.equals(features.index)
    volatility = np.maximum(features.SOXX_vol20.to_numpy(float), .0001)
    expected = np.clip(features.SOXX_ret5.to_numpy(float) / (volatility * np.sqrt(5)), -5, 5)
    np.testing.assert_allclose(panel.sector_5.to_numpy(float), expected, equal_nan=True, atol=1e-12)
    ready = panel.feature_ready & np.isfinite(panel[FIELDS]).all(axis=1)
    known = np.isfinite(panel.sector_5)
    eligible = panel.loc[ready & known]
    mixed = eligible.relative_5.gt(0) & eligible.sector_5.lt(0)
    out = {
        'post_result_input_only': True,
        'new_models_trained': False,
        'new_capital_results_computed': False,
        'input_manifest_sha256': sha(run / 'input_manifest.json'),
        'audit_source_sha256': sha(Path(__file__)),
        'rows_checked': len(panel),
        'existing_ready_queries': int(ready.sum()),
        'ready_queries_with_absolute_sector_state': int((ready & known).sum()),
        'additional_missing_ready_queries': int((ready & ~known).sum()),
        'ready_dates': int(eligible.date.nunique()),
        'relative_positive_absolute_negative_queries': int(mixed.sum()),
        'relative_positive_absolute_negative_dates': int(eligible.loc[mixed, 'date'].nunique()),
        'first_date': eligible.date.min(),
        'last_date': eligible.date.max(),
        'formula': 'clip(SOXX_ret5 / (max(SOXX_vol20,0.0001)*sqrt(5)), -5, 5)',
        'existing_path_fields': FIELDS,
        'proposed_field': 'sector_5',
        'already_used_in_older_supervised_models': True,
        'availability_evidence': 'Existing causal feature audit inherited; this receipt rechecks the transformation and coverage only.',
        'counts_are_not_new_independent_observations': True,
    }
    dump(run / 'entry_state_input_audit.json', out)
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    inspect(parser.parse_args().run)
