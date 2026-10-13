"""Create short proxy labels without modifying the original execution labels."""
import argparse
import shutil
from datetime import datetime, timezone
from short_learning import *


def prepare_inputs(output, phase='development', base=None):
    output = Path(output); assert not output.exists()
    source, source_sha = verified_source()
    if base is None:
        assert phase == 'development', 'Case feature extension must be audited separately first'
        base = source
    base = Path(base); old = read(base / 'input_manifest.json'); assert old['phase'] == phase
    for f, h in old['derived_files_sha256'].items():
        assert sha(base / f) == h
    market = core.load_market(); assert market.manifest == old['full_market']
    market = restrict_market(market, pd.Timestamp(old['observation_cutoff_exclusive']))
    output.mkdir(parents=True)
    dump(output / 'input_registration.json', {'started_at': datetime.now(timezone.utc).isoformat(),
         'phase': phase, 'protocol_sha256': sha(SOURCE / 'protocol.json'),
         'source_sha256': {str(f.relative_to(core.ROOT)): sha(f) for f in [SOURCE / 'prepare_short.py', SOURCE / 'short_learning.py']},
         'new_capital_outcomes_seen': False})
    copied = ['features.parquet', 'financial_panel.parquet', 'activity_panel.parquet', 'baseline_labels.json', 'stress_labels.json']
    for name in copied:
        shutil.copyfile(base / name, output / name)
    dump(output / 'base_input_manifest.json', old)
    original = pd.read_parquet(output / 'activity_panel.parquet')
    records = [short_label(market, s, at) for at in original.index for s in core.ETFS]
    dump(output / 'short_labels.json', records)
    panel = build_short_panel(original, records)
    panel.to_parquet(output / 'short_panel.parquet', compression='zstd', compression_level=7)
    manifest = {'phase': phase, 'source_v56_run': str(source), 'source_v56_manifest_sha256': source_sha,
                'base_inputs': str(base), 'base_input_manifest_sha256': sha(base / 'input_manifest.json'),
                'full_market': old['full_market'], 'observation_cutoff_exclusive': old['observation_cutoff_exclusive'],
                'original_inputs_sha256': {n: sha(base / n) for n in copied},
                'derived_files_sha256': {f.name: sha(f) for f in output.iterdir() if f.is_file()}}
    dump(output / 'input_manifest.json', manifest)
    receipt = {'queries': len(panel), 'short_known_pairs': int(panel.terminal_release.notna().sum()),
               'original_feature_ready': int(original.feature_ready.sum()),
               'short_known_feature_ready_pairs': int((panel.feature_ready & panel.terminal_release.notna()).sum()),
               'unknown_pairs_retained': int(panel.terminal_release.isna().sum()),
               'original_fourteen_day_inputs_unchanged': True, 'new_capital_outcomes_seen': False}
    dump(output / 'coverage.json', receipt); print(receipt, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['development', 'cases'], default='development'); p.add_argument('--base', type=Path)
    a = p.parse_args(); prepare_inputs(a.output, a.phase, a.base)
