"""Post-result development diagnosis and old-control audit; no action changes."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from local_training import core, ROUTES
from verify_continuation import read


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_run(run):
    run = Path(run); panel = pd.read_parquet(run/'panel.parquet')
    diag = read(run/'diagnosis.json'); frozen = read(run/'decisions_frozen.json')['routes']
    cutoff = pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive'])
    locked = {n: sha(run/n) for n in ['predictions.json', 'decisions_frozen.json', 'results.json']}
    checks = 0
    for name in ROUTES:
        rows = []; unknown = []
        for e in frozen[name]:
            at = pd.Timestamp(e['at']); row = panel.loc[at]
            rec = read(run/'training'/(e['date'][:10]+'.json')); fc = rec['forecasts'][name]['event']
            if not fc['ready']:
                continue
            if pd.isna(row.release) or row.release >= cutoff:
                unknown.append(e['date'][:10]); continue
            sign = 1 if fc['top_symbol'] == 'SOXL' else -1
            actual = row.C+sign*row.D; state = ('support' if row.relative_5 > 0 and row.breadth_5 > 0 and row.volume_direction_diffusion > 0 else
                'weakness' if row.relative_5 < 0 and row.breadth_5 < 0 and row.volume_direction_diffusion < 0 else 'mixed')
            rows.append([fc['adjusted'][fc['top_symbol']], actual, (sign == 1) == (row.D >= 0),
                (fc['adjusted']['D']-row.D)**2, row.C+abs(row.D)-actual, e['allocation'] > 0, state])
        assert unknown == diag['routes'][name]['unknown_dates']
        frame = pd.DataFrame(rows, columns=['predicted','actual','correct','mse','regret','allowed','state'])
        groups = {'mature_opportunities': frame, 'allowed_mature_opportunities': frame[frame.allowed.eq(True)]}
        groups.update({s: frame[frame.state == s] for s in ['support', 'weakness', 'mixed']})
        for group, f in groups.items():
            saved = diag['routes'][name][group] if group.endswith('opportunities') else diag['routes'][name]['states'][group]
            assert saved['dates'] == len(f)
            if not len(f):
                continue
            expected = {'direction_correct': int(f.correct.sum()), 'predicted_mean_log': f.predicted.mean(),
                'actual_mean_log': f.actual.mean(), 'D_mean_squared_error': f.mse.mean(), 'regret_mean_log': f.regret.mean()}
            for k, v in expected.items():
                np.testing.assert_allclose(saved[k], v, atol=1e-12, rtol=1e-10); checks += 1
    assert all(sha(run/n) == h for n, h in locked.items())
    receipt = {'passed': True, 'post_result_read_only': True, 'diagnostic_statistics_checked': checks,
        'diagnosis_sha256': sha(run/'diagnosis.json'), 'frozen_inputs_unchanged': locked}
    core.dump(run/'diagnosis_verification.json', receipt)
    return receipt



def audit(run):
    run = Path(run)
    receipt = audit_run(run)
    current = read(run/'predictions.json')['global_base_cal']
    old = read(core.ROOT/'.cache/soxl_soxs_14d5pct_v21/runs/early_20261012/predictions.json')['early_base_cal']
    assert current == old
    receipt['exact_v21_global_control_events'] = len(current)
    core.dump(run/'diagnosis_verification.json', receipt)
    print(receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run)
