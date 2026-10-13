"""Prepare immutable causal features and all hypothetical own-price outcomes."""
import argparse
import shutil
from datetime import datetime, timezone
from transfer_learning import *


def prepare(output):
    output = Path(output); assert not output.exists()
    output.mkdir(parents=True)
    sources = {str(f.relative_to(ROOT)): sha(f) for f in sorted((ROOT/'research').glob('soxl_soxs_14d5pct_v*/*.py'))}
    dump(output/'input_registration.json', {'created_at': datetime.now(timezone.utc).isoformat(),
         'protocol_sha256': sha(SOURCE/'protocol.json'), 'code_sha256': sources,
         'strategy_capital_outcomes_seen': False})
    market = load_market()
    for name in ['features.parquet', 'baseline_labels.json', 'stress_labels.json']:
        shutil.copyfile(BASE/name, output/name)
    features = pd.read_parquet(output/'features.parquet')
    prefix_index = pd.DatetimeIndex([pd.Timestamp(str(d.date())+' '+c, tz=core.ET).tz_convert('UTC')
         for d in market.sessions[(market.sessions>='2024-10-14') & (market.sessions<=features.date.max())]
         for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c, tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])])
    prefixes = prefix_table(market, prefix_index)
    for s, p in prefixes.items():
        p.to_parquet(output/(s+'_prefix.parquet'), index=True)
    base = labels(output/'baseline_labels.json'); all_labels = []; panels = {}; receipts = []
    for name, (long, short, _) in PAIRS.items():
        rows = []
        for i, (at, row) in enumerate(features.iterrows()):
            x, receipt = feature_row(market, prefixes, name, at)
            pair_labels = []
            for s in [long, short]:
                if name == 'semiconductor':
                    label = base[at.isoformat()+'|'+s]
                else:
                    label = auxiliary_label(market, s, at)
                    expiry = label.get('expiry_at'); start = label.get('entry_at')
                    f = market.bars[s]
                    path = f[f.start_at.ge(start) & f.end_at.le(expiry) & f.session_type.eq('regular')] if start is not None and expiry is not None else f.iloc[:0]
                    label['label_available_at'] = path.available_at.max() if label.get('complete_window') and len(path) else None
                pair_labels.append(label); all_labels.append(label)
            known = all(l.get('complete_window') and l.get('label_available_at') is not None and
                        l.get('net_return') is not None and np.isfinite(l['net_return']) and l['net_return'] > -1
                        for l in pair_labels)
            y = np.log1p([l['net_return'] for l in pair_labels]) if known else np.array([np.nan, np.nan])
            rows.append({'at': at, 'date': row.date, 'clock': row.clock, 'feature_ready': x is not None,
                         **dict(zip(FIELDS, x if x is not None else np.full(len(FIELDS), np.nan))),
                         'C': float(y.mean()), 'D': float((y[0]-y[1])/2),
                         'terminal_release': max(l['label_available_at'] for l in pair_labels) if known else pd.NaT})
            receipts.append({'at': at, 'pair': name, **receipt})
            if (i+1)%700 == 0:
                print('Prepared', name, i+1, '/',len(features), flush=True)
        panel = pd.DataFrame(rows).set_index('at')
        panel['terminal_release'] = pd.to_datetime(panel.terminal_release, utc=True).astype('datetime64[ns, UTC]')
        panel.to_parquet(output/(name+'_panel.parquet'), index=True); panels[name] = panel
    dump(output/'all_labels.json', all_labels)
    dump(output/'feature_receipts.json', receipts)
    dump(output/'input_manifest.json', {'phase':'development', 'observation_cutoff_exclusive': CUTOFF,
         'market_manifest':market.manifest, 'protocol_sha256':sha(SOURCE/'protocol.json'),
         'auxiliary_input_manifest_sha256':sha(ROOT/'research/soxl_soxs_14d5pct_v70/evidence/input_manifest.json'),
         'derived_files_sha256':{f.name:sha(f) for f in output.iterdir() if f.is_file()},
         'coverage':{k:{'rows':len(p),'ready':int(p.feature_ready.sum()),'mature':int(p.terminal_release.notna().sum())} for k,p in panels.items()}})
    print(read(output/'input_manifest.json')['coverage'], flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();prepare(a.output)
