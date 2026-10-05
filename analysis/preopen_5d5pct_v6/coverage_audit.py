"""Requested, feature, raw and scoreable coverage are separate evidence layers."""
import json
import pandas as pd
from common import OUT,write,now

def main():
    quality=json.loads((OUT/'data_quality.json').read_text())['records']
    for rid in ['R00','R01','R02','R03','R04','R05']:
        dest=OUT if rid=='R00' else OUT/rid;path=dest/'cache/panel.parquet'
        if not path.exists() or not (dest/'coverage.json').exists():continue
        f=pd.read_parquet(path,columns=['symbol','day','y']);f.day=f.day.astype(str);known=f[f.y.notna()]
        coverage=json.loads((dest/'coverage.json').read_text());coverage.setdefault('requested',['2024-10-04','2026-09-30'])
        raw=[r for r in quality if 'error' not in r and ('cache/backfill/' in r['path'] if rid=='R03' else 'preopen_ranked_policy_v5/raw/' in r['path'])]
        coverage['raw_ohlc_boundary']=[min(r['valid_first'] for r in raw),max(r['valid_last'] for r in raw)] if raw else None
        coverage['scoreable_features']=dict(known_rows=len(known),unknown_rows=len(f)-len(known),mature_stock_days=len(known[['symbol','day']].drop_duplicates()),known_label_date_range=[known.day.min(),known.day.max()] if len(known) else None,note='Five-day full-window labels including action exclusions; bounds do not imply every stock/time is scoreable.')
        coverage['coverage_audited_at']=now();write(dest/'coverage.json',coverage)
    print(json.dumps(dict(status='completed')),flush=True)
if __name__=='__main__':main()
