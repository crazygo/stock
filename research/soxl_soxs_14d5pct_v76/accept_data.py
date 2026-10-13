"""Release audited inputs for causal research, without granting a model qualification."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v76'
SOURCE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    d=read(CACHE/'data_audit.json');e=read(CACHE/'extension_audit.json')
    anomaly=read(CACHE/'source_anomaly_review.json')
    assert d['structure_passed'] and d['actions_passed'] and e['passed']
    assert e['data_audit_sha256']==sha(CACHE/'data_audit.json')
    source_comparison=[];diagnostics=[]
    for r in d['symbols']:
        x=pd.read_csv(CACHE/f"{r['symbol']}_price_comparison.csv")
        closes=x[x.field=='close'];assert len(closes)==390
        # One raw cent plus numerical tolerance; no price rescaling fitted from discrepancies.
        assert closes.difference.abs().max()<=.0100001
        source_comparison.append(dict(symbol=r['symbol'],daily_ohlc_values=len(x),
            close_max_absolute=float(closes.difference.abs().max()),
            field_max_absolute=x.assign(abs=x.difference.abs()).groupby('field')['abs'].max().to_dict(),
            field_max_relative=x.assign(abs=x.relative_difference.abs()).groupby('field')['abs'].max().to_dict(),
            exact_cross_provider_prices=False))
    for row in read(CACHE/'daily_diagnostics/receipt.json'):
        path=ROOT/row['path'];assert sha(path)==row['sha256']
        daily=pd.read_parquet(path).iloc[0]
        comparison=pd.read_csv(CACHE/f"{row['symbol']}_price_comparison.csv")
        comparison=comparison[comparison.date==row['date']].set_index('field')
        diagnostics.append(dict(symbol=row['symbol'],date=row['date'],
            minute_aggregated={k:float(comparison.loc[k,'futu']) for k in ['open','high','low','close']},
            futu_daily={k:float(daily[k]) for k in ['open','high','low','close']},
            ibkr_daily_unadjusted={k:float(comparison.loc[k,'ibkr_unadjusted']) for k in ['open','high','low','close']},
            source_sha256=row['sha256']))
    result=dict(accepted_for_causal_research=True,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        data_audit_sha256=sha(CACHE/'data_audit.json'),extension_audit_sha256=sha(CACHE/'extension_audit.json'),
        rows=sum(r['rows'] for r in d['symbols']),symbols=len(d['symbols']),additional_calendar_sessions=385,
        price_comparisons=source_comparison,diagnostic_cases=diagnostics,
        known_source_disagreement=anomaly,
        missing_prefixes={r['symbol']:r['prefix_missing'] for r in d['symbols'] if r['prefix_missing']},
        conditions=['Use raw Futu paths without replacing prices from IBKR.',
            'Preserve missing-prefix readiness through all trailing-feature dependencies.',
            'Use official half-day close mapping and end+1s historical availability assumption.',
            'Existing raw parent interval is immutable; earlier economic units can differ by a constant split scale.',
            'Daily cross-provider prices are not identical. Contract audit establishes code/path consistency, not actual execution or minute-level consolidated coverage.',
            'Preserve the flagged SPY intraday-low disagreement; no retrospective daily-low replacement. Register activity-source sensitivity in the next model.',
            'No proof of historical overnight tradability; current inputs use completed 04:00 onward prefixes and regular-session exit paths.'],
        new_model_trained=False,new_strategy_evaluated=False,model_qualified=False,goal_complete=False)
    assert not (CACHE/'acceptance.json').exists()
    (CACHE/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':main()
