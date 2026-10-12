"""Post-run issued-choice diagnostics; never changes the strategy or thresholds."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def diagnose(run):
    run = Path(run)
    protocol = json.loads((run / 'registration.json').read_text())['preregistration']['protocol']
    name = protocol['primary']
    forecasts = pd.read_json(run / (name + '_predictions.json'), orient='table')
    trades = pd.read_csv(run / (name + '_trades.csv'))
    equity = pd.read_csv(run / (name + '_equity.csv'))
    labels = pd.DataFrame(json.loads((run / 'labels.json').read_text()))
    labels['decision_at'] = pd.to_datetime(labels.decision_at, utc=True)
    rows = []
    for trade in trades.itertuples():
        at = pd.Timestamp(trade.decision_at)
        other = 'SOXS' if trade.symbol == 'SOXL' else 'SOXL'
        alternative = labels[labels.decision_at.eq(at) & labels.symbol.eq(other)].iloc[0]
        rows.append({'predicted': forecasts.loc[at, trade.symbol + '_adjusted'],
            'actual_log': np.log1p(trade.net_return), 'actual_net': trade.net_return,
            'other_net': alternative.net_return})
    frame = pd.DataFrame(rows)
    result = {'diagnostic_only_after_run': True, 'no_policy_or_threshold_changed': True,
        'actual_entries': len(frame), 'eligible_days': len(equity),
        'predicted_mean_log_return': float(frame.predicted.mean()),
        'actual_mean_log_return': float(frame.actual_log.mean()),
        'actual_mean_net_trade_return': float(frame.actual_net.mean()),
        'selected_mean_log_overestimate': float((frame.predicted - frame.actual_log).mean()),
        'losing_entries': int(frame.actual_net.lt(0).sum()),
        'both_etf_labels_negative_at_entry_clock': int((frame.actual_net.lt(0) & frame.other_net.lt(0)).sum()),
        'warning': 'This is issued-choice diagnostic evidence from exposed cases, not a new threshold or independent proof. Trade-average is not monthly capital return.'}
    (run / 'selection_diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
    diagnose(parser.parse_args().run)
