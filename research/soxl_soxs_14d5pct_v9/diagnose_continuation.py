"""Post-run attribution only; this command cannot change models or decisions."""
import argparse
import json
from pathlib import Path
import pandas as pd
from continuation import core


def diagnose(run):
    run = Path(run)
    results = json.loads((run / 'results.json').read_text())
    trades = pd.read_csv(run / (results['primary_name'] + '_trades.csv'))
    events = json.loads((run / 'manager_predictions.json').read_text())
    trades['entry_day'] = pd.to_datetime(trades.entry_date).dt.date
    trades['exit_day'] = pd.to_datetime(trades.exit_at, utc=True).dt.tz_convert(core.ET).dt.date
    same = trades.entry_day.eq(trades.exit_day)
    actual_reviews = []
    for t in trades.itertuples():
        key = pd.Timestamp(t.decision_at).isoformat() + '|' + t.symbol
        reviews = [e for e in events if e['key'] == key and e['cost_path'] == 'base'
                   and pd.Timestamp(e['at']) < pd.Timestamp(t.exit_at)]
        if reviews:
            actual_reviews.append({'entry_date': t.entry_date, 'symbol': t.symbol, 'outcome': t.reason,
                                   'net_return': t.net_return, 'reviews': reviews})
    case_events = [e for e in events if pd.Timestamp('2026-05-01', tz=core.ET) <= pd.Timestamp(e['at'])
                   < pd.Timestamp('2026-10-01', tz=core.ET) and e['cost_path'] == 'base']
    anchors = []
    for date in ['2026-05-01', '2026-07-01', '2026-09-01']:
        r = json.loads((run / 'training' / (date + '.json')).read_text())
        anchors.append({'date': date, 'cutoff': r['cutoff'], 'latest_label_release': r['latest_label_release'],
                        'entry_first_training_date': r['fits']['managed']['dates'][0],
                        'entry_last_training_date': r['fits']['managed']['dates'][-1],
                        'entry_training_dates': len(r['fits']['managed']['dates']),
                        'manager_training_dates': len(r['fits']['manager']['dates'])})
    out = {'post_run_only': True, 'models_and_actions_unchanged': True, 'actual_entries': len(trades),
           'same_day_exits': int(same.sum()), 'same_day_risk_stops': int((same & trades.reason.eq('risk_stop')).sum()),
           'target_exits': int(trades.reason.eq('target').sum()), 'risk_stops': int(trades.reason.eq('risk_stop').sum()),
           'actual_continuation_exits': int(trades.reason.eq('continuation_exit').sum()),
           'counterfactual_case_reviews': len(case_events),
           'counterfactual_case_exit_intents': sum(e['action'] == 'EXIT' for e in case_events),
           'actual_holding_reviews': actual_reviews, 'training_anchors': anchors,
           'interpretation': 'Daily manager was ready, but most selected trades reached a barrier before the next review. '
                             'Its exits on unselected hypothetical episodes are not actual capital gains. '
                             'Historical failure motivates earlier observable risk information, not case-specific thresholds.'}
    core.dump(run / 'diagnosis.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'actual_holding_reviews'}, indent=2))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
    diagnose(parser.parse_args().run)
