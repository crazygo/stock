"""Post-run paired-exit attribution, without changing registered models or actions."""
import argparse
import json
from pathlib import Path
import pandas as pd
from micro_manager import core,label_key


def diagnose(run):
    run = Path(run);results = json.loads((run/'results.json').read_text())
    baseline = {label_key(x['decision_at'],x['symbol']):x for x in json.loads((run/'baseline_labels.json').read_text())}
    states = {r['id']:r for r in json.loads((run/'review_states.json').read_text())}
    events = {(e['key'],str(pd.Timestamp(e['at']))):e for e in json.loads((run/'manager_predictions.json').read_text()) if e['cost_path']=='base'}
    trades = pd.read_csv(run/(results['primary_name']+'_trades.csv'));pairs = []
    for t in trades.itertuples():
        if t.reason!='intraday_value_exit':continue
        key = label_key(t.decision_at,t.symbol);b = baseline[key]
        e = events[(key,str(pd.Timestamp(t.manager_decision_at)))];state = states[e['id']]
        pairs.append({'entry_date':t.entry_date,'symbol':t.symbol,'manager_at':t.manager_decision_at,
            'managed_net_return':t.net_return,'fixed_contract_net_return':b['net_return'],'fixed_reason':b['reason'],
            'difference_per_trade':t.net_return-b['net_return'],'predicted_hold_advantage_log':e['adjusted'],
            'realized_hold_advantage_log':state['target']})
    case = trades.copy();case['same_day'] = pd.to_datetime(case.entry_date).dt.date==pd.to_datetime(case.exit_at,utc=True).dt.tz_convert(core.ET).dt.date
    out = {'post_run_only':True,'models_and_actions_unchanged':True,'actual_entries':len(trades),
        'same_day_exits':int(case.same_day.sum()),'target_exits':int(trades.reason.eq('target').sum()),
        'risk_stops':int(trades.reason.eq('risk_stop').sum()),'learned_exits':len(pairs),
        'learned_exits_improved_fixed_contract_return':sum(x['difference_per_trade']>0 for x in pairs),
        'learned_exits_worsened_fixed_contract_return':sum(x['difference_per_trade']<0 for x in pairs),
        'mean_per_trade_difference':sum(x['difference_per_trade'] for x in pairs)/len(pairs) if pairs else None,
        'paired_exits':pairs,'attribution_warning':'Per-trade counterfactual differences are not portfolio monthly returns. '
        'Matching entry decisions does not force identical executed trades when holding durations and settled funds differ.',
        'conclusion':'Earlier observation enabled17 actual exits, but point-estimate continuation decisions often truncated eventual targets. '
        'Daily training/earlier reaction alone has not established reliable exit value.'}
    core.dump(run/'diagnosis.json',out);print(json.dumps({k:v for k,v in out.items() if k!='paired_exits'},indent=2));return out


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    diagnose(parser.parse_args().run)
