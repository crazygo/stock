"""Small registered premarket timing follow-up; retain the failed first route."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

import core
from ledger import replay, statistics
import model
from study import save_replay

SOURCE = Path(__file__).resolve().parent


def study(data_root, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory')
    output.mkdir(parents=True)
    protocol = json.loads((SOURCE / 'protocol_clock_v1.json').read_text())
    core.dump(output / 'registration.json', {
        'created_at': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
        'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in SOURCE.iterdir() if p.suffix in {'.py', '.json'}}})
    market = core.load_market(data_root)
    core.dump(output / 'input_manifest.json', market.manifest)
    clocks = [protocol['reference_analysis_clock_et'], protocol['primary_analysis_clock_et']]
    candidates, labels, metadata = core.candidate_table(market, clocks)
    candidates.to_parquet(output / 'features.parquet', compression='zstd', compression_level=7)
    core.dump(output / 'labels.json', labels)
    core.dump(output / 'candidate_coverage.json', metadata)
    evaluation = candidates[(candidates.date >= pd.Timestamp('2026-05-01')) & (candidates.date <= pd.Timestamp('2026-09-30'))]
    results = {}; primary_actions = None; primary_candidates = None
    for clock in clocks:
        table = evaluation[evaluation.clock == clock]
        actions = model.actions(table, None, 'trend20')
        name = 'trend20_premarket' if clock == protocol['primary_analysis_clock_et'] else 'trend20_0855_reference'
        outcome = replay(market, table, labels, actions, '2026-05-01', '2026-09-30')
        save_replay(output, name, outcome)
        score = statistics(outcome[0], outcome[1], ['2026-05', '2026-07', '2026-09'])
        score['scheduled_analysis_clock_et'] = clock
        score['algorithm'] = 'fixed SOXX 20-completed-session return sign'
        results[name] = score
        if name == 'trend20_premarket':
            primary_actions = actions; primary_candidates = table
        print(name, score['primary_average'], score['primary_pass'], flush=True)
    # Stress is diagnostic; it cannot replace the registered primary result.
    doubled_labels = [core.label_entry(market, symbol, at, cost_multiple=2)
                      for at in primary_candidates.index for symbol in core.ETFS]
    outcome = replay(market, primary_candidates, doubled_labels, primary_actions,
                     '2026-05-01', '2026-09-30', cost_multiple=2)
    save_replay(output, 'double_cost', outcome)
    results['double_cost'] = statistics(outcome[0], outcome[1], ['2026-05', '2026-07', '2026-09'])
    results['acceptance'] = {
        'primary_metric': protocol['primary_metric'], 'primary_clock_et': protocol['primary_analysis_clock_et'],
        'adaptive_model_pass': False,
        'registered_trend_baseline_pass': results['trend20_premarket']['primary_pass'],
        'any_registered_strategy_pass': results['trend20_premarket']['primary_pass'],
        'no_every_month_requirement': True, 'all_wins_losses_and_pending_retained': True,
        'prospective_effectiveness_established': False,
        'first_model_route_remains_failed': True,
    }
    core.dump(output / 'results.json', results)
    core.dump(output / 'status.json', {'stage': 'replay_complete', 'acceptance': results['acceptance']})
    print(json.dumps(results['acceptance'], indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=core.DATA_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); study(args.data_root, args.output)
