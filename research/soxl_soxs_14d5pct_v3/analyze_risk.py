"""Current research entry; unqualified policies abstain before suggesting buys."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from risk import core
from position import manage

DEFAULT_RUN = core.ROOT / '.cache/soxl_soxs_14d5pct_v3/current'


def analyze(run, at, data_root=core.DATA_ROOT, position=None, current=False):
    at = core.timestamp(at)
    result = {'analysis_at': at, 'analysis_at_et': at.tz_convert(core.ET), 'action': 'NO_ACTION',
              'broker_order_submitted': False, 'guaranteed_nonnegative_month': False,
              'any_timestamp_supported': True, 'registered_premarket_clock_et': '09:25:30'}
    run = Path(run)
    if position is not None:
        # Lack of new-buy admission must never cancel an existing exit obligation.
        result.update(manage(core.load_market(data_root), position, at))
        result['existing_position_risk_plan'] = 'Retain the actual attached broker orders; this command does not retrofit an unqualified research policy'
        return result
    if not (run / 'results.json').is_file():
        result['reason'] = 'risk_validation_missing'
        return result
    scores = json.loads((run / 'results.json').read_text())
    result['acceptance'] = scores['acceptance']
    if not scores['acceptance']['new_buy_eligible']:
        result.update({'reason': 'risk_policy_not_qualified', 'selected_research_policy': scores['selected_policy'],
                       'failed_months': scores['frozen_candidate']['negative_months'],
                       'data_refresh_skipped': 'No admitted policy; fetching a newer quote cannot grant research acceptance'})
        return result
    # No policy from the present run reaches this branch. Keep the executable
    # command fail-closed until a separate prospective paper protocol is registered.
    result['reason'] = 'prospective_paper_launch_not_registered'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--at')
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--data-root', type=Path, default=core.DATA_ROOT)
    p.add_argument('--position-json', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = analyze(args.run, args.at or datetime.now(timezone.utc).isoformat(), args.data_root,
                     json.loads(args.position_json.read_text()) if args.position_json else None, args.at is None)
    if args.output:
        core.dump(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
