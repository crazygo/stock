"""Offline source-readiness audit; acquisition counts never imply training readiness."""
import argparse
import hashlib
import json
from pathlib import Path
from announcement_calendar import PANEL, snapshot, validate

SOURCE = Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())


def audit():
    examples = read(SOURCE / 'verified_examples.json')
    validate(examples['records'])
    acquisitions = read(SOURCE / 'evidence/body_acquisition.json')
    rows = []
    inputs = [SOURCE / 'verified_examples.json', SOURCE / 'evidence/body_acquisition.json']
    for symbol in PANEL:
        f = SOURCE / 'evidence/indexes' / (symbol + '.json')
        inputs.append(f); d = read(f)
        records = d['records']
        body = [r for r in acquisitions if r['symbol'] == symbol]
        rows.append(dict(symbol=symbol, index_records=len(records),
                         index_range_traversed=d.get('index_lower_boundary_reached', bool(d.get('years'))),
                         index_method=d.get('method', 'cached_official_html_pagination'),
                         index_error=d.get('error'),
                         body_attempts=len(body),
                         complete_html=sum(r['status'] == 'html_downloaded_body_review_pending' for r in body),
                         body_failures=sum(r['status'] == 'failed' for r in body),
                         verified_example_sources=len({r['source_url'] for r in examples['records'] if r['symbol'] == symbol}),
                         full_body_catalogue_verified=False,
                         initial_pending_event_inventory_resolved=False))
    query = snapshot(examples['records'], '2024-05-01T09:25:30-04:00', '2024-05-14', examples['coverage'])
    assert query['count_in_window'] is None and len(query['missing_coverage']) == 8
    return dict(schema='semiconductor_announcement_source_audit_v1', requested_start='2022-10-01', requested_end='2026-04-30',
                panel=list(PANEL), issuers=rows, verified_examples_only=True,
                verified_example_sources=len({r['source_url'] for r in examples['records']}),
                verified_event_versions=len(examples['records']),
                training_ready=False, new_model_trained=False, new_capital_backtest=False,
                case_regression_run=False, goal_achieved=False,
                gate_reasons=['Historical publication catalogues incomplete for NVDA/AMD/INTC.',
                              'Full title/body/event reconciliation has not been completed for any issuer.',
                              'Initial pending announcements and later revisions require reconciliation.',
                              'Verified examples are semantic tests, not a complete training dataset.'],
                unknown_coverage_query=query,
                inputs={str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path); args = p.parse_args()
    result = audit()
    if args.output:
        # Receipts are immutable; a replay must use a fresh filename.
        with args.output.open('x') as f:json.dump(result, f, indent=2, ensure_ascii=False); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['unknown_coverage_query', 'inputs', 'issuers']}, indent=2))
