#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from screener import load_env_file


def main() -> int:
    load_env_file(Path('.env'))
    api_key = os.environ.get('MASSIVE_API_KEY')
    if not api_key:
        print(json.dumps({'available': False, 'reason': 'missing_api_key'}))
        return 0
    params = urlencode({
        'tickers': 'NVDA',
        'timeframe': 'trailing_twelve_months',
        'limit': 1,
        'apiKey': api_key,
    })
    url = f'https://api.massive.com/stocks/financials/v1/income-statements?{params}'
    try:
        with urlopen(Request(url, headers={'Accept': 'application/json'}), timeout=30) as response:
            payload = json.loads(response.read().decode('utf-8'))
        rows = payload.get('results') if isinstance(payload, dict) else None
        print(json.dumps({
            'available': bool(isinstance(rows, list) and rows),
            'status': payload.get('status') if isinstance(payload, dict) else None,
            'rows': len(rows) if isinstance(rows, list) else 0,
        }))
    except HTTPError as exc:
        body = exc.read().decode('utf-8', errors='replace')[:500]
        print(json.dumps({'available': False, 'http_status': exc.code, 'body': body}))
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(json.dumps({'available': False, 'reason': str(exc)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
