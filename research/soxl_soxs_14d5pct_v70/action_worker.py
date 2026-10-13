"""One read-only quote-context corporate-action request; parent bounds lifetime."""
import argparse
import datetime
import json
from pathlib import Path
import futu as ft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True,
                        choices=['TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'SPY'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Preserve existing capture')
    ft.SysConfig.enable_proto_encrypt(False)
    quote = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
    try:
        requested = datetime.datetime.now(datetime.timezone.utc).isoformat()
        ret, data = quote.get_rehab('US.' + args.symbol)
        if ret != ft.RET_OK:
            raise RuntimeError('Corporate action response unsuccessful')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(args.output, index=False, compression='zstd', compression_level=7)
        print('@@TRANSFER_ACTION@@ ' + json.dumps({
            'symbol': args.symbol, 'ok': True, 'rows': len(data),
            'requested_at': requested,
            'received_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}),
            flush=True)
    finally:
        quote.close()


if __name__ == '__main__':
    main()
