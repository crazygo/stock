"""Bounded parent-process worker: read membership only; never save account IDs."""
import argparse
import json
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False)
    result = {'ok': True, 'observed_at': datetime.now(timezone.utc).isoformat(),
              'watchlists': {}, 'holdings': [], 'errors': [], 'source': 'OpenD read-only membership'}
    q = ft.OpenQuoteContext(host=args.host, port=args.port)
    try:
        ret, groups = q.get_user_security_group()
        if ret != ft.RET_OK:
            raise RuntimeError('group enumeration failed')
        result['group_names'] = list(groups.group_name)
        for name in ['全部', '特别关注']:
            if name not in result['group_names']:
                result['errors'].append({'group': name, 'reason': 'actual_group_absent'})
                continue
            ret, rows = q.get_user_security(group_name=name)
            if ret != ft.RET_OK:
                result['errors'].append({'group': name, 'reason': 'query_failed'})
                continue
            result['watchlists'][name] = [{k: str(getattr(r, k, ''))
                                         for k in ['code', 'name', 'stock_type', 'listing_date']}
                                        for r in rows.itertuples()]
    finally:
        q.close()
    for firm, label in [(ft.SecurityFirm.FUTUINC, 'Moomoo US'),
                        (ft.SecurityFirm.FUTUSECURITIES, '富途证券')]:
        ctx = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US,
                                    security_firm=firm, host=args.host, port=args.port)
        try:
            ret, accounts = ctx.get_acc_list()
            if ret != ft.RET_OK:
                result['errors'].append({'group': 'holdings', 'reason': 'account_query_failed'})
                continue
            for account in accounts.itertuples():
                if str(account.trd_env) != 'REAL':
                    continue
                ret, rows = ctx.position_list_query(acc_id=int(account.acc_id), trd_env=ft.TrdEnv.REAL)
                if ret != ft.RET_OK:
                    result['errors'].append({'group': 'holdings', 'reason': 'position_query_failed'})
                    continue
                result['holdings'].extend({'code': str(r.code), 'name': str(r.stock_name),
                                           'account_type': label}
                                          for r in rows.itertuples() if float(r.qty) != 0)
        finally:
            ctx.close()
    print('@@STDV1_RESULT@@ ' + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
