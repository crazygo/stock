import pandas as pd, sys, time
from futu import OpenQuoteContext, KLType, AuType, RET_OK, Session
q = OpenQuoteContext(host='127.0.0.1', port=11111)
for code in ['US.SOXL','US.SOXS']:
    frames=[]; key=None; n=0
    while True:
        ret, df, key = q.request_history_kline(code, start='2026-07-20', end='2026-11-23', ktype=KLType.K_5M, autype=AuType.NONE, max_count=1000, page_req_key=key, extended_time=True, session=Session.ALL)
        n+=1
        if ret != RET_OK: print(code, 'ERR', df, flush=True); break
        print(code, 'page', n, len(df), df.time_key.iloc[0], df.time_key.iloc[-1], 'next', key is not None, flush=True)
        frames.append(df)
        if key is None: break
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(f'data/futu_{code[3:]}_5m_raw.parquet')
    print(code, len(df), df.time_key.min(), df.time_key.max(), flush=True)
ret, quota = q.get_history_kl_quota(get_detail=False); print('quota', quota, flush=True)
q.close()
