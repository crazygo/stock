"""Read-only, one symbol timestamp diagnostic. Run inside a bounded parent."""
from pathlib import Path
import datetime, hashlib, json, time
import futu as ft
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.cache/soxl_soxs_14d5pct_v76/time_probe'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'Preserve prior diagnostic'
    OUT.mkdir(parents=True)
    ft.SysConfig.enable_proto_encrypt(False)
    q = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
    rows = []
    try:
        ret, quota = q.get_history_kl_quota(get_detail=False)
        assert ret == ft.RET_OK and quota[1] >= 1
        for name, kind, session in [('regular', ft.KLType.K_5M, ft.Session.RTH),
                                    ('daily', ft.KLType.K_DAY, ft.Session.RTH)]:
            time.sleep(1.2)
            before = datetime.datetime.now(datetime.timezone.utc).isoformat()
            ret, data, key = q.request_history_kline('US.SOXL', start='2023-04-03',
                end='2023-04-05', ktype=kind, autype=ft.AuType.NONE,
                max_count=1000, session=session)
            assert ret == ft.RET_OK and key is None, 'Incomplete diagnostic'
            path = OUT / (name + '.parquet')
            data.to_parquet(path, index=False, compression='zstd', compression_level=7)
            rows.append(dict(name=name, rows=len(data), path=str(path.relative_to(ROOT)),
                sha256=sha(path), requested_at=before,
                received_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        (OUT / 'receipt.json').write_text(json.dumps(dict(ok=True, quota=dict(used=int(quota[0]),
            remaining=int(quota[1])), captures=rows, source_sha256=sha(Path(__file__))), indent=2)+'\n')
        print({'ok': True, 'captures': rows}, flush=True)
    finally:
        q.close()


if __name__ == '__main__':
    main()
