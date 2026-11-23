import sys, json; sys.path.insert(0,'.')
from strategy import *
src = sys.argv[1]
if src == 'pkg':
    fr = {s: frame_from_standard(f'/workspace/soxl-alert/ref/market_data/us_5m/{s}/2026.parquet', s) for s in SYMS}
elif src == 'futu':
    fr = {s: frame_from_futu(f'/workspace/soxl-alert/data/futu_{s}_5m_raw.parquet', s) for s in SYMS}
fr = window(fr, '2026-11-02', '2026-11-19')
print(src, {s: len(f) for s, f in fr.items()})
out = {}
for path in ('OLHC', 'OHLC'):
    r = run(fr, path); out[path] = r
    print(f"{src} {path} paper={r['paper']} buys={r['buys']} wins={r['wins']} losses={r['losses']} fees={r['fees']:.2f} net={r['net']:.2f} open={r['open']}")
    for x in r['trades']:
        print(f"   {x['symbol']} {x['entry'][:16]} P={x['entry_px']:.4f} score={x['score']:.6f} p2={x['p2']:.4f} sigc={x['sigc']:.4f} -> {x['exit'][:16]} {x['reason']} {x['exit_px']:.6f} net={x['net']:.2f}")
json.dump(out, open(f'/workspace/soxl-alert/indep/result_{src}.json','w'), default=str, indent=1)
