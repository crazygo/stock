"""Generate an offline standalone map and a focused in-conversation fragment."""
from pathlib import Path
import argparse,json

OUT=Path(__file__).resolve().parent

def compact(d):
    keep=['status','as_of','start','actual_last_day','window','x','y','p','dominant','entropy','direction','price_return','xy_interval',
        'measurements','z','sensitivity','clear_tendency','observed_stable_weeks','calendar_verified','low_liquidity_observed']
    history_keep=['status','as_of','x','y','p','dominant','entropy','direction','price_return','observed_stable_weeks','calendar_verified']
    records=[]
    for r in d['records']:
        row={k:r.get(k) for k in ['code','name','kind','symbol','favorite','ai_related','quality_grade','structure']}
        row['coverage']={k:r.get('coverage',{}).get(k) for k in ['source','last_day','status','fetch_error']}
        row['windows']={n:{k:t[k] for k in keep if k in t} for n,t in r['windows'].items()}
        row['history']=[[t.get(k) for k in history_keep] for t in r['history']]
        records.append(row)
    bundle={**{k:d[k] for k in ['model_version','run_id','as_of','watchlist_observed_at','summary','classes','feature_names','datum','weight_x','weight_y','history_dates','excluded']},
        'history_fields':history_keep,'quality_as_of':max((r.get('quality_as_of') or '' for r in d['records']),default='') or '日期未记录','records':records}
    def rounded(v):
        if isinstance(v,float):return round(v,6)
        if isinstance(v,list):return [rounded(x) for x in v]
        if isinstance(v,dict):return {k:rounded(x) for k,x in v.items()}
        return v
    return rounded(bundle)

def main():
    p=argparse.ArgumentParser();p.add_argument('--inline-dir',type=Path);a=p.parse_args()
    d=compact(json.loads((OUT/'results.json').read_text()));template=(OUT/'map_template.html').read_text()
    def fragment(standalone):
        data=json.dumps(dict(d,standalone=standalone),ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
        return template.replace('__DATA__',data)
    inline=fragment(False);inline_path=None
    if a.inline_dir:
        a.inline_dir.mkdir(parents=True,exist_ok=True);inline_path=a.inline_dir/'watchlist-trend-quadrants.html'
        if len(inline.encode())>=1_000_000:raise ValueError('Inline visualization exceeds 1 MB')
        inline_path.write_text(inline)
    base='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>自选四象限概率地图</title><style>
    :root{--background:#fff;--foreground:#202020;--muted:#ededed;--border:#c4c4c4;color-scheme:light dark;font-family:system-ui,-apple-system,sans-serif;font-size:14px;}
    @media(prefers-color-scheme:dark){:root{--background:#191919;--foreground:#e7e7e7;--muted:#333;--border:#555;}}
    *{box-sizing:border-box}body{margin:0;background:var(--background);color:var(--foreground);line-height:1.5}main{max-width:1080px;padding:22px;margin:auto}h2{font-size:21px;margin:0 0 8px;font-weight:500}.text-small{font-size:12px}.viz-row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:16px 0}.btn{font:inherit;background:var(--background);color:var(--foreground);border:1px solid var(--border);padding:6px 9px;border-radius:2px}.btn-ghost{border:0;text-decoration:underline}.form-select{font:inherit;width:100%;min-width:0;padding:6px;border:1px solid var(--border);background:var(--background);color:var(--foreground)}.table{border-collapse:collapse;font-size:12px;width:100%}.table td,.table th{padding:7px;text-align:right;border-bottom:1px solid var(--border)}.table td:first-child,.table th:first-child{text-align:left}summary{cursor:pointer}p{max-width:80ch}strong{font-weight:500}button:focus-visible,select:focus-visible{outline:2px solid var(--foreground);outline-offset:2px}@media(max-width:460px){main{padding:12px}.btn,.form-select{min-height:44px}.form-select{font-size:16px}}
    </style></head><body><main>'''
    (OUT/'index.html').write_text(base+fragment(True)+'</main></body></html>\n')
    print(json.dumps({'standalone':str(OUT/'index.html'),'inline':str(inline_path) if inline_path else None,'inline_bytes':len(inline.encode())}))

if __name__=='__main__':main()
