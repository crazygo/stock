from pathlib import Path
import json
OUT=Path(__file__).resolve().parent

def main():
    d=json.loads((OUT/'results.json').read_text())
    def short(v):
        if isinstance(v,float):return round(v,6)
        if isinstance(v,list):return [short(x) for x in v]
        if isinstance(v,dict):return {k:short(x) for k,x in v.items()}
        return v
    data={k:d[k] for k in ['version','as_of','watchlist_observed_at','summary','records','periods','run_id']}
    text=json.dumps(short(data),ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    prices=json.loads((OUT/'prices.json').read_text())
    if prices['source_run_id']!=d['run_id'] or prices['as_of']!=d['as_of']:
        raise ValueError('Price projection and trait snapshot must have the same cutoff and run')
    price_text=json.dumps(prices,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html=(OUT/'template.html').read_text().replace('__DATA__',text).replace('__PRICES__',price_text).replace('__TIMELINE__',(OUT/'timeline.js').read_text()).replace('__PRICE_CHART__',(OUT/'price_chart.js').read_text())
    (OUT/'index.html').write_text(html)
    print(json.dumps({'path':str(OUT/'index.html'),'bytes':len(html.encode())}))

if __name__=='__main__':main()
