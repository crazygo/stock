"""Read-only independent wireframe API. No scheduler, orders or v5 writes."""
import argparse,json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
import numpy as np
import pandas as pd
from common import OUT,OLD,DATES,SESSIONS,clean

def raw_path(symbol):
    if not symbol or not all(c.isalnum() or c in '.-' for c in symbol):raise ValueError('invalid symbol')
    p=OUT/'cache/backfill'/f'{symbol}.parquet'
    if not p.exists():p=OLD/'raw'/f'{symbol}.parquet'
    if not p.exists():p=OUT/'cache/acquired'/f'{symbol}.parquet'
    if not p.exists():raise ValueError('minute history unavailable')
    return p

@lru_cache(maxsize=8)
def raw(symbol,stamp):return pd.read_parquet(raw_path(symbol)).sort_values('start')

def chart(symbol,start,end):
    p=raw_path(symbol);f=raw(symbol,p.stat().st_mtime)
    prices=f[['open','high','low','close']].to_numpy(float)
    valid=np.isfinite(prices).all(1)&(prices>0).all(1)&np.isfinite(f.volume)&(f.volume>=0)&(f.high>=prices.max(1))&(f.low<=prices.min(1))
    invalid=int((~valid).sum());f=f[valid]
    assoc=(f.start+pd.to_timedelta((f.minute>=1200).astype(int),unit='D')).dt.strftime('%Y-%m-%d')
    g=f[(assoc>=start)&(assoc<=end)&assoc.isin(SESSIONS)]
    bars=[[str(r.start),str(r.end),float(r.open),float(r.high),float(r.low),float(r.close),float(r.volume)] for r in g.itertuples()]
    daily=[]
    for day,d in f.groupby('day'):
        if day not in SESSIONS:continue
        close=570+SESSIONS[day]['duration_minutes'];d=d[(d.minute>=570)&(d.minute<close)]
        if not len(d):continue
        complete=np.array_equal(d.minute.to_numpy(),np.arange(570,close,5))
        if not complete:continue
        daily.append(dict(day=day,o=float(d.open.iloc[0]),h=float(d.high.max()),l=float(d.low.min()),c=float(d.close.iloc[-1]),partial=False))
    return dict(symbol=symbol,bars=bars,daily=daily,calendar={k:v['duration_minutes'] for k,v in SESSIONS.items()},
        mini_range=['2024-10-03','2026-10-02'],grain_minutes=5,price_basis='NONE',invalid_bars_omitted=invalid,actual=[str(f.start.min()),str(f.end.max())])

def route(arm):
    if arm in ['A0','A1','B1','B2','C1','ALG_LR','ALG_ET']:return OUT
    if arm in ['S0','S1','S2','S3']:return OUT/'R01'
    if arm in ['P0','P1','E0','E1']:return OUT/'R02'
    if arm in ['F0','F1']:return OUT/'R04'
    if arm in ['H0','H1','H2','H3']:return OUT/'R03'
    if arm in ['D0','D1']:return OUT/'R05'
    raise ValueError('unknown registered arm')

def events(arm,variant):
    dest=route(arm)
    if variant not in ['candidate_only','top_one']:raise ValueError('unknown calibration')
    p=dest/f'{arm}_{variant}_signals.json'
    return json.loads(p.read_text())['events'] if p.exists() else []

def inspect_event(q):
    ev=next((e for e in events(q['arm'],q['variant']) if e['symbol']==q['symbol'] and e['day']==q['day']),None)
    if ev is None:raise ValueError('signal unavailable')
    ev=dict(ev)
    if not ev.get('label_end'):
        endday=DATES[DATES.index(ev['day'])+4];endminute=570+SESSIONS[endday]['duration_minutes']
        ev['label_end']=endday+f' {endminute//60:02d}:{endminute%60:02d}:00';ev['display_deadline_from_calendar']=True
    dest=route(q['arm']);meta=json.loads((dest/'cache/runs'/f"{q['arm']}_{ev['day'][:7]}"/'meta.json').read_text())
    f=pd.read_parquet(dest/'cache'/meta.get('panel_file','panel.parquet'),columns=['symbol','day','minute']+meta['features'],filters=[('symbol','==',ev['symbol']),('day','==',ev['day']),('minute','==',ev['minute'])])
    p=raw_path(ev['symbol']);r=raw(ev['symbol'],p.stat().st_mtime);entry=pd.Timestamp(ev['day'])+pd.Timedelta(minutes=ev['minute']+5)
    mask=(r.start>=entry)&(r.end<=pd.Timestamp(ev['label_end']))&(r.minute>=570)&(r.minute<r.day.map({d:570+s['duration_minutes'] for d,s in SESSIONS.items()}))
    target=ev.get('target');touches=r[mask&(r.high>=target)] if target is not None and np.isfinite(target) else r.iloc[:0]
    phase='pre' if ev['minute']<=570 else 'regular'
    # An emitted signal's selection gate required a fitted candidate calibrator.
    # Use this signal month's metadata, never the newest exported month's fit.
    calibration_status=dict(candidate_calibrated=True,
        extra_first_calibrated=q['variant']=='top_one' and meta['top_calibration_counts'][phase]['calibrator_fitted'],
        model_month=meta['month'],interpretation='research estimate; calibration fit does not prove reliability')
    return dict(event=ev,features=f.iloc[0].to_dict(),first_touch=str(touches.end.iloc[0]) if ev['y']==1 and len(touches) else None,
         algorithm=meta['algorithm'],training=meta['training'],candidate_calibration=meta['candidate_calibration'],top_calibration=meta['top_calibration'],selection=meta['selection'],calibration_status=calibration_status)

class Handler(BaseHTTPRequestHandler):
    def send(self,value,status=200,mime='application/json'):
        data=(json.dumps(clean(value),ensure_ascii=False,allow_nan=False) if mime=='application/json' else value).encode()
        self.send_response(status);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        p=urlparse(self.path);q={k:v[0] for k,v in parse_qs(p.query).items()}
        try:
            if p.path in ['/','/index.html','/opportunity.html','/sector.html','/events.html']:
                return self.send((OUT/('index.html' if p.path=='/' else p.path[1:])).read_text(),mime='text/html')
            if p.path=='/api/results':
                reports=[]
                for rid,base in [('R00',OUT),('R01',OUT/'R01'),('R02',OUT/'R02'),('R03',OUT/'R03'),('R04',OUT/'R04'),('R05',OUT/'R05')]:
                    path=base/'results.json'
                    if path.exists():
                        r=json.loads(path.read_text());reports.extend(dict(**arm,round=rid) for arm in r['arms'])
                return self.send(dict(arms=reports,status='exposed_historical_development',independent_pass=False))
            if p.path=='/api/status':
                names=['coverage','earnings_audit','sec_audit','r2_summary','history_acquisition','corporate_actions_audit']
                status={n:json.loads((OUT/(n+'.json')).read_text()) if (OUT/(n+'.json')).exists() else None for n in names}
                if q.get('arm'):
                    path=route(q['arm'])/'coverage.json';status['route_coverage']=json.loads(path.read_text()) if path.exists() else None
                return self.send(status)
            if p.path=='/api/chart':return self.send(chart(q['symbol'],q.get('start','2026-08-01'),q.get('end','2026-09-30')))
            if p.path=='/api/events':return self.send(events(q['arm'],q.get('variant','top_one')))
            if p.path=='/api/inspect':return self.send(inspect_event(q))
            return self.send({'error':'not_found'},404)
        except (KeyError,ValueError,FileNotFoundError) as e:return self.send({'error':str(e)},400)
    def log_message(self,*args):pass

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8771);a=ap.parse_args()
    print(f'http://127.0.0.1:{a.port}',flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
