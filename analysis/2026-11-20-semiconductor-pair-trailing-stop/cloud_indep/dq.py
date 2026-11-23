import pandas as pd, numpy as np
def load_futu(s):
    d=pd.read_parquet(f'data/futu_{s}_5m_raw.parquet'); d['tk']=pd.to_datetime(d.time_key); return d
def load_r2(s):
    d=pd.read_parquet(f'data/r2/{s}_2026.parquet'); d['tk']=pd.to_datetime(d.time_key); return d
def checks(name,d):
    out={}
    out['rows']=len(d); out['range']=(str(d.tk.min()),str(d.tk.max()))
    out['dup_ts']=int(d.tk.duplicated().sum())
    out['nonpos']=d[(d[['open','high','low','close']]<=0).any(axis=1)].tk.astype(str).tolist()[:10]
    bad=d[(d.high<d[['open','close']].max(axis=1)-1e-9)|(d.low>d[['open','close']].min(axis=1)+1e-9)|(d.high<d.low)]
    out['ohlc_bad_n']=len(bad); out['ohlc_bad']=bad[['tk','open','high','low','close']].astype(str).values.tolist()[:8]
    z=d[d.volume<=0]; out['vol0_n']=len(z); out['vol0']=z.tk.astype(str).tolist()[:10]
    st=d.tk.sort_values().diff().dt.total_seconds()/60
    out['sorted']=bool(d.tk.is_monotonic_increasing)
    gaps=d.tk.sort_values()[st>5]
    g=[(str(a-pd.Timedelta(minutes=m)),str(a),int(m)) for a,m in zip(gaps, st[st>5]) if m!=5]
    out['gap_n']=len(g); out['dow']=d.tk.dt.day_name().value_counts().to_dict()
    return out,g
for s in ['SOXL','SOXS']:
    for name,f in [('futu',load_futu),('r2',load_r2)]:
        d=f(s); o,g=checks(name,d); print(s,name,o)
        # show unusual gaps (not weekend/overnight standard)
        print('  gaps sample (prev,next,min):', [x for x in g if x[2] not in (5,)][:12])
