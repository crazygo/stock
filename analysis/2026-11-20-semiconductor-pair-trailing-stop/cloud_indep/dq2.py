import pandas as pd, numpy as np
exec(open('dq.py').read().split('for s in')[0])
for s in ['SOXL','SOXS']:
    f=load_futu(s); r=load_r2(s)
    _,gf=checks('f',f); _,gr=checks('r',r)
    print(s,'futu unusual gaps:',[x for x in gf if x[2] not in (2885,)])
    print(s,'r2 unusual gaps:',[x for x in gr if x[2] not in (485,3365)][:40])
    lo,hi=pd.Timestamp('2026-07-20'),pd.Timestamp('2026-09-25 20:00')
    hhmm=f.tk.dt.hour*60+f.tk.dt.minute
    fo=f[(f.tk>=lo)&(f.tk<=hi)&(hhmm>240)&(hhmm<=1200)]
    ro=r[(r.tk>=lo)&(r.tk<=hi)]
    m=fo.merge(ro,on='tk',how='outer',suffixes=('_f','_r'),indicator=True)
    print(s,'overlap rows futu',len(fo),'r2',len(ro),m._merge.value_counts().to_dict())
    print('  only futu:',m[m._merge=='left_only'].tk.astype(str).tolist()[:15])
    print('  only r2:',m[m._merge=='right_only'].tk.astype(str).tolist()[:15])
    b=m[m._merge=='both'].copy()
    for c in ['open','high','low','close']:
        b['d_'+c]=(b[c+'_f']/b[c+'_r']-1).abs()
    b['dmax']=b[['d_open','d_high','d_low','d_close']].max(axis=1)
    b['dvol']=(b.volume_f/b.volume_r.replace(0,np.nan)-1).abs()
    print('  exact OHLC equal:',int((b.dmax==0).sum()),'/',len(b),' >0.5%:',int((b.dmax>0.005).sum()),' >0.1%:',int((b.dmax>0.001).sum()))
    print('  vol exact equal:',int((b.volume_f==b.volume_r).sum()),' vol diff>10%:',int((b.dvol>0.1).sum()))
    w=b.sort_values('dmax',ascending=False).head(8)
    print(w[['tk','open_f','open_r','high_f','high_r','low_f','low_r','close_f','close_r','volume_f','volume_r']].to_string(index=False))
    if s=='SOXS':
        z=r[r.volume<=0]; print('  r2 vol0 dates:', z.tk.dt.date.value_counts().sort_index().to_dict())
        zz=z.merge(f,on='tk',suffixes=('_r','_f')); print('  r2 vol0 bars, futu view:'); print(zz[['tk','open_r','close_r','volume_r','open_f','close_f','volume_f']].head(6).to_string(index=False))
