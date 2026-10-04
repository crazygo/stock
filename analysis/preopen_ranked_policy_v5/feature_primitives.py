"""Frozen v3 segment definitions, extracted without changing arithmetic.

This module removes the runtime dependency on uncommitted v2/v3 experiments.
The fixed related-stock groups are retrospective, not historical PIT membership.
"""
import numpy as np

GROUPS={
 'optics':['AAOI','COHR','LITE','CRDO','CIEN','MRVL','VRT','ALAB'],
 'chips':['AMD','AVGO','ARM','AMAT','QCOM','SNPS','TER','RMBS','SMTC','NVDA','INTC'],
 'memory':['MU','WDC','STX'],
 'cloud':['NBIS','CRWV','ORCL','PLTR','GOOG'],
 'quantum':['IONQ','RGTI','QBTS'],
 'biology':['TXG','TWST','SDGR','LIFE'],
 'other':['AXTI','NOK','CBRS']}
GROUP_OF={s:k for k,v in GROUPS.items() for s in v}

def stat(g,prefix,anchor):
 if g is None or not len(g):return {prefix+'_missing':1.,prefix+'_count':0.,prefix+'_volume':np.nan}
 o,h,l,c,v=[g[k].to_numpy(float) for k in ['open','high','low','close','volume']]
 step=np.diff(np.log(np.r_[o[0],c]));travel=np.abs(step).sum();total=v.sum();spread=max(h.max()-l.min(),1e-9)
 weights=v/total if total else np.zeros(len(v));vw=np.sum((h+l+c)/3*v)/total if total else np.nan
 signed=np.sign(c-o);last=max(1,len(v)//3)
 corr=float(np.corrcoef(np.abs(step),np.log1p(v))[0,1]) if np.std(np.abs(step))>1e-9 and np.std(np.log1p(v))>1e-9 and len(v)>2 else 0.
 return {prefix+'_missing':0.,prefix+'_count':float(len(g)),prefix+'_ret':float(c[-1]/o[0]-1),prefix+'_gap':float(c[-1]/anchor-1),
  prefix+'_range':float((h.max()-l.min())/anchor),prefix+'_eff':float(abs(np.log(c[-1]/o[0]))/travel) if travel else 0.,
  prefix+'_position':float((c[-1]-l.min())/spread),prefix+'_recovery':float(c[-1]/l.min()-1),prefix+'_fade':float(c[-1]/h.max()-1),
  prefix+'_trough':float(np.argmin(l)/max(1,len(l)-1)),prefix+'_volume':float(total),prefix+'_active':float((v>0).mean()),
  prefix+'_vol_hhi':float(np.sum(weights**2)),prefix+'_signed_volume':float(np.sum(v*signed)/total) if total else 0.,
  prefix+'_late_volume':float(v[-last:].sum()/max(total,1)),prefix+'_burst':float(v.max()/max(v.mean(),1)),
  prefix+'_bar_vwap_gap':float(c[-1]/vw-1) if np.isfinite(vw) else np.nan,prefix+'_vol_price_corr':corr,
  prefix+'_rv':float(np.sqrt(np.sum(step**2))),prefix+'_positive':float((step>0).mean())}
