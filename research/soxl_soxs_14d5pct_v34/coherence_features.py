"""Past-return synchronization and contemporaneous leadership diffusion."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v32'))
from hierarchical_error import core,np,pd
from breadth import PANEL,load_panel,PANEL_ROOT

FLOW=['participation_5','participation_now','laggard_catchup']
SYNC=['synchrony_20','synchrony_change','synchronous_flow']
COHERENCE=FLOW+SYNC


def participation(z):
    z=np.asarray(z,float);den=len(z)*float(z@z);total=float(z.sum())
    return total*abs(total)/den if den>0 else 0.


def correlation_mean(R):
    if len(R)<5 or not np.isfinite(R).all() or np.any(np.std(R,axis=0)<1e-12):return None
    C=np.corrcoef(R,rowvar=False)
    return float(C[np.triu_indices(C.shape[0],1)].mean())


def measurements(R,current):
    """R contains20completed returns; current isknownclose/lastclose-1."""
    R=np.asarray(R,float);current=np.asarray(current,float)
    if R.shape!=(20,len(PANEL)) or current.shape!=(len(PANEL),) or not np.isfinite(current).all():return None
    rho20=correlation_mean(R);rho5=correlation_mean(R[-5:])
    if rho20 is None or rho5 is None:return None
    vol=np.maximum(np.std(R,axis=0,ddof=1),1e-4)
    trend=np.clip((np.prod(1+R[-5:],axis=0)-1)/(vol*np.sqrt(5)),-5,5)
    now=np.clip(current/vol,-5,5);order=np.argsort(trend,kind='stable')
    catchup=float((now[order[:4]].mean()-now[order[4:]].mean())/2)
    flow=participation(now)
    return dict(zip(COHERENCE,[participation(trend),flow,catchup,rho20,rho5-rho20,rho20*flow]))


def build(features,leaders,sessions):
    bars,daily,_=leaders;rows=[];receipts=[]
    for at,row in features.iterrows():
        dates=sessions[sessions<row.date][-21:];closes=[];current=[];known=[];reason=None
        if len(dates)!=21:reason='fewer_than21prior_sessions'
        for s in PANEL:
            if reason:break
            d=daily[s].reindex(dates)
            if (d.complete.isna().any() or not d.complete.all() or d.close.isna().any() or
                d.available_at.isna().any() or not d.available_at.lt(at).all() or not d.close.gt(0).all()):
                reason=s+':incomplete_or_unavailable_history';break
            f=bars[s].loc[:at];f=f[(f.available_at<=at)&f.date.eq(row.date)]
            if not len(f) or (at-f.index[-1]).total_seconds()>360 or not np.isfinite(f.econ_close.iloc[-1]) or f.econ_close.iloc[-1]<=0:
                reason=s+':current_bar_unavailable';break
            prices=d.close.to_numpy(float);closes.append(prices);current.append(float(f.econ_close.iloc[-1]/prices[-1]-1))
            known.append({'symbol':s,'last_bar_end':f.index[-1],'last_available':f.available_at.iloc[-1],
                          'history_latest_available':d.available_at.max(),'history_last_date':dates[-1]})
        values=None
        if not reason:
            C=np.array(closes).T;R=C[1:]/C[:-1]-1;values=measurements(R,current)
            if values is None:reason='zero_variance_or_nonfinite_returns'
        rows.append({'at':at,'coherence_ready':values is not None,**(values or {c:np.nan for c in COHERENCE})})
        receipts.append({'at':at,'date':row.date,'history_dates':list(dates),'ready':values is not None,'reason':reason,'stocks':known})
    return pd.DataFrame(rows).set_index('at'),receipts
