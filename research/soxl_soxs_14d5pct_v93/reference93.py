"""Independent scalar construction of date-weighted paired candidates."""
import numpy as np
import pandas as pd
from reference91 import coordinates,matrix,posterior,loss_gradient,solve,predict


def data(panel,parents,history,clocks):
    rows=[];masses=[];dates=[];N=len(parents)
    for i,p in enumerate(parents):
        daily=[]
        for timestamp in p['members']:
            t=pd.Timestamp(timestamp);e=history[t]['events']['recent_ratio'];r=panel.loc[t]
            for side,column in [(0,'L_gap'),(1,'S_gap')]:
                daily.append([t,side,list(e['probabilities'][side]),int(r[column]),coordinates(r,side),clocks.index(r['clock'])])
        if daily:
            mass=2**(-(N-i-1)/63);masses.append(mass);dates.append(p['date'])
            rows.extend([*r,mass/len(daily)] for r in daily)
    total=sum(masses);ess=total*total/sum(x*x for x in masses) if masses else 0.
    weights=np.array([r[-1]*ess/total for r in rows])
    meta=dict(dates=dates,real_dates=len(dates),date_ess=ess,members=[[r[0],r[1]] for r in rows],weights=weights)
    return np.array([r[2] for r in rows]).reshape(-1,4),np.array([r[3] for r in rows],int),weights,np.array([r[4] for r in rows]).reshape(-1,19),np.array([r[1] for r in rows],int),np.array([r[5] for r in rows],int),meta
