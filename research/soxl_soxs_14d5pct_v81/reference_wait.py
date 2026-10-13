"""Independent pair calendar, label construction and ridge normal equations."""
import numpy as np
import pandas as pd


def bank(panel,market,records,clocks):
    byday={d:g for d,g in panel.groupby('date',sort=True)};result=[]
    for d,g in byday.items():
        expected=[pd.Timestamp(str(d.date())+' '+c,tz='America/New_York').tz_convert('UTC') for c in clocks]
        expected=[t for t in expected if t<pd.Timestamp(market.calendar[d]['close_at'])]
        if list(g.index)!=expected or not g.feature_ready.all() or not all(records[t]['events']['recent_ratio']['ready'] for t in expected):continue
        later=[x for x in market.sessions if x>d];nextdate=later[0] if later else None
        nexttime=pd.Timestamp(str(nextdate.date())+' '+clocks[0],tz='America/New_York').tz_convert('UTC') if nextdate is not None else None
        release=None
        if nextdate in byday:
            ng=byday[nextdate];expected2=[pd.Timestamp(str(nextdate.date())+' '+c,tz='America/New_York').tz_convert('UTC') for c in clocks]
            expected2=[t for t in expected2 if t<pd.Timestamp(market.calendar[nextdate]['close_at'])]
            valid=list(ng.index)==expected2 and ng.feature_ready.all() and nexttime in records and records[nexttime]['events']['recent_ratio']['ready']
            if valid and g.terminal_release.notna().all() and ng.terminal_release.notna().all():release=max([nexttime]+g.terminal_release.tolist()+ng.terminal_release.tolist())
        result.append({'date':d,'members':expected,'next_date':nextdate,'next_at':nexttime,'release':release,
            'expired_after':nextdate+pd.Timedelta(days=14) if nextdate is not None else None})
    return result


def parents(bank,cutoff):
    today=cutoff.tz_convert('America/New_York').tz_localize(None).normalize();prior=[p for p in bank if p['date']<today];out=[];pending=None
    for p in prior:
        if p['release'] is None or p['release']>=cutoff:pending=p;break
        out.append(p)
    return out[-120:],{'cutoff':cutoff,'mature_prefix_dates':len(out),'first_pending':pending['date'] if pending else None,
        'expired_unresolved_prefix':bool(pending and pending['expired_after'] is not None and pending['expired_after']<=today)}


def x(row,e,fields):
    side=0 if e['top_symbol']=='SOXL' else 1;raw=row[fields].to_numpy(float);ids=[0,1,2,3,4,5,6,7,8,9,10,16,17]
    vals=[raw[j]*(-1 if side==1 and j in {0,1,2,5,6,7,8,9,10} else 1) for j in ids]
    return np.array(vals+[side,e['values'][side],e['values'][1-side]],float),side


def data(panel,parents,records,fields,clocks):
    weights=[];X=[];Y=[];cs=[];members=[];b=[2**(-(len(parents)-1-i)/63) for i in range(len(parents))]
    ess=sum(b)**2/sum(v*v for v in b);dw=[v*ess/sum(b) for v in b]
    for i,p in enumerate(parents):
        future=records[p['next_at']]['events']['recent_ratio']['action']
        for t in p['members']:
            row=panel.loc[t];xx,j=x(row,records[t]['events']['recent_ratio'],fields)
            y0=float(row['L_log' if j==0 else 'S_log']);y1=0. if future=='CASH' else float(panel.loc[p['next_at'],'L_log' if future=='SOXL' else 'S_log'])
            X.append(xx);Y.append([y0,y1]);cs.append(clocks.index(row.clock));weights.append(dw[i]/len(p['members']))
            members.append({'at':t,'side':j,'wait_at':p['next_at'],'wait_action':future})
    return np.array(X),np.array(Y),np.array(cs),np.array(weights),{'dates':[p['date'] for p in parents],'members':members,'real_dates':len(parents),'date_ess':ess,'date_weights':dw}


def solve(X,Y,clock,w,mode):
    ids=list(range(16)) if mode=='state' else [13,14,15] if mode=='score' else []
    total=sum(w);center=np.array([sum(w*X[:,j])/total for j in ids]);scale=np.array([np.sqrt(sum(w*(X[:,j]-center[k])**2)/total) for k,j in enumerate(ids)]);scale[scale<1e-12]=1.
    A=np.c_[np.eye(8)[clock],(X[:,ids]-center)/scale] if mode!='mean' else np.zeros((len(w),0))
    acenter=np.array([sum(w*A[:,j])/total for j in range(A.shape[1])]);b=np.array([sum(w*Y[:,j])/total for j in range(2)])
    Z=A-acenter;pen=np.array([10.]*8+[100.]*len(ids)) if mode!='mean' else np.zeros(0)
    coef=np.linalg.solve(Z.T@(w[:,None]*Z)+np.diag(pen),Z.T@(w[:,None]*(Y-b))) if len(pen) else np.zeros((0,2))
    return {'mode':mode,'ids':ids,'center':center,'scale':scale,'design_center':acenter,'intercept':b,'coefficients':coef,'penalty':pen,'weights':w,'rows':len(w)},A


def forecast(m,x,clock):
    if m['mode']=='mean':return np.array(m['intercept'])
    a=[float(j==clock) for j in range(8)]+[(x[j]-m['center'][k])/m['scale'][k] for k,j in enumerate(m['ids'])]
    return np.array([m['intercept'][s]+sum((a[j]-m['design_center'][j])*m['coefficients'][j][s] for j in range(len(a))) for s in range(2)])
