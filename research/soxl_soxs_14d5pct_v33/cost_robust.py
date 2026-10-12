"""Learn scenario-worst realized payoff with jointly mature cost feedback."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v32'))
from hierarchical_error import (core,np,pd,read,dump,sha,labels,prepare_inputs,base_forecast,
    day_cutoff,restrict_market,development_gate,CLOCKS,tag,columns,registry,pool,
    block_counts,decide,encode,fit_map,forecast)
from hierarchical_error import fit_day as source_fit_day,issue_hierarchical as source_issue
from resolved_feedback import prefix_receipt
from verify_continuation import key

VARIANTS=['worst','average','baseline']
ROUTES=['worst_band','worst_point','average_band','matched_baseline','source_v32','premarket_only','cash','double_cost']


def target_values(frame,kind):
    a=frame[['base_L','base_S']].to_numpy(float);b=frame[['stress_L','stress_S']].to_numpy(float)
    if kind=='worst':return np.minimum(a,b)
    if kind=='average':return (a+b)/2
    assert kind=='baseline';return a


def joint_feedback(market,panel,base,stress):
    bank=panel.copy();bank['terminal_release']=pd.Series(pd.NaT,index=bank.index,dtype='datetime64[ns, UTC]')
    for name in ['base_L','base_S','stress_L','stress_S']:bank[name]=np.nan
    receipts=[]
    for at,row in panel[panel.feature_ready].iterrows():
        sides={}
        for name,lab in [('base',base),('stress',stress)]:
            for s,suffix in zip(core.ETFS,['L','S']):
                label=lab[key(at,s)];receipt=prefix_receipt(market,label);sides[name+'_'+s]=receipt
                if receipt['known']:
                    assert label['net_return']>-1
                    bank.loc[at,name+'_'+suffix]=np.log1p(label['net_return'])
        release=max(v['release'] for v in sides.values()) if all(v['known'] for v in sides.values()) else None
        if release is not None:bank.loc[at,'terminal_release']=release
        receipts.append({'at':at,'date':row.date,'joint_terminal_release':release,'sides':sides})
    return bank,receipts


def fit_day(bank,history,reg,at):
    parents,fb=pool(reg,history,at);ids=[t for p in parents for t in p['members']];n=len(parents)
    receipt={'date':bank.loc[at,'date'],'cutoff':day_cutoff(at),'feedback':fb,'parents':parents,'members':ids,'ready':n>=60,'variants':{}}
    if n<60:return receipt
    dw=2.**(-np.arange(n)[::-1]/63.);dw*=n/dw.sum();groups=np.repeat(np.arange(n),[len(p['members']) for p in parents])
    w=np.concatenate([np.repeat(dw[j]/len(p['members']),len(p['members'])) for j,p in enumerate(parents)])
    X=encode(bank,history,ids);counts,starts,seed=block_counts(n,at,10)
    receipt.update(day_weights=dw,row_weights=w,row_groups=groups,block_starts=starts,seed=seed)
    for kind in VARIANTS:
        Y=target_values(bank.loc[ids],kind)-X[:,:2];assert np.isfinite(X).all() and np.isfinite(Y).all()
        receipt['variants'][kind]={'point_model':fit_map(X,Y,w,'partial',n),
            'bootstrap_models':[fit_map(X,Y,w*c[groups],'partial',n) for c in counts]}
    return receipt


def issue_cost(bank,at,raw,daily):
    ready=raw is not None and bank.loc[at,'feature_ready'] and daily['ready'];out={}
    for kind in VARIANTS:
        info={'ready':bool(ready),'point':None,'lower':None}
        if ready:
            x=encode(bank,{at:raw},[at])[0];m=daily['variants'][kind]
            point=raw+forecast(m['point_model'],x)[0]
            draws=np.asarray([raw+forecast(z,x)[0] for z in m['bootstrap_models']])
            radius=max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
            info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
        out[kind]=info
    return out


def events(at,date,clock,estimates,feedback,old):
    primary=decide(estimates['worst'],feedback,True);cash={'ready':True,'action':'CASH','allocation':0.}
    out={'worst_band':primary,'worst_point':decide(estimates['worst'],feedback,False),
        'average_band':decide(estimates['average'],feedback,True),'matched_baseline':decide(estimates['baseline'],feedback,True),
        'source_v32':old,'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {n:{**e,'at':at,'date':date} for n,e in out.items()}
