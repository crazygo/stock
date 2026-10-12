"""Causal mature-path overlap weights for base learning and date calibration."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v40'))
from nonlinear_direction import *
from weighted_base import fit_weighted
from verify_continuation import key
SOURCE=Path(__file__).resolve().parent
VARIANTS=['unique','base_only','cal_only','baseline']
ROUTES=['unique_band','unique_point','base_only_band','cal_only_band','baseline_band','source_v35','premarket_only','cash','double_cost']


def attach_spans(panel,labels,market):
    grid=pd.DatetimeIndex([],tz='UTC')
    for d in market.sessions:
        s=market.calendar[d];grid=grid.append(pd.date_range(s['open_at'],s['close_at'],freq='5min',inclusive='right'))
    assert grid.is_unique and grid.is_monotonic_increasing
    out=panel.copy();out['span_start']=-1;out['span_end']=-1;receipts=[]
    for at,row in panel.iterrows():
        if pd.isna(row.terminal_release):continue
        pair=[labels[key(at,s)] for s in core.ETFS]
        assert all(z['entry_at'] is not None and z['exit_at'] is not None for z in pair)
        starts=[pd.Timestamp(z['entry_at']) for z in pair];ends=[pd.Timestamp(z['exit_at']) for z in pair]
        assert starts[0]==starts[1] and starts[0]>=at and min(ends)>starts[0]
        assert row.terminal_release>max(ends)
        left=int(grid.searchsorted(starts[0],side='right'));right=int(grid.get_indexer([max(ends)])[0]);assert 0<=left<=right
        out.loc[at,['span_start','span_end']]=[left,right]
        receipts.append({'at':at,'entry_at':starts[0],'exits':ends,'start':left,'end':right,'first_bar_end':grid[left],'last_bar_end':grid[right]})
    return out,{'definition':'paired regular5m price-dependence interval, not cash occupation','grid':list(grid),'rows':receipts}


def uniqueness(starts,ends):
    """Difference-array counts, prefix sums of 1/concurrency; selected IDs only."""
    a=np.asarray(starts,dtype=np.int64);b=np.asarray(ends,dtype=np.int64)
    assert a.ndim==b.ndim==1 and len(a)==len(b) and np.all(a>=0) and np.all(b>=a)
    if not len(a):return np.array([],float),{'offset':None,'count_runs':[],'uniqueness':[]}
    offset=int(a.min());a=a-offset;b=b-offset
    delta=np.zeros(int(b.max())+2,dtype=np.int64);np.add.at(delta,a,1);np.add.at(delta,b+1,-1)
    counts=np.cumsum(delta)[:-1];inv=np.divide(1.,counts,out=np.zeros(len(counts)),where=counts>0);prefix=np.r_[0.,np.cumsum(inv)]
    u=(prefix[b+1]-prefix[a])/(b-a+1)
    assert np.all((u>0)&(u<=1+1e-12))
    changes=np.r_[0,np.flatnonzero(np.diff(counts))+1,len(counts)]
    runs=[[int(counts[l]),int(r-l)] for l,r in zip(changes[:-1],changes[1:])]
    return u,{'offset':offset,'count_runs':runs,'uniqueness':u.tolist()}


def weights(n,half,unique=None):
    w=2.**(-np.arange(n)[::-1]/half)
    if unique is not None:w=w*np.asarray(unique)
    if n:w*=n/w.sum()
    return w


def fit_bases(panel,at):
    same=panel[panel.clock==panel.loc[at,'clock']];dates,fb=training_dates(same,day_cutoff(at));days=dates['early'];rows=same[same.date.isin(days)]
    u,receipt=uniqueness(rows.span_start,rows.span_end);cols=columns(panel.loc[at,'clock'])
    a,ai=fit_weighted(same,days,cols,weights(len(days),126.,u));b,bi=fit_clock(same,days,cols)
    models={'unique':a,'base_only':a,'cal_only':b,'baseline':b};fits={'unique':ai,'base_only':ai,'cal_only':bi,'baseline':bi};out={};raws={}
    for k in VARIANTS:
        raw=predict(models[k],same.loc[[at]]);out[k]={'fit':fits[k],'feedback':fb,'raw':raw}
        if k in ['unique','base_only']:out[k]['overlap']=receipt
        raws[k]=np.array([raw[s] for s in core.ETFS]) if raw else None
    return models,out,raws


def common_history(histories):
    return {t:histories['baseline'][t] if all(t in histories[k] and histories[k][t] is not None for k in VARIANTS) else None for t in sorted(histories['baseline'])}


def fit_calibration(panel,histories,reg,at):
    parents,fb=pool(reg,common_history(histories),at);ids=[t for p in parents for t in p['members']];n=len(parents)
    d={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'feedback':fb,'parents':parents,'members':ids,'ready':n>=60,'variants':{}}
    if n<60:return d
    starts=[int(panel.loc[p['members'],'span_start'].min()) for p in parents];ends=[int(panel.loc[p['members'],'span_end'].max()) for p in parents]
    u,receipt=uniqueness(starts,ends);groups=np.repeat(np.arange(n),[len(p['members']) for p in parents])
    truth=panel.loc[ids,['C','D']].to_numpy(float);Ytrue=np.column_stack([truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]])
    counts,starts_draw,seed=block_counts(n,at,10);d.update(row_groups=groups,block_starts=starts_draw,seed=seed,overlap=receipt,span_starts=starts,span_ends=ends)
    for k in VARIANTS:
        dw=weights(n,63.,u if k in ['unique','cal_only'] else None)
        w=np.concatenate([np.repeat(dw[j]/len(p['members']),len(p['members'])) for j,p in enumerate(parents)])
        X=encode(panel,histories[k],ids);Y=Ytrue-X[:,:2]
        d['variants'][k]={'day_weights':dw,'row_weights':w,'point_model':fit_map(X,Y,w,'partial',n),
            'bootstrap_models':[fit_map(X,Y,w*c[groups],'partial',n) for c in counts]}
    return d


def issue(panel,at,raws,daily):
    out={}
    for k in VARIANTS:
        raw=raws[k];ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];info={'ready':bool(ready),'point':None,'lower':None}
        if ready:
            x=encode(panel,{at:raw},[at])[0];m=daily['variants'][k];point=raw+forecast(m['point_model'],x)[0]
            draws=np.array([raw+forecast(z,x)[0] for z in m['bootstrap_models']])
            radius=max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
            info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
        out[k]=info
    return out


def events(at,date,clock,estimates,fb,old):
    primary=decide(estimates['unique'],fb,True);cash={'ready':True,'action':'CASH','allocation':0.}
    out={'unique_band':primary,'unique_point':decide(estimates['unique'],fb,False),
        'base_only_band':decide(estimates['base_only'],fb,True),'cal_only_band':decide(estimates['cal_only'],fb,True),
        'baseline_band':decide(estimates['baseline'],fb,True),'source_v35':old,
        'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in out.items()}
