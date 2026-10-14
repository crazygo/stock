"""Daily v91 fits with support-bounded, same-side/same-clock probability corrections."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v91'))
import calibration91 as previous
from calibration91 import *
from projection92 import project,centered
SOURCE=Path(__file__).resolve().parent
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v92'
PREVIOUS=ROOT/'.cache/soxl_soxs_14d5pct_v91/runs/development'
BMODELS={'bounded_fundamental':'eligible_fundamental','bounded_reaction':'eligible_reaction','top_bounded_fundamental':'top_fundamental'}
ROUTES=list(BMODELS)+previous.ROUTES+['bounded_premarket_only','bounded_double_cost']


def verify_parent():
    hashes=previous.verify_parent();m=read(previous.SOURCE/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(previous.SOURCE/f)==h,f
    for f,h in m['cache_files_sha256'].items():assert sha(PREVIOUS/f)==h,f
    v=read(PREVIOUS/'verification.json');assert v['passed'] and v['results_sha256']==sha(PREVIOUS/'results.json')
    hashes['91']=sha(previous.SOURCE/'evidence/delivery_manifest.json');return hashes


def raw_correction(model,x,side,clock):
    z=np.r_[1.,side,np.eye(8)[clock,1:],((np.asarray(x)-model['center'])/model['scale'])[INDICES[model['mode']]]]
    return centered(z@model['beta'])


def supports(daily,panel,history):
    if not daily['ready']:return {}
    parents=[{**p,'members':[pd.Timestamp(t) for t in p['members']]} for p in daily['parents']]
    arrays={pop:samples(panel,parents,history,pop) for pop in ['eligible','top']};out={}
    for name,parent_name in BMODELS.items():
        model=daily['models'][parent_name];pop,mode=MODELS[parent_name]
        P,y,w,X,sides,clocks,meta=arrays[pop];Z,center,scale=design(X,sides,clocks,w,mode)
        np.testing.assert_allclose(center,model['center'],atol=1e-12);np.testing.assert_allclose(scale,model['scale'],atol=1e-12)
        scores=centered(Z@model['beta']);groups={}
        for side in range(2):
            for clock in range(8):
                ids=np.flatnonzero((sides==side)&(clocks==clock));members=[meta['members'][i] for i in ids]
                dates=sorted({str(pd.Timestamp(t).tz_convert(core.ET).date()) for t,j in members})
                groups[f'{side}:{clock}']=dict(corrections=scores[ids],members=members,dates=dates)
        out[name]=groups
    return out


def daily_models(panel,parents,feedback,history,at):
    base=previous.daily_models(panel,parents,feedback,history,at)
    return dict(base=base,supports=supports(base,panel,history))


def apply_correction(P,correction):
    return probabilities(np.asarray(P).reshape(1,4),np.ones((1,1)),np.asarray(correction).reshape(1,4))[0]


def issue(at,row,daily,old,base,wait):
    out=previous.issue(at,row,daily['base'],old,base,wait);src=old['events']['recent_ratio']
    for name,parent_name in BMODELS.items():
        e=dict(out[parent_name])
        if e['ready']:
            side=core.ETFS.index(e['top_symbol']);clock=CLOCKS.index(row.clock);x=state(row,side)
            model=daily['base']['models'][parent_name];group=daily['supports'][name][f'{side}:{clock}']
            info=project(raw_correction(model,x,side,clock),group['corrections'])
            q=apply_correction(src['probabilities'][side],info['correction']);value=float(q@np.array(base['distributions']['gap']['payoff_means'][side]))
            buy=e['source_eligible'] and value>1e-12
            e.update(probabilities=q,value=value,action=e['top_symbol'] if buy else 'CASH',allocation=float(buy),
                projection={**info,'context_dates':len(group['dates']),'parent_model':parent_name})
        out[name]=e
    out['bounded_premarket_only']=dict(out['bounded_fundamental']) if row.clock==CLOCKS[0] else out['cash']
    out['bounded_double_cost']=dict(out['bounded_fundamental'])
    return out
